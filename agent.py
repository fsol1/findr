"""
narrative_scanner/agent.py

An LLM-driven agent that plans searches, browses X (Twitter) like a logged-in
human via crawl4AI + a persisted browser session, evaluates what it finds,
and logs promising "memecoin narrative" candidates to disk.

Architecture
------------
- Claude does the THINKING: it decides what to search, reads back what the
  crawler found, judges whether it's promising, and decides the next move.
- crawl4AI does the FETCHING: it drives a real (headless) Chromium browser
  using your logged-in session cookies, scrolls like a human, and returns
  clean markdown of whatever renders on screen.
- Three tools are exposed to Claude: `search_x`, `read_url`, `log_finding`.
  Claude calls these via normal Anthropic tool-use; you don't hardcode the
  query list.

What's new in this version
---------------------------
1. **tokens.json profile**: a JSON file of example narratives/tokens you
   consider "the kind of thing I'm looking for" (and optionally ones you're
   NOT interested in). Loaded once, summarized, and folded into the system
   prompt so the agent's search taste is grounded in your actual examples
   instead of generic instructions.
2. **lessons.json**: a small growing file of "here's a pattern that turned
   out to be a false positive / dead end, and why" entries. The agent reads
   it in the system prompt at the start of every run, and can append to it
   mid-run via a `log_lesson` tool when it changes its mind about something
   it initially flagged (self-critique), so repeated runs get less naive
   over time. You can also hand-edit this file after the fact (e.g. "the
   FROG_XYZ finding from Tuesday never went anywhere, here's why") and it
   will be picked up next run.
3. **Token optimization**:
   - System prompt + tool defs + profile/lessons are marked with
     `cache_control` (prompt caching) since they're identical every round —
     this is normally the largest fixed cost in a multi-round tool-use loop.
   - Tool output truncation is tighter and strips obvious boilerplate before
     truncating, instead of dumping raw markdown.
   - Old rounds get compacted: once the transcript passes a size threshold,
     earlier tool_result blocks are collapsed to a one-line placeholder
     (the model already made its decision on them; it doesn't need the full
     text of page 1's search results by round 6).

Explicitly NOT included: automated liking/following/clicking to shape the
account's home feed. That's fake-engagement / algorithm manipulation from a
bot account, which is a different (and worse) thing than automated reading,
and I'm not willing to build it even with human-like timing. If you want the
account's feed to reflect a niche, do that curation yourself as a human.

Setup required before running
------------------------------
1. pip install crawl4ai anthropic
   playwright install chromium
2. Log into x.com in a real browser, then export that session so crawl4AI
   can reuse it (see export_session.py in this folder for one way to do
   this from Chrome/Playwright). You'll end up with a JSON file
   like `x_session.json` (Playwright "storage_state" format).
3. export ANTHROPIC_API_KEY=...   (put this in your shell env or a .env
   file that's gitignored — never commit it)
4. Optionally create `tokens.json` (see PROFILE FORMAT below) and
   `lessons.json` (created automatically once the agent logs its first
   lesson).
5. Use a burner/secondary account for this, not your main — sustained
   automated browsing is against X's ToS and accounts can get flagged or
   locked regardless of how human-like the behavior is.

PROFILE FORMAT (tokens.json)
-----------------------------
A JSON array of example objects. Any shape is fine as long as each entry has
a short human-readable description; the loader just serializes and truncates
them. Suggested shape:

[
  {
    "name": "PONKE",
    "why": "started as a single low-effort meme image, got picked up by 3+
            unrelated large accounts within 48h before any token existed,
            simple one-word clonable name",
    "positive": true
  },
  {
    "name": "some-forgettable-thing",
    "why": "high engagement but single-account, no cross-pickup, name
            wasn't distinct enough to clone",
    "positive": false
  }
]

`positive: false` examples are just as useful as positive ones — they teach
the agent what NOT to chase.

Usage
-----
    python agent.py --seed "viral non-crypto moments today that could become a coin" --rounds 6

Good seeds describe the KIND of raw material to hunt for, not crypto terms —
e.g. "viral animals and zoo events today", "a phrase or slang spreading in
replies this week", "scam/grift stories people are dunking on", "github
projects blowing up outside their normal audience". Avoid seeds like
"emerging memecoin narratives" — that steers the model back toward searching
for coins that already exist.

Output
------
Findings are appended as JSON lines to `findings.jsonl` in this folder, a
`seen_urls.json` file prevents re-surfacing the same post across runs, and
`lessons.json` accumulates self-critique the agent can reuse next time.
"""

import argparse
import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai import UndetectedAdapter
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

HERE = Path(__file__).parent
SESSION_FILE = HERE / "x_session.json"
FINDINGS_FILE = HERE / "findings.jsonl"
SEEN_FILE = HERE / "seen_urls.json"
TOKENS_FILE = HERE / "tokens.json"
LESSONS_FILE = HERE / "lessons.json"

MODEL = "claude-haiku-4-5"

# Keep the fixed part of the transcript from growing unboundedly. Once the
# running message list exceeds this many rounds, older tool_result blocks
# get collapsed (see compact_old_rounds()).
COMPACT_AFTER_ROUNDS = 4
TOOL_OUTPUT_CHAR_LIMIT = 4000  # was 6000; tighter now that we also strip boilerplate

BASE_SYSTEM_PROMPT = """You are a research agent scanning X (Twitter) for raw \
material that DOESN'T have a token yet but has the shape of something that \
could — a pre-memecoin, not a memecoin. You are NOT searching for existing \
tokens, ticker chatter, or crypto-native language. The best finds usually \
have nothing to do with crypto at all until someone makes them into a coin.

The shape you're looking for, regardless of category:
- A single, instantly-legible visual or verbal hook — something you could
  describe to a stranger in five words and they'd picture it exactly.
- A name or phrase that's short, spellable, and already forming around it
  organically (a nickname the replies gave it, a caption that's getting
  reused, a misheard word).
- Spread that's outward and organic, not one account's follower count —
  multiple unrelated accounts picking it up, quote-tweets adding jokes,
  it crossing from one community into another.
- Early. If it's already inescapable or already has fan accounts and merch,
  it's late — you want the thing three days before that, not the thing
  during that.

Categories to actively hunt across (this list is illustrative, not
exhaustive — branch into whatever's actually moving today):
- Viral animals / animal events: a zoo birth, an animal doing something
  unexpected, a pet with a distinct look or name.
- News/current events with a strong single image or phrase, stripped of
  political charge — the internet's reaction/meme layer on top of an event,
  not the event's substance.
- A phrase, typo, or slang term that's visibly spreading and getting reused
  as a caption/reaction rather than staying inside one post.
- A viral screenshot format, a scam/grift story people are dunking on, an
  awkward DM, an AI-generation fail — anything becoming a template.
- A GitHub repo, product launch, or tool that's blowing up outside its
  normal audience because of a name or logo, not its function.
- An account/persona having a breakout moment (a reply-guy going viral, a
  random person's post getting main-charactered for a day).

You have three tools:
- search_x(query): runs a live search on X and returns rendered results \
(post text, author, rough engagement, links) as markdown.
- read_url(url): opens a specific X post/thread/profile and returns its \
rendered content, so you can check replies, quote-tweets, or an author's \
other recent posts.
- log_finding(...): records a candidate worth tracking.

Work iteratively:
1. Decide what to search for. Do NOT search crypto/memecoin-flavored terms
   ("memecoin", "narrative", ticker-style names, "$"-prefixed anything) as
   your primary strategy — that surfaces coins that already exist, which is
   the opposite of the goal. Instead search the underlying cultural moment:
   plain-language descriptions of what's happening ("baby elephant zoo
   born", "guy scammed by", specific phrasing you see repeating, an animal's
   name once you learn it, a repo name), general virality trackers (trending
   topics, "this you" style reply chains, quote-tweet volume), and follow-up
   searches once you've found the actual name/phrase people gave a thing.
2. After each tool result, decide: is this a real candidate, noise, or does
   it need one more check (e.g. read_url on the thread to see if it's
   spreading, or a search on the specific name/phrase once you know it, to
   see if it's crossed into other communities)?
3. When you find something worth logging, call log_finding with a concise
   structured record — include what the hook is, what name/phrase (if any)
   is already forming, and evidence of cross-account spread, not just raw
   engagement numbers.
4. Keep going for the number of rounds you're given, favoring breadth
   (new angles, new categories) over repeating similar searches.
5. If, based on a later result, you realize an earlier assumption or search
   angle was wrong (e.g. a whole category you searched turned out to be
   stale reposts, or a pattern you initially found looked promising and then
   revealed itself as an obvious single-account push with no legs), call
   log_lesson so future runs don't repeat the mistake. Don't log a lesson for
   every dead-end search — only genuine "I was wrong to think X because Y"
   moments.

Be skeptical by default — most viral posts are just viral posts, not
narrative-with-legs, and most viral posts are ALREADY too crypto-flavored or
already too late by the time they'd show up in a "memecoin" search. Only
log_finding for things with a real reason (cross-account pickup, a name/hook
that's clonable, visible early acceleration) and where nobody has obviously
tokenized it yet — check for that before logging.

Calibrate against the taste profile and past lessons below — they're worth
more than generic instincts about what's "viral."
"""

TOOLS = [
    {
        "name": "search_x",
        "description": "Search X (Twitter) live search as a logged-in user and return rendered results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'raccoon viral' or a hashtag"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": "Open a specific X URL (post, thread, or profile) and return its rendered content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full x.com URL to open"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "log_finding",
        "description": "Record a candidate narrative worth tracking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "urls": {"type": "array", "items": {"type": "string"}},
                "why_promising": {"type": "string"},
            },
            "required": ["title", "description", "urls", "why_promising"],
        },
    },
    {
        "name": "log_lesson",
        "description": (
            "Record a self-critique: a pattern or assumption that turned out to be "
            "wrong during this run, so future runs don't repeat it. Only call this "
            "for genuine reversals, not routine dead ends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The pattern/assumption that turned out wrong, e.g. 'high like count on a single quote-tweet chain'",
                },
                "why_wrong": {"type": "string", "description": "What you learned instead"},
            },
            "required": ["pattern", "why_wrong"],
        },
    },
]


# --------------------------------------------------------------------------
# Profile / lessons loading
# --------------------------------------------------------------------------

def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"[warn] {path.name} isn't valid JSON, ignoring it for this run.")
        return []


def summarize_profile(entries: list, max_entries: int = 40) -> str:
    """Turn tokens.json examples into a compact block for the system prompt.

    Capped at max_entries so a huge tokens.json doesn't blow up every single
    API call's input tokens — this runs once per process, not once per
    round, but it's still worth keeping lean since it's re-sent (from cache)
    every round.
    """
    if not entries:
        return ""
    # If there are more than max_entries, sample across positive/negative
    # rather than just taking the first N, so both signal types survive.
    positives = [e for e in entries if e.get("positive", True)]
    negatives = [e for e in entries if not e.get("positive", True)]
    if len(entries) > max_entries:
        half = max_entries // 2
        positives = positives[:half]
        negatives = negatives[:max_entries - len(positives)]

    lines = ["TASTE PROFILE (examples of what you're actually looking for):"]
    if positives:
        lines.append("\nThings that WERE good candidates:")
        for e in positives:
            name = e.get("name", "?")
            why = e.get("why", "")
            lines.append(f"- {name}: {why}")
    if negatives:
        lines.append("\nThings that LOOKED tempting but WEREN'T:")
        for e in negatives:
            name = e.get("name", "?")
            why = e.get("why", "")
            lines.append(f"- {name}: {why}")
    return "\n".join(lines)


def summarize_lessons(entries: list, max_entries: int = 30) -> str:
    if not entries:
        return ""
    recent = entries[-max_entries:]  # most recent lessons are most relevant
    lines = ["PAST LESSONS (self-critiques from earlier runs — don't repeat these mistakes):"]
    for e in recent:
        lines.append(f"- Pattern: {e.get('pattern', '?')} -> {e.get('why_wrong', '')}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    profile = summarize_profile(load_json_list(TOKENS_FILE))
    lessons = summarize_lessons(load_json_list(LESSONS_FILE))
    parts = [BASE_SYSTEM_PROMPT]
    if profile:
        parts.append(profile)
    if lessons:
        parts.append(lessons)
    return "\n\n".join(parts)


def append_lesson(record: dict):
    lessons = load_json_list(LESSONS_FILE)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    lessons.append(record)
    LESSONS_FILE.write_text(json.dumps(lessons, indent=2))


# --------------------------------------------------------------------------
# Findings / seen-url bookkeeping
# --------------------------------------------------------------------------

def load_seen() -> set:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()


def save_seen(seen: set):
    SEEN_FILE.write_text(json.dumps(sorted(seen)))


def append_finding(record: dict):
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    with FINDINGS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# Crawling
# --------------------------------------------------------------------------

async def human_scroll(page, rounds: int = 4):
    """Scroll down incrementally with randomized pauses, like a real user."""
    for _ in range(rounds):
        await page.mouse.wheel(0, random.randint(600, 1400))
        await asyncio.sleep(random.uniform(1.0, 3.0))


def strip_boilerplate(markdown: str) -> str:
    """Drop obvious repeated chrome (nav links, 'Log in' / 'Sign up' prompts,
    footer link lists) before truncating, so the token budget goes to actual
    post content instead of X's UI furniture. Deliberately conservative —
    only strips lines that are almost certainly not post content.
    """
    if not markdown:
        return markdown
    noisy_exact = {
        "log in", "sign up", "don't miss what's happening",
        "see new posts", "what's happening", "relevant people",
        "trending now", "who to follow", "terms of service",
        "privacy policy", "cookie policy", "accessibility",
        "ads info", "more",
    }
    kept = []
    for line in markdown.splitlines():
        stripped = line.strip().strip("[]() ").lower()
        if stripped in noisy_exact:
            continue
        if len(stripped) < 2:
            continue
        kept.append(line)
    return "\n".join(kept)


async def crawl_x_url(crawler: AsyncWebCrawler, url: str) -> str:
    """Fetch a rendered X page as markdown, scrolling like a human first."""
    run_cfg = CrawlerRunConfig(
        js_code="""
            (async () => {
                for (let i = 0; i < 4; i++) {
                    window.scrollBy(0, 800 + Math.random() * 600);
                    await new Promise(r => setTimeout(r, 1000 + Math.random() * 2000));
                }
            })();
        """,
        wait_for="css:article, [data-testid='primaryColumn']",
        delay_before_return_html=2.0,
    )
    result = await crawler.arun(url=url, config=run_cfg)
    if not result.success:
        return f"[fetch failed: {result.error_message}]"
    # fit_markdown is the cleaned/boilerplate-stripped version when available
    content = getattr(result, "fit_markdown", None) or result.markdown or "[no content extracted]"
    return strip_boilerplate(content)


# --------------------------------------------------------------------------
# Token-optimization helpers
# --------------------------------------------------------------------------

def truncate_output(output: str) -> str:
    if isinstance(output, str) and len(output) > TOOL_OUTPUT_CHAR_LIMIT:
        return output[:TOOL_OUTPUT_CHAR_LIMIT] + "\n...[truncated]"
    return output


def compact_old_rounds(messages: list, current_round: int):
    """Collapse tool_result content from rounds older than COMPACT_AFTER_ROUNDS
    ago into a one-line placeholder. The model already acted on that
    information (it either logged a finding, moved on, or asked a follow-up
    read_url) — keeping the full page dump around forever just burns input
    tokens every subsequent round for no benefit. We keep assistant text/tool
    calls intact since those preserve the reasoning trail; only the bulky
    tool_result payloads get collapsed.

    Mutates `messages` in place.
    """
    if current_round < COMPACT_AFTER_ROUNDS:
        return
    cutoff_idx = len(messages) - (COMPACT_AFTER_ROUNDS * 2)  # rough: 2 msgs/round
    for msg in messages[:cutoff_idx]:
        if msg["role"] != "user" or not isinstance(msg["content"], list):
            continue
        for block in msg["content"]:
            if block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                if not block["content"].startswith("[collapsed"):
                    block["content"] = "[collapsed older tool result — already evaluated]"


# --------------------------------------------------------------------------
# Main agent loop
# --------------------------------------------------------------------------

async def run_agent(seed: str, rounds: int):
    if not SESSION_FILE.exists():
        raise SystemExit(
            f"Missing {SESSION_FILE}. Export your logged-in X session (Playwright "
            "storage_state JSON) before running — see the module docstring."
        )

    anthropic = Anthropic()  # reads ANTHROPIC_API_KEY from env
    seen = load_seen()
    system_prompt = build_system_prompt()

    browser_cfg = BrowserConfig(
        headless=True,
        storage_state=str(SESSION_FILE),
        user_agent_mode="random",  # vary UA per session rather than a static default
    )

    # crawl4ai's enable_stealth=True currently no-ops against playwright-stealth 2.x
    # (a known upstream bug), so use UndetectedAdapter instead — it doesn't depend
    # on playwright-stealth and is the documented option for sites with tougher
    # bot detection (which X/Google both are).
    undetected_adapter = UndetectedAdapter()
    crawler_strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=browser_cfg,
        browser_adapter=undetected_adapter,
    )

    async with AsyncWebCrawler(config=browser_cfg, crawler_strategy=crawler_strategy) as crawler:

        async def search_x(query: str) -> str:
            url = f"https://x.com/search?q={query.replace(' ', '%20')}&f=live"
            return await crawl_x_url(crawler, url)

        async def read_url(url: str) -> str:
            return await crawl_x_url(crawler, url)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Seed focus: {seed}\n"
                    f"You have {rounds} tool-call rounds. Start by planning 2-3 "
                    f"initial searches, then adapt based on what you find."
                ),
            }
        ]

        for round_num in range(rounds):
            compact_old_rounds(messages, round_num)

            response = anthropic.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=TOOLS,
                messages=messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )

            messages.append({"role": "assistant", "content": response.content})

            # Surface any plain-text reasoning Claude included alongside tool calls
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\n--- round {round_num + 1} reasoning ---\n{block.text.strip()}")

            usage = response.usage
            cached = getattr(usage, "cache_read_input_tokens", 0) or 0
            print(
                f"[usage] in={usage.input_tokens} cached={cached} "
                f"out={usage.output_tokens}"
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                print("\nAgent stopped calling tools — ending run.")
                break

            tool_results = []
            for tool_use in tool_uses:
                name = tool_use.name
                args = tool_use.input

                if name == "search_x":
                    print(f"[search_x] {args['query']!r}")
                    output = await search_x(args["query"])
                elif name == "read_url":
                    if args["url"] in seen:
                        output = "[already read this URL earlier in the run]"
                    else:
                        print(f"[read_url] {args['url']}")
                        output = await read_url(args["url"])
                        seen.add(args["url"])
                elif name == "log_finding":
                    print(f"[log_finding] {args['title']}")
                    append_finding(args)
                    output = "logged."
                elif name == "log_lesson":
                    print(f"[log_lesson] {args['pattern']}")
                    append_lesson(args)
                    output = "lesson recorded."
                else:
                    output = f"unknown tool {name}"

                output = truncate_output(output)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": output,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

    save_seen(seen)
    print(f"\nDone. Findings appended to {FINDINGS_FILE}")
    print(f"Lessons file: {LESSONS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="LLM-driven X narrative scanner")
    parser.add_argument("--seed", required=True, help="What to focus the search on")
    parser.add_argument("--rounds", type=int, default=6, help="Number of plan/act rounds")
    args = parser.parse_args()
    asyncio.run(run_agent(args.seed, args.rounds))


if __name__ == "__main__":
    main()
