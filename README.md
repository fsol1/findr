crawl4AI et api claude

# narrative_scanner

An LLM-driven agent that browses X (Twitter) like a logged-in human — via
crawl4AI driving a real browser session — to find and evaluate emerging
"memecoin narrative" candidates, rather than following a fixed script of
keyword searches.

## How it works

- **Claude plans and judges.** It decides what to search, reads back the
  rendered results, decides whether something is promising or noise, and
  chooses the next move (broaden, narrow, follow a thread, check an author's
  other posts).
- **crawl4AI fetches.** It drives headless Chromium with your logged-in
  session, scrolls with randomized timing, and returns clean markdown of
  whatever's on screen.
- Findings Claude considers genuinely promising get appended to
  `findings.jsonl`.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
export ANTHROPIC_API_KEY=sk-ant-
```

Then export a logged-in X session once:

```bash
python export_session.py
```

This opens a real browser window — log in manually (with 2FA if you use it),
then press Enter in the terminal once you're on your home feed. This saves
`x_session.json`, which `agent.py` reuses on every run so it never logs in
programmatically.

**Use a secondary/burner account for this**, not your main one. Sustained
automated browsing is against X's Terms of Service even when done to look
human, and accounts doing this can get rate-limited, shadow-limited on
search, or locked. Budget for rotating accounts periodically rather than
relying on one account long-term.

## Running it

```bash
python agent.py --seed "emerging memecoin narratives on crypto twitter today" --rounds 8
```

- `--seed`: what to point the agent at. Can be broad ("scan for viral animal
  clips") or narrow ("follow up on the raccoon video trend").
- `--rounds`: how many plan → search/read → evaluate cycles to run. Each
  round is one Claude call plus however many tool calls it makes.

Output:
- `findings.jsonl` — one JSON record per logged candidate (title,
  description, URLs, why it looked promising, timestamp).
- `seen_urls.json` — URLs already read, so repeated runs don't re-process
  the same posts.

## Extending

- **TikTok / YouTube / Facebook**: add parallel `search_<platform>` /
  `read_url` tool implementations following the same pattern — export a
  session the same way, point crawl4AI's `arun` at the platform's search
  URL, and add the tool to the `TOOLS` list and the dispatch block in
  `agent.py`. TikTok in particular fingerprints beyond cookies (device
  signals), so expect it to degrade faster than X even with a valid session.
- **Velocity tracking**: `findings.jsonl` currently logs one-off snapshots.
  For real narrative-vs-noise signal, re-run periodically (e.g. hourly via
  cron) with the same seed and diff mention counts/follower-weighted
  engagement over time rather than trusting a single pass.
- **Structured extraction**: right now results are fed to Claude as raw
  rendered markdown. If you want consistent fields (post ID, exact
  engagement counts, timestamp) reliably, add a `CrawlerRunConfig` with an
  `extraction_strategy` (crawl4AI supports schema-based or LLM extraction)
  rather than relying on Claude to parse free text every time.
