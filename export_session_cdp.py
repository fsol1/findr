"""
export_session_cdp.py

Alternative to export_session.py that avoids automating the login step
entirely. Instead of driving a Playwright-launched browser to x.com/login
(which is exactly what trips Google/X bot detection), this connects to a
REAL Chrome window that you log into completely manually, then just reads
the resulting session out of it.

How it works
------------
1. You launch your everyday Chrome from the terminal with a debug port open
   and a throwaway profile directory (so it doesn't touch your normal
   Chrome profile/cookies).
2. You log into X in that window like any normal human — no automation
   touches this step at all, so there's nothing for detection to flag.
3. This script attaches to that already-running Chrome via CDP and exports
   its storage_state (cookies + localStorage) to x_session.json.
4. agent.py uses that file exactly as before.

Step 1 — quit Chrome completely first, then run in a terminal:

    # macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \\
        --remote-debugging-port=9222 \\
        --user-data-dir="$HOME/chrome-debug-profile"

A new Chrome window opens using a separate profile. Log into X in it
manually (2FA etc. as normal). Once you're on your home feed, come back
here and run:

    python export_session_cdp.py

Leave the Chrome window open until this script finishes.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

SESSION_FILE = Path(__file__).parent / "x_session.json"
CDP_URL = "http://localhost:9222"


async def main():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp(CDP_URL)
        except Exception as e:
            raise SystemExit(
                f"Couldn't connect to Chrome at {CDP_URL} ({e}).\n"
                "Make sure you launched Chrome with --remote-debugging-port=9222 "
                "and that it's still running."
            )

        # Use whichever context/tab is already open and logged in
        context = browser.contexts[0]
        pages = context.pages

        # Sanity check: make sure something looks logged into x.com
        x_page = next((pg for pg in pages if "x.com" in pg.url or "twitter.com" in pg.url), None)
        if x_page is None:
            print(
                "Warning: no open tab on x.com detected. Make sure you're logged "
                "in and on x.com in the debug Chrome window before running this."
            )

        await context.storage_state(path=str(SESSION_FILE))
        print(f"Session saved to {SESSION_FILE}")

        # Don't close the browser — it's your real Chrome window, just detach
        await browser.close()  # this only closes the CDP connection, not Chrome itself


if __name__ == "__main__":
    asyncio.run(main())