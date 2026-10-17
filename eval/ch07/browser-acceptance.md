# Chapter 7 browser checks

2026-10-16: real installed Microsoft Edge, headless, 1280×850; Playwright 1.58.2.
Command: `PLAYWRIGHT_MODULE=<installed playwright path> node scripts/validate_ch07_browser.cjs`.

The real application HTML was served over local HTTP with deterministic list/detail/SSE fixtures. No paid model calls or customer writes.

- First-question preview and summary marker: passed.
- Switching back loads original user and assistant text: passed.
- A's late response cannot overwrite B after switching: passed.
- Starting a new conversation preserves old sidebar entries: passed.
- List failure stays silent and subsequent chat succeeds: passed.

RED: existing page had no summary marker (0 instead of 1). GREEN after implementation. One test repair replaced body.textContent with innerText: textContent included the inline script's error string even when no error was visible.

API ownership and original-message storage were checked separately against disposable real MySQL databases. This fixture browser run does not establish live model accuracy or the full 24-turn acceptance criteria.
