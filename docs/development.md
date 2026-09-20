# Development

```bash
uv sync --extra inference --group dev
uv run ruff check .
uv run --extra inference --group dev python -m pytest -n 1
uv build
```

## Optional test dependencies

Some tests are skipped unless their environment is present:

| Test | Requires |
| :--- | :--- |
| Browser | Node, `npm ci` in `webui/`, and Chromium (`LIGHT_DEPLOY_CHROME` selects the executable) |
| Real simulation | `SONIC_TEST_PYTHON` and `SONIC_TEST_CHECKPOINT` |
| HTTP end-to-end | `LIGHT_DEPLOY_E2E_URL` pointing at a running, GPU-backed Control Server with Sonic configured |

The end-to-end test covers the real text/thinking, image/direct, skeleton and simulation paths. It generates
action but never sends hardware commands.

## Known rough edge

The web console keeps **Stop** disabled during an active job. API clients can stop the session or cancel a
Sonic job explicitly.
