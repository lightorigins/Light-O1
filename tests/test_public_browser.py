from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBUI_ROOT = REPO_ROOT / "webui"
PACKAGED_WEBUI = REPO_ROOT / "server" / "webui"
FOLLOW_UP_OWNER = "public WebUI release maintainer"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def _browser_executable() -> Path | None:
    configured = os.environ.get("LIGHT_DEPLOY_CHROME")
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    cache = Path.home() / ".cache" / "ms-playwright"
    candidates = sorted(cache.glob("chromium-*/chrome-linux*/chrome"), reverse=True)
    return candidates[0] if candidates else None


def _browser_prerequisites() -> tuple[str, Path]:
    node = shutil.which("node")
    browser = _browser_executable()
    playwright = WEBUI_ROOT / "node_modules" / "playwright-core" / "package.json"
    missing = []
    if node is None:
        missing.append("node")
    if browser is None:
        missing.append("Chromium")
    if not playwright.is_file():
        missing.append("webui/node_modules/playwright-core")
    if missing:
        pytest.skip(
            f"{FOLLOW_UP_OWNER} must install browser E2E prerequisites: {', '.join(missing)}; "
            "run npm ci in webui and install the matching Playwright Chromium"
        )
    assert node is not None and browser is not None
    return node, browser


def test_packaged_public_webui_browser_generation_candidate_and_sonic(tmp_path: Path) -> None:
    node, browser = _browser_prerequisites()
    assert (PACKAGED_WEBUI / "index.html").is_file()
    script = WEBUI_ROOT / "tests" / "public_e2e.mjs"
    screenshot = Path(os.environ.get("LIGHT_DEPLOY_BROWSER_SCREENSHOT", tmp_path / "public-webui.png"))
    handler = partial(_QuietHandler, directory=str(PACKAGED_WEBUI))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/"

    try:
        completed = subprocess.run(
            [str(node), str(script), base_url, str(screenshot), str(browser)],
            cwd=WEBUI_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence == {
        "selected_generation": "g2",
        "action_request": "/api/generations/g2/action",
        "sonic_request": "/api/generations/g2/sonic/sim",
        "canvas_changed": True,
        "webgl_renderer": "three",
        "sonic_render_forced": True,
        "screenshot": str(screenshot),
    }
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert screenshot.stat().st_size > 10_000
