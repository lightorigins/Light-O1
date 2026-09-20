from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_browser_uses_system_fonts_without_distributing_fonts() -> None:
    css = (REPO / "webui/src/styles.css").read_text()
    assert "@font-face" not in css
    assert "system-ui" in css
    for directory in (REPO / "webui/public", REPO / "server/webui"):
        assert not [path for path in directory.rglob("*") if path.suffix in {".woff", ".woff2", ".ttf", ".otf"}]
    assert not (REPO / "third_party/licenses/INTER-OFL.txt").exists()


def test_three_notice_is_in_the_packaged_webui() -> None:
    original = (REPO / "third_party/licenses/THREE-MIT.txt").read_bytes()
    packaged = REPO / "server/webui/licenses/THREE-MIT.txt"
    assert packaged.is_file()
    assert packaged.read_bytes() == original
    notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text()
    assert "Three.js" in notices
    assert "## Inter" not in notices
