from __future__ import annotations

import json
import re
from pathlib import Path

from light_deploy.action_tokenizer.representation import JOINT_PARENTS, NUM_JOINTS

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBUI = REPO_ROOT / "webui"


def test_public_webui_uses_runtime_service_generation_and_sonic_apis() -> None:
    api = (WEBUI / "src" / "api.js").read_text(encoding="utf-8")

    assert "/api/service/status" in api
    assert "/api/service/start" in api
    assert "/api/service/stop" in api
    assert "/api/generations" in api
    assert "/sonic/sim" in api
    assert "/api/approve" not in api
    assert "/api/send" not in api
    assert "/api/humanoid-mesh" not in api


def test_public_webui_exposes_approved_model_and_sampling_controls() -> None:
    index = (WEBUI / "index.html").read_text(encoding="utf-8")

    expected = {
        "data-model-path",
        'name="reasoning_token_budget"',
        'name="reasoning_temperature"',
        'name="reasoning_top_p"',
        'name="action_min_seconds"',
        'name="action_max_seconds"',
        'name="action_temperature"',
        'name="action_top_p"',
    }
    assert all(marker in index for marker in expected)
    assert 'name="reasoning_token_budget" type="number" value="384"' in index
    assert 'name="reasoning_temperature" type="number" value="0.6"' in index
    assert 'name="reasoning_top_p" type="number" value="0.95"' in index
    assert 'name="action_temperature" type="number" value="0.6"' in index
    assert 'name="action_top_p" type="number" value="0.95"' in index
    assert "top_k" not in index.lower()


def test_public_preview_uses_standalone_mannequin_with_skeleton_fallback() -> None:
    preview = (WEBUI / "src" / "preview.js").read_text(encoding="utf-8")
    sources = "\n".join(path.read_text(encoding="utf-8") for path in sorted((WEBUI / "src").glob("*.js")))

    assert "payload?.positions" in preview
    assert "payload?.parents" in preview
    assert "MannequinAvatar" in preview
    assert 'this.skeleton.visible = !showMesh' in preview
    assert "humanoidViewer" not in preview
    assert "/api/humanoid-mesh" not in sources
    assert not (WEBUI / "src" / "humanoid_viewer.js").exists()
    assert not (REPO_ROOT / "assets" / "humanoid").exists()
    assert not (REPO_ROOT / "robot_gateway").exists()


def test_public_preview_fallback_matches_versioned_humanoid22_topology() -> None:
    preview = (WEBUI / "src" / "preview.js").read_text(encoding="utf-8")
    parents_match = re.search(r"const DEFAULT_PARENTS = (\[[^;]+\]);", preview, flags=re.DOTALL)
    standby_match = re.search(r"const STANDBY_POSE = (\[[^;]+\]);", preview, flags=re.DOTALL)

    assert parents_match is not None
    assert standby_match is not None
    parents = json.loads(parents_match.group(1))
    standby = json.loads(standby_match.group(1))
    assert tuple(parents) == JOINT_PARENTS
    assert len(standby) == NUM_JOINTS
    assert all(len(position) == 3 for position in standby)


def test_public_webui_has_sonic_action_without_robot_approval_copy() -> None:
    html = (WEBUI / "index.html").read_text(encoding="utf-8").lower()
    javascript = (WEBUI / "src" / "main.js").read_text(encoding="utf-8").lower()

    assert "sonic mujoco" in html
    assert "data-sonic" in html
    assert "approve &amp; execute" not in html
    assert "approveandsend" not in javascript
    assert "robotprofile" not in javascript


def test_public_webui_groups_primary_actions_and_defaults_reasoning_on() -> None:
    index = (WEBUI / "index.html").read_text(encoding="utf-8")
    javascript = (WEBUI / "src" / "main.js").read_text(encoding="utf-8")
    composer_start = index.index('class="composer-tools"')
    composer = index[composer_start : index.index("</section>", composer_start)]
    toggle_start = composer.index('<button type="button" role="switch"')
    thinking_toggle = composer[toggle_start : composer.index("</button>", toggle_start)]

    assert 'class="composer-options"' in composer
    assert 'class="composer-actions"' in composer
    assert composer.index("data-thinking-toggle") < composer.index("data-generate")
    assert composer.index("data-generate") < composer.index("data-sonic")
    assert 'aria-checked="true"' in thinking_toggle
    assert "data-thinking-state" in index
    assert "data-preview-mode" not in index
    assert "Policy tracking · G1 embodiment · video output" not in index
    assert "when thinking mode is enabled" not in index
    assert 'enable_thinking: elements.thinkingToggle.getAttribute("aria-checked") === "true"' in javascript
    assert 'elements.thinkingToggle.addEventListener("click"' in javascript


def test_public_webui_is_text_only() -> None:
    index = (WEBUI / "index.html").read_text(encoding="utf-8")
    javascript = (WEBUI / "src" / "main.js").read_text(encoding="utf-8")

    assert 'type="file"' not in index
    assert "data-images" not in index
    assert "Add images" not in index
    assert "vision" not in index.lower()
    assert "elements.images" not in javascript


def test_public_preview_matches_the_sonic_camera_convention() -> None:
    preview = (WEBUI / "src" / "preview.js").read_text(encoding="utf-8")

    assert "const SONIC_CAMERA_AZIMUTH_DEGREES = 135;" in preview
    assert "const SONIC_CAMERA_ELEVATION_DEGREES = -15;" in preview
    assert "const SONIC_CAMERA_DISTANCE = 3.0;" in preview
    assert "const SONIC_CAMERA_LOOK_AT_HEIGHT = 0.8;" in preview
    assert "SONIC_CAMERA_AZIMUTH_DEGREES + 180" in preview
    assert "-SONIC_CAMERA_ELEVATION_DEGREES" in preview
    assert "this.orbit.target.set(root.x, SONIC_CAMERA_LOOK_AT_HEIGHT, root.z)" in preview


def test_public_video_sync_cleans_previous_playback_and_has_animation_frame_fallback() -> None:
    javascript = (WEBUI / "src" / "main.js").read_text(encoding="utf-8")
    start = javascript.index("async function runSonic()")
    run_sonic = javascript[start : javascript.index("elements.serviceStart.addEventListener", start)]

    assert "let videoFrameCallbackKind = null;" in javascript
    assert "window.requestAnimationFrame" in javascript
    assert "window.cancelAnimationFrame" in javascript
    assert "stopSonicPlayback();" in run_sonic


def test_vite_dev_server_proxies_public_runtime_api() -> None:
    config = (WEBUI / "vite.config.js").read_text(encoding="utf-8")

    assert '"/api": "http://127.0.0.1:8090"' in config
    assert '"/runtime"' not in config


def test_candidate_picker_switches_the_rendered_skeleton() -> None:
    main = (WEBUI / "src" / "main.js").read_text(encoding="utf-8")

    assert "preview.setActions(state.previews.slice(0, 1), 20)" in main
    assert "secondPreview.setActions(state.previews.slice(1), 20)" in main
    assert "secondPreview.seek(frame)" in main
    assert "`${generation.num_frames} frames · compact output`" in main


def test_public_webui_build_is_packaged_with_server() -> None:
    assert (REPO_ROOT / "server" / "webui" / "index.html").is_file()
    assert any((REPO_ROOT / "server" / "webui" / "assets").glob("*.js"))


def test_public_webui_uses_the_approved_inference_workspace_hierarchy() -> None:
    index = (WEBUI / "index.html").read_text(encoding="utf-8")

    expected = {
        "data-control-workspace",
        "data-activity-panel",
        "data-prompt-composer",
        "data-inference-output",
        "data-skeleton-panel",
        "data-sonic-panel",
        "Human Action View",
        "G1 · Sonic MuJoCo",
        "Action Representation",
    }
    assert all(marker in index for marker in expected)
    assert index.index("data-control-workspace") < index.index("data-activity-panel")
    assert index.index("data-prompt-composer") < index.index("data-inference-output")
    assert index.index("data-inference-output") < index.index("data-skeleton-panel")
    assert index.index("data-skeleton-panel") < index.index("data-sonic-panel")


def test_loaded_model_metadata_lives_with_checkpoint_controls() -> None:
    index = (WEBUI / "index.html").read_text(encoding="utf-8")

    checkpoint = index.index("data-model-path")
    loaded_model = index.index("data-loaded-model")
    representation = index.index("data-action-representation")
    service_section_end = index.index("</section>", representation)
    assert checkpoint < loaded_model < representation < service_section_end
    assert "Reasoning text-to-action · human_action_138_v1" not in index
    assert "humanoid22_v1 viewport" not in index


def test_reasoning_and_action_stats_precede_the_visual_outputs() -> None:
    index = (WEBUI / "index.html").read_text(encoding="utf-8")

    summary = index.index("data-inference-output")
    reasoning = index.index("data-reasoning", summary)
    frames = index.index("data-runtime-frames", summary)
    duration = index.index("data-runtime-duration", summary)
    skeleton = index.index("data-skeleton-panel")
    assert summary < reasoning < skeleton
    assert summary < frames < skeleton
    assert summary < duration < skeleton


def test_public_webui_always_renders_sonic_and_logs_results() -> None:
    index = (WEBUI / "index.html").read_text(encoding="utf-8")
    javascript = (WEBUI / "src" / "main.js").read_text(encoding="utf-8")

    assert "data-sonic-render" not in index
    assert "data-sonic-result" not in index
    assert "sonicRender" not in javascript
    assert "render:" not in javascript
    assert "logSonicResult" in javascript


def test_public_preview_uses_threejs_without_private_mesh_assets() -> None:
    package = json.loads((WEBUI / "package.json").read_text(encoding="utf-8"))
    preview = (WEBUI / "src" / "preview.js").read_text(encoding="utf-8")

    assert package["dependencies"]["three"]
    assert 'from "three"' in preview
    assert "WebGLRenderer" in preview
    assert "preserveDrawingBuffer: true" in preview
    assert "SphereGeometry" in preview
    assert "SkinnedMesh" not in preview
    assert "GLTFLoader" not in preview


def test_public_preview_does_not_redraw_while_idle() -> None:
    preview = (WEBUI / "src" / "preview.js").read_text(encoding="utf-8")
    loop_body = preview.split("  loop(time) {", maxsplit=1)[1].split("\n  destroy()", maxsplit=1)[0]

    assert "\n      this.render();" in loop_body
    assert "\n    this.render();" not in loop_body
