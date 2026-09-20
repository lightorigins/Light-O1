import asyncio
import io
import json
import os
import signal
import sys
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.config import ControlConfig


def test_control_has_one_http_api_not_a_nested_runtime(tmp_path):
    app = create_app(ControlConfig(webui_dist=tmp_path))
    paths = {route.path for route in app.routes}
    assert "/runtime" not in paths
    assert "/api/generations" in paths


def action_result(frames=3):
    action = np.zeros((frames, 138), dtype=np.float32)
    action[:, 2] = 0.9
    action[:, 4:136] = np.tile([1, 0, 0, 0, 1, 0], 22)
    return {
        "schema_version": 1,
        "representation": "human_action_138_v1",
        "fps": 20,
        "num_frames": frames,
        "action": action.tolist(),
        "reasoning": "Wave.",
        "action_token_ids": [10] * frames,
        "action_codebook_ids": [0] * frames,
        "action_token_count": frames,
        "raw_token_count": frames,
        "finish_reason": "action_end",
    }


def test_remote_generation_preview_and_config_do_not_open_checkpoint(tmp_path):
    from server.inference_client import InferenceClient

    requests = []

    def transport(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "model": "remote-model",
                    "representation": "human_action_138_v1",
                    "action_dim": 138,
                    "fps": 20,
                },
            )
        return httpx.Response(200, json=action_result())

    config = ControlConfig(inference_url="http://gpu:8030", webui_dist=tmp_path)
    inference = InferenceClient(config.inference_url, transport=httpx.MockTransport(transport))
    with TestClient(create_app(config, inference_client=inference)) as client:
        assert client.get("/api/service/status").json()["state"] == "ready"
        defaults = client.get("/api/defaults").json()["defaults"]
        assert defaults["mode"] == "remote"
        response = client.post("/api/generations", json={"prompt": "wave"})
        assert response.status_code == 202
        generation_id = response.json()["generation_id"]
        for _ in range(100):
            generation = client.get(f"/api/generations/{generation_id}").json()
            if generation["state"] != "running":
                break
        assert generation["state"] == "done", generation
        assert generation["num_frames"] == 3
        assert len(generation["action_sha256"]) == 64
        preview = client.get(f"/api/generations/{generation_id}/action").json()
        assert np.asarray(preview["positions"]).shape == (3, 22, 3)
        download = client.get(f"/api/generations/{generation_id}/download")
        assert download.status_code == 200
        assert download.headers["cache-control"] == "no-store"
        with zipfile.ZipFile(io.BytesIO(download.content)) as bundle:
            exported = np.load(io.BytesIO(bundle.read("action.npy")), allow_pickle=False)
            np.testing.assert_array_equal(exported, np.asarray(action_result()["action"], dtype=np.float32))
            assert exported.shape == (3, 138)
            metadata = json.loads(bundle.read("generation.json"))
            assert metadata["settings"]["prompt"] == "wave"
            assert "images" not in metadata["settings"]
            assert metadata["action_sha256"] == generation["action_sha256"]
            assert bundle.read("reasoning.txt").decode() == "Wave."
        assert client.post("/api/service/start", json={"model_path": "/remote/ckpt"}).status_code == 409
        assert client.get("/api/generations/missing/action").status_code == 404
        assert client.post("/api/service/stop").status_code == 200
        assert client.get(f"/api/generations/{generation_id}").status_code == 404
        assert client.get(f"/api/generations/{generation_id}/download").status_code == 404
    assert all(request.url.host == "gpu" for request in requests)


def test_generation_store_is_bounded_and_action_immutable():
    from server.generation_store import GenerationStore

    store = GenerationStore(max_items=1)
    first = store.create({"prompt": "wave"})
    with pytest.raises(RuntimeError, match="busy"):
        store.create({"prompt": "another"})
    store.complete(first.generation_id, action_result())
    assert not first.action.flags.writeable
    with pytest.raises(ValueError):
        first.action.setflags(write=True)
    with pytest.raises(RuntimeError, match="completed"):
        store.complete(first.generation_id, action_result())
    second = store.create({"prompt": "second"})
    assert first.generation_id != second.generation_id
    with pytest.raises(KeyError):
        store.get(first.generation_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fps", 30),
        ("representation", "other"),
        ("num_frames", 4),
        ("schema_version", 2),
        ("action", [[0] * 138] * 401),
    ],
)
def test_store_rejects_invalid_remote_contract(field, value):
    from server.generation_store import GenerationStore

    store = GenerationStore()
    job = store.create({"prompt": "wave"})
    result = action_result()
    result[field] = value
    with pytest.raises(ValueError):
        store.complete(job.generation_id, result)


def test_remote_unavailable_and_error_diagnostics(tmp_path):
    from server.inference_client import InferenceClient

    async def check():
        def fail(request):
            raise httpx.ConnectError("private details", request=request)

        async with InferenceClient("http://gpu:8030", transport=httpx.MockTransport(fail)) as inference:
            with pytest.raises(RuntimeError, match="unavailable"):
                await inference.health()

    asyncio.run(check())


def test_managed_process_stops_real_child_and_does_not_claim_readiness(tmp_path):
    from server.process import ManagedProcess

    process = ManagedProcess(tmp_path)
    process.start([sys.executable, "-c", "import time; time.sleep(60)"])
    assert process.running
    child = process.child
    process.stop()
    assert child.poll() is not None
    assert not process.running


def test_control_import_does_not_initialize_vllm():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import server.app, sys; assert 'vllm' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cpu_control_and_fk_work_without_torch_or_vllm():
    import subprocess

    script = """
import importlib.abc, sys
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'vllm', 'mujoco', 'onnxruntime'}:
            raise ImportError('GPU/simulation packages forbidden in Control: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
import server.app
import numpy as np
from server.humanoid_preview import decode_human_action_for_web
action = np.zeros((1,138),dtype=np.float32)
action[:,2] = .9
action[:,4:136] = np.tile([1,0,0,0,1,0],22)
assert len(decode_human_action_for_web(action)['positions']) == 1
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stop_cancels_request_and_prevents_stale_publication(tmp_path):
    from server.inference_client import InferenceClient

    cancelled = []

    async def transport(request):
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "model": "remote",
                    "action_dim": 138,
                    "representation": "human_action_138_v1",
                    "fps": 20,
                },
            )
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    config = ControlConfig(inference_url="http://gpu:8030", webui_dist=tmp_path)
    with TestClient(
        create_app(config, inference_client=InferenceClient(config.api_url, transport=httpx.MockTransport(transport)))
    ) as client:
        first = client.post("/api/generations", json={"prompt": "wave"}).json()
        assert client.post("/api/generations", json={"prompt": "again"}).status_code == 409
        assert client.post("/api/service/stop").json()["state"] == "stopped"
        assert client.get(f"/api/generations/{first['generation_id']}").status_code == 404
        assert client.post("/api/generations", json={"prompt": "again"}).status_code == 503
    assert cancelled == [True]


def test_local_configuration_used_when_start_payload_is_empty(tmp_path, monkeypatch):
    from server.process import ManagedProcess

    commands = []
    original_start = ManagedProcess.start

    def start(self, command):
        commands.append(command)
        original_start(self, [sys.executable, "-c", "import time; time.sleep(60)"])

    monkeypatch.setattr(ManagedProcess, "start", start)
    config = ControlConfig(
        model_path=tmp_path / "model",
        webui_dist=tmp_path,
        max_model_len=777,
        kv_cache_memory_bytes=123456,
        deterministic=True,
        inference_port=18031,
    )
    with TestClient(create_app(config)) as client:
        assert client.get("/api/service/status").json()["state"] != "ready"
    assert commands[0] == [
        sys.executable,
        "-m",
        "light_deploy.api",
        "--model",
        str(config.model_path),
        "--port",
        "18031",
        "--device",
        "cuda:0",
        "--max-model-len",
        "777",
        "--kv-cache-memory-bytes",
        "123456",
        "--deterministic",
    ]


def test_sonic_video_range_and_symlink_rejection(tmp_path):
    import tempfile

    app = create_app(ControlConfig(webui_dist=tmp_path))
    with TestClient(app) as client:
        directory = tempfile.TemporaryDirectory(dir=tmp_path)
        video = Path(directory.name) / "result.mp4"
        video.write_bytes(b"0123456789")
        app.state.session.sonic_jobs["example"] = {"state": "done", "directory": directory}
        response = client.get("/api/sonic/jobs/example/video", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["content-range"] == "bytes 2-5/10"
        video.unlink()
        video.symlink_to(tmp_path / "outside.mp4")
        assert client.get("/api/sonic/jobs/example/video").status_code == 404


def test_process_stop_kills_descendant_even_if_leader_exits(tmp_path):
    from server.process import ManagedProcess

    process = ManagedProcess(tmp_path)
    script = """
import os, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
time.sleep(60)
"""
    process.start([sys.executable, "-u", "-c", script])
    log = Path(process.directory.name) / "inference.log"
    deadline = time.monotonic() + 5
    while not log.read_text().strip() and time.monotonic() < deadline:
        time.sleep(0.01)
    descendant = int(log.read_text().strip())
    try:
        process.stop()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = Path(f"/proc/{descendant}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Inference descendant survived Stop")
    finally:
        try:
            os.kill(descendant, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_control_to_inference_api_contract_in_process(tmp_path):
    from types import SimpleNamespace

    from light_deploy.api import InferenceConfig
    from light_deploy.api import create_app as create_gpu_app
    from server.inference_client import InferenceClient

    received = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, **kwargs):
            received.append((prompt, kwargs))
            result = action_result()
            if not kwargs["enable_thinking"]:
                result["reasoning"] = None
            result["action"] = np.asarray(result["action"], dtype=np.float32)
            return SimpleNamespace(**result)

        def close(self):
            pass

    async def check():
        gpu = create_gpu_app(InferenceConfig(model=tmp_path / "model"), inference_factory=Runtime)
        inference = InferenceClient("http://gpu", transport=httpx.ASGITransport(app=gpu))
        control = create_app(
            ControlConfig(inference_url="http://gpu", webui_dist=tmp_path), inference_client=inference
        )
        async with gpu.router.lifespan_context(gpu), control.router.lifespan_context(control):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=control), base_url="http://control"
            ) as client:
                response = await client.post("/api/generations", json={"prompt": "wave", "enable_thinking": False})
                identity = response.json()["generation_id"]
                await control.state.session.generation_task
                generation = (await client.get(f"/api/generations/{identity}")).json()
                assert generation["state"] == "done", generation
                assert generation["num_frames"] == 3
                preview = (await client.get(f"/api/generations/{identity}/action")).json()
                assert len(preview["positions"]) == 3

    asyncio.run(check())
    assert received[0][0] == "wave"
    assert received[0][1]["enable_thinking"] is False
    assert received[0][1]["action_temperature"] == 0.6
    assert received[0][1]["reasoning_top_p"] == 0.95


@pytest.mark.parametrize("field,value", [("ready", "false"), ("ready", 1), ("model", None), ("model", "")])
def test_malformed_health_never_reports_ready(tmp_path, field, value):
    from server.inference_client import InferenceClient

    health = {
        "ready": True,
        "model": "remote",
        "representation": "human_action_138_v1",
        "action_dim": 138,
        "fps": 20,
    }
    health[field] = value
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=health))
    config = ControlConfig(inference_url="http://gpu", webui_dist=tmp_path)
    with TestClient(
        create_app(config, inference_client=InferenceClient(config.api_url, transport=transport))
    ) as client:
        status = client.get("/api/service/status").json()
        assert status["state"] == "error"
        assert "health" in status["error"]["message"]
