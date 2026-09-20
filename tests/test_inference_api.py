import base64
import importlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


class Runtime:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.error = None
        self.entered = threading.Event()
        self.release = None

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        if self.error is not None:
            raise self.error
        action = np.zeros((2, 138), dtype=np.float32)
        action[:, 2] = 1.0
        action[:, 4:136].reshape(2, 22, 6)[..., [0, 4]] = 1.0
        return SimpleNamespace(
            action=action,
            representation="human_action_138_v1",
            reasoning="A short wave" if kwargs["enable_thinking"] else None,
            finish_reason="action_end",
            raw_token_count=4,
            action_token_count=2,
            action_token_ids=[65538, 65539],
            action_codebook_ids=[2, 3],
        )

    def close(self):
        self.closed = True


@pytest.fixture
def api():
    assert importlib.util.find_spec("light_deploy.api") is not None, "GPU HTTP API is missing"
    return importlib.import_module("light_deploy.api")


def test_gpu_http_api_module_exists():
    assert importlib.util.find_spec("light_deploy.api") is not None, "GPU HTTP API is missing"


def test_runtime_lifecycle_uses_one_device_owning_thread(api):
    threads = []

    class ThreadBoundRuntime(Runtime):
        def generate(self, prompt, **kwargs):
            threads.append(threading.get_ident())
            return super().generate(prompt, **kwargs)

        def close(self):
            threads.append(threading.get_ident())
            super().close()

    def factory(*args, **kwargs):
        threads.append(threading.get_ident())
        return ThreadBoundRuntime()

    app = api.create_app(api.InferenceConfig(model=Path("/models/test-model")), inference_factory=factory)
    with TestClient(app) as client:
        assert client.post("/api/generate", json={"prompt": "wave"}).status_code == 200
    assert len(threads) == 3
    assert len(set(threads)) == 1


def app_with_runtime(api, runtime):
    def factory(model_path, **kwargs):
        assert model_path == Path("/models/test-model")
        assert kwargs["max_model_len"] == 1024
        return runtime

    return api.create_app(api.InferenceConfig(model=Path("/models/test-model")), inference_factory=factory)


def test_resident_api_readiness_generation_defaults_and_shutdown(api):
    runtime = Runtime()
    app = app_with_runtime(api, runtime)
    assert TestClient(app).get("/health").status_code == 503
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["ready"]
        assert health["model"] == "test-model"
        assert health["representation"] == "human_action_138_v1"
        assert health["action_dim"] == 138
        response = client.post("/api/generate", json={"prompt": " wave "})
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == 1
        assert payload["representation"] == "human_action_138_v1"
        assert np.asarray(payload["action"]).shape == (2, 138)
        assert payload["num_frames"] == 2
        assert payload["fps"] == 20
        assert payload["action_codebook_ids"] == [2, 3]
        prompt, kwargs = runtime.calls[0]
        assert prompt == "wave"
        assert kwargs["enable_thinking"]
        assert kwargs["reasoning_temperature"] == kwargs["action_temperature"] == 0.6
        assert kwargs["reasoning_top_p"] == kwargs["action_top_p"] == 0.95
        assert kwargs["reasoning_token_budget"] == 384
        assert kwargs["min_action_tokens"] == 40
        assert kwargs["max_action_tokens"] == 400
        assert kwargs["max_new_tokens"] == 785
    assert runtime.closed


def test_direct_mode_image_and_custom_sampling(api):
    runtime = Runtime()
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color=(30, 40, 50)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    with TestClient(app_with_runtime(api, runtime)) as client:
        response = client.post(
            "/api/generate",
            json={
                "prompt": "wave",
                "images": [encoded],
                "enable_thinking": False,
                "action_min_seconds": 3,
                "action_max_seconds": 4,
                "reasoning_temperature": 0.4,
                "reasoning_top_p": 0.8,
                "action_temperature": 0.7,
                "action_top_p": 0.9,
            },
        )
    assert response.status_code == 200
    assert response.json()["reasoning"] is None
    kwargs = runtime.calls[0][1]
    assert kwargs["max_new_tokens"] == 80
    assert kwargs["min_action_tokens"] == 60
    assert kwargs["images"][0].size == (16, 12)
    assert kwargs["images"][0].getpixel((0, 0)) == (30, 40, 50)
    assert kwargs["reasoning_temperature"] == 0.4
    assert kwargs["action_top_p"] == 0.9


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": "   "},
        {"action_min_seconds": 8, "action_max_seconds": 2},
        {"action_max_seconds": 21},
        {"top_k": 10},
        {"images": ["not base64"]},
        {"images": [base64.b64encode(b"not an image").decode()]},
        {"action_top_p": 0},
    ],
)
def test_invalid_requests_do_not_enter_model(api, changes):
    runtime = Runtime()
    with TestClient(app_with_runtime(api, runtime)) as client:
        response = client.post("/api/generate", json={"prompt": "wave", **changes})
    assert response.status_code in (400, 422)
    assert not runtime.calls


def test_generation_failure_is_diagnostic_without_exposing_private_paths(api):
    runtime = Runtime()
    runtime.error = RuntimeError("private details /private/checkpoint")
    with TestClient(app_with_runtime(api, runtime)) as client:
        response = client.post("/api/generate", json={"prompt": "wave"})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "generation_failed"
        assert "/private" not in response.text
        runtime.error = None
        assert client.post("/api/generate", json={"prompt": "wave"}).status_code == 200


def test_concurrent_generation_is_rejected_without_entering_model_twice(api):
    runtime = Runtime()
    runtime.release = threading.Event()
    with TestClient(app_with_runtime(api, runtime)) as client, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/api/generate", json={"prompt": "first"})
        try:
            assert runtime.entered.wait(timeout=5)
            response = client.post("/api/generate", json={"prompt": "second"})
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "busy"
            assert len(runtime.calls) == 1
        finally:
            runtime.release.set()
        assert pending.result(timeout=5).status_code == 200


def test_startup_failure_is_not_reported_as_ready(api):
    def factory(*args, **kwargs):
        raise RuntimeError("cannot load model")

    app = api.create_app(api.InferenceConfig(model=Path("/models/missing")), inference_factory=factory)
    with pytest.raises(RuntimeError, match="cannot load model"), TestClient(app):
        pass
    assert TestClient(app).get("/health").status_code == 503


def test_generation_budget_cannot_exceed_context(api):
    runtime = Runtime()
    with TestClient(app_with_runtime(api, runtime)) as client:
        response = client.post("/api/generate", json={"prompt": "wave", "reasoning_token_budget": 4096})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert not runtime.calls


def test_tokenized_context_overflow_has_actionable_safe_error(api):
    assert hasattr(api, "ContextLengthError"), "context overflow needs a typed exception"
    runtime = Runtime()
    runtime.error = api.ContextLengthError(1200, 1024)
    with TestClient(app_with_runtime(api, runtime)) as client:
        response = client.post("/api/generate", json={"prompt": "wave"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert "1024" in error["message"]
    assert "reduce" in error["message"].lower()
