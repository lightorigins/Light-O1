"""Opt-in check against a running GPU-backed Control Server with Sonic configured."""

import base64
import io
import os
import time

import httpx
import pytest
from PIL import Image


@pytest.mark.gpu_e2e
@pytest.mark.skipif(
    not os.environ.get("LIGHT_DEPLOY_E2E_URL"), reason="Set LIGHT_DEPLOY_E2E_URL to a running demo service"
)
def test_real_generation_preview_and_sonic():
    def finished(client, path):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            response = client.get(path)
            response.raise_for_status()
            result = response.json()
            if result["state"] == "done":
                return result
            assert result["state"] not in {"error", "cancelled"}, result
            time.sleep(0.2)
        pytest.fail("Real service job timed out")

    with httpx.Client(base_url=os.environ["LIGHT_DEPLOY_E2E_URL"], timeout=30, trust_env=False) as client:
        assert client.get("/api/service/status").json()["state"] == "ready"
        assert client.get("/api/defaults").json()["defaults"]["capabilities"]["sonic"]
        buffer = io.BytesIO()
        Image.new("RGB", (128, 128), (100, 140, 180)).save(buffer, format="PNG")
        image = base64.b64encode(buffer.getvalue()).decode()
        generations = []
        for thinking, images in ((True, []), (False, [image])):
            response = client.post(
                "/api/generations",
                json={
                    "prompt": "a person raises the right arm and waves",
                    "enable_thinking": thinking,
                    "images": images,
                },
            )
            assert response.status_code == 202, response.text
            identity = response.json()["generation_id"]
            generation = finished(client, f"/api/generations/{identity}")
            assert 40 <= generation["num_frames"] <= 400
            assert bool(generation["reasoning"]) is thinking
            assert generation["representation"] == "human_action_138_v1"
            preview = client.get(f"/api/generations/{identity}/action")
            assert preview.status_code == 200, preview.text
            assert len(preview.json()["positions"]) == generation["num_frames"]
            assert all(len(frame) == 22 for frame in preview.json()["positions"])
            generations.append(identity)
            print(
                {
                    "generation": identity,
                    "thinking": thinking,
                    "images": len(images),
                    "frames": generation["num_frames"],
                }
            )
        response = client.post(f"/api/generations/{generations[0]}/sonic/sim", json={})
        assert response.status_code == 202, response.text
        identity = response.json()["job_id"]
        result = finished(client, f"/api/sonic/jobs/{identity}")
        video = client.get(f"/api/sonic/jobs/{identity}/video", headers={"Range": "bytes=0-1023"})
        assert video.status_code == 206
        assert len(video.content) == 1024
        assert b"ftyp" in video.content[:32]
        print({"sonic_job": identity, "result": result["result"], "range": video.headers["content-range"]})
