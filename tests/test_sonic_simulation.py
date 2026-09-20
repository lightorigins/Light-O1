import importlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest


def test_encoder_observation_uses_actual_robot_orientation():
    policy = importlib.import_module("examples.sonic.policy")
    joints = np.arange(4 * 24 * 3).reshape(4, 24, 3) / 100
    roots = np.tile(np.eye(3), (4, 1, 1))
    wrists = np.zeros((4, 6))
    identity = policy.encoder_observation(joints, roots, wrists, np.eye(3), np.eye(3))
    base = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    turning = policy.encoder_observation(joints, roots, wrists, base, np.eye(3))
    assert identity.shape == (1247,)
    assert identity.dtype == np.float32
    assert identity[0] == 2
    np.testing.assert_array_equal(identity[1:911], 0)
    np.testing.assert_allclose(identity[911:1199], joints.ravel())
    np.testing.assert_allclose(turning[1199:1223], np.tile(base.T[:, :2].ravel(), 4))
    assert not np.array_equal(identity, turning)


def test_decoder_history_is_oldest_first_and_zero_padded():
    policy = importlib.import_module("examples.sonic.policy")
    history = policy.StateHistory()
    history.push(np.eye(3), np.ones(3), policy.DEFAULT_ANGLES.copy(), np.ones(29), np.ones(29) * 2)
    observation = history.observation(np.ones(64) * 3)
    assert observation.shape == (994,)
    np.testing.assert_array_equal(observation[:64], 3)
    np.testing.assert_array_equal(observation[64:91], 0)
    np.testing.assert_array_equal(observation[91:94], 1)
    np.testing.assert_array_equal(observation[94:384], 0)
    np.testing.assert_array_equal(observation[-3:], [0, 0, -1])


@pytest.mark.parametrize("cancel", [True, False])
def test_simulation_subprocess_cancels_or_times_out(cancel):
    sonic = importlib.import_module("server.sonic")
    event = threading.Event()
    if cancel:
        event.set()
    start = time.monotonic()
    with pytest.raises(sonic.SonicSimulationError, match="cancelled" if cancel else "timed out"):
        sonic._run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"], env=os.environ, timeout=0.1, cancel_event=event
        )
    assert time.monotonic() - start < 3


def test_termination_kills_descendants_even_when_group_leader_exits():
    sonic = importlib.import_module("server.sonic")
    script = """
import os, signal, time
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    child = int(process.stdout.readline())
    try:
        sonic.terminate_process_group(process, grace_seconds=0.5)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Sonic descendant survived process-group termination")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


def test_invalid_simulation_settings_fail_before_launch(tmp_path):
    sonic = importlib.import_module("server.sonic")
    with pytest.raises(ValueError, match="positive"):
        sonic.run_sonic_simulation(
            tmp_path / "plan.npz",
            tmp_path / "out.json",
            sim_python=sys.executable,
            sonic_checkpoint=tmp_path,
            mp4_path=tmp_path / "out.mp4",
            replan_frames=0,
        )


def test_runtime_readiness_reports_missing_checkpoint(tmp_path):
    sonic = importlib.import_module("server.sonic")
    ready, reason = sonic.check_runtime(sys.executable, tmp_path)
    assert not ready
    assert "model_encoder.onnx" in reason


def test_runtime_readiness_reports_missing_optional_dependencies(tmp_path):
    sonic = importlib.import_module("server.sonic")
    for name in ("model_encoder.onnx", "model_decoder.onnx"):
        (tmp_path / name).touch()
    ready, reason = sonic.check_runtime(sys.executable, tmp_path)
    assert not ready
    assert reason


def test_standalone_module_help_has_no_private_runtime_dependency():
    process = subprocess.run([sys.executable, "-m", "examples.sonic", "--help"], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--checkpoint" in process.stdout


@pytest.mark.skipif(not os.environ.get("SONIC_TEST_PYTHON"), reason="Requires optional MuJoCo environment")
def test_robot_orientation_reads_current_qpos_not_cached_mujoco_kinematics():
    script = """
import numpy as np
from types import SimpleNamespace
from examples.sonic.simulation import SonicSimulator
simulator = SonicSimulator.__new__(SonicSimulator)
simulator.pelvis = 0
simulator.data = SimpleNamespace(qpos=np.array([0,0,1,np.sqrt(.5),0,0,np.sqrt(.5)]), xmat=np.eye(3).reshape(1,9))
np.testing.assert_allclose(simulator.rotation(), [[0,-1,0],[1,0,0],[0,0,1]], atol=1e-7)
"""
    result = subprocess.run([os.environ["SONIC_TEST_PYTHON"], "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not os.environ.get("SONIC_TEST_PYTHON"),
    reason="Set SONIC_TEST_PYTHON and SONIC_TEST_CHECKPOINT for MuJoCo/ONNX integration",
)
@pytest.mark.parametrize("frames,yaw,lean", [(1, 0, False), (41, 0, False), (101, 0.015, False), (120, 0, True)])
def test_real_standalone_dynamics_and_video(tmp_path, frames, yaw, lean):
    sonic = importlib.import_module("server.sonic")
    ready, reason = sonic.check_runtime(os.environ["SONIC_TEST_PYTHON"], os.environ["SONIC_TEST_CHECKPOINT"])
    assert ready, reason
    value = np.zeros((frames, 138), np.float32)
    value[:, 2] = 0.95
    value[:, 3] = yaw
    value[:, 4:136] = np.tile([1, 0, 0, 0, 1, 0], 22)
    if lean:
        value[:, 4:10] = [1, 0, 0, 0, 0, -1]
    plan = sonic.write_sonic_plan(value, tmp_path / "plan.npz", provenance={})
    result = sonic.run_sonic_simulation(
        plan,
        tmp_path / "result.json",
        sim_python=os.environ["SONIC_TEST_PYTHON"],
        sonic_checkpoint=os.environ["SONIC_TEST_CHECKPOINT"],
        mp4_path=tmp_path / "result.mp4",
        replan_frames=3,
        timeout=120,
    )
    assert result["planned_s"] == frames / 20
    assert result["survived_s"] > 0
    assert np.isfinite(result["pelvis_min"])
    assert result["physics_steps"] >= round(result["survived_s"] * 200)
    if lean:
        assert result["fell_at"] == result["survived_s"] < result["planned_s"]
        assert result["pelvis_min"] < 0.35
    else:
        assert result["fell_at"] is None
        assert result["survived_s"] == round(frames * 2.5) / 50
    assert Path(result["mp4_path"]).stat().st_size > 1000
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
            result["mp4_path"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "640,480,25/1"
