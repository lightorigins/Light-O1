from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from examples.sonic.adapter import from_human_action
from examples.sonic.reference import FPS
from server.process import terminate_process_group


class SonicSimulationError(RuntimeError):
    pass


def check_runtime(sim_python: str | Path | None, sonic_checkpoint: str | Path | None) -> tuple[bool, str]:
    if sim_python is None or sonic_checkpoint is None:
        return False, "Configure Sonic simulation Python and low-latency checkpoint directory"
    python = Path(sim_python).expanduser().absolute()
    checkpoint = Path(sonic_checkpoint).expanduser().resolve()
    for path in (python, checkpoint / "model_encoder.onnx", checkpoint / "model_decoder.onnx"):
        if not path.is_file():
            return False, f"Sonic required file not found: {path}"
    if shutil.which("ffmpeg") is None:
        return False, "Install ffmpeg to render Sonic videos"
    script = """
import sys
from pathlib import Path
import mujoco
import numpy
import onnxruntime as ort
from PIL import Image
options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
for name, shape in (("encoder", (1, 1247)), ("decoder", (1, 994))):
    session = ort.InferenceSession(str(Path(sys.argv[1]) / f"model_{name}.onnx"),
        sess_options=options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or tuple(session.get_inputs()[0].shape) != shape:
        raise ValueError(f"Expected low-latency {name} input shape {shape}")
"""
    environment = dict(os.environ)
    environment.setdefault("MUJOCO_GL", "egl")
    try:
        result = subprocess.run(
            [str(python), "-c", script, str(checkpoint)], env=environment, capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"Sonic runtime check failed: {error}"
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr else "probe failed"
        return False, f"Sonic runtime unavailable: {detail}"
    return True, "Sonic low-latency policy and simulation dependencies available"


def write_sonic_plan(action: np.ndarray, output_path: str | Path, *, provenance: Mapping[str, Any]) -> Path:
    reference = from_human_action(action)
    path = Path(output_path).absolute()
    if path.suffix != ".npz":
        raise ValueError("Sonic plan path must end in .npz")
    metadata = json.dumps(
        {"schema_version": 1, "representation": "human_action_138_v1", "fps": FPS, "provenance": dict(provenance)},
        allow_nan=False,
    )
    with path.open("xb") as file:
        np.savez_compressed(file, metadata=np.array(metadata), **reference)
    return path


def _run_process(command, *, env, timeout, cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise SonicSimulationError("Sonic simulation cancelled")
    process = subprocess.Popen(
        command, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise SonicSimulationError("Sonic simulation cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SonicSimulationError(f"Sonic simulation timed out after {timeout:g}s")
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    finally:
        terminate_process_group(process, grace_seconds=0.5)
        process.communicate()


def run_sonic_simulation(
    plan_path: str | Path,
    output_path: str | Path,
    *,
    sim_python: str | Path,
    sonic_checkpoint: str | Path,
    mp4_path: str | Path,
    replan_frames: int = 8,
    lookahead_frames: int = 12,
    timeout: float = 3600,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    if replan_frames < 1 or lookahead_frames < 1 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("Sonic replan frames, lookahead frames, and timeout must be positive")
    plan, output, video = (Path(path).absolute() for path in (plan_path, output_path, mp4_path))
    checkpoint = Path(sonic_checkpoint).expanduser().resolve()
    python = Path(sim_python).expanduser().absolute()
    for path in (plan, python, checkpoint / "model_encoder.onnx", checkpoint / "model_decoder.onnx"):
        if not path.is_file():
            raise FileNotFoundError(f"Sonic required file not found: {path}")
    for path in (output, video):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Sonic output already exists: {path}")
        if path.parent != plan.parent:
            raise ValueError("Sonic outputs must be stored beside their generation plan")
    command = [
        str(python),
        "-m",
        "examples.sonic",
        "--plan",
        str(plan),
        "--out",
        str(output),
        "--mp4",
        str(video),
        "--checkpoint",
        str(checkpoint),
        "--replan-frames",
        str(replan_frames),
        "--lookahead-frames",
        str(lookahead_frames),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    environment.setdefault("MUJOCO_GL", "egl")
    completed = _run_process(command, env=environment, timeout=timeout, cancel_event=cancel_event)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout)[-2000:]
        raise SonicSimulationError(f"Sonic simulation exited with code {completed.returncode}: {details}")
    if not output.is_file() or not video.is_file():
        raise SonicSimulationError("Sonic simulation did not produce its result and MP4")
    try:
        result = json.loads(output.read_text())
        if not isinstance(result, dict):
            raise ValueError("expected an object")
    except (OSError, ValueError) as error:
        raise SonicSimulationError("Sonic simulation produced an invalid result") from error
    return {**result, "result_path": str(output), "mp4_path": str(video)}
