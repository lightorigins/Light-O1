"""Closed-loop ONNX policy tracking with 200 Hz MuJoCo physics and 50 Hz control."""

import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image

from examples.sonic.policy import (
    ACTION_SCALE,
    DEFAULT_ANGLES,
    EFFORT_LIMIT,
    ISAACLAB_TO_MUJOCO,
    JOINT_NAMES,
    KDS,
    KPS,
    StateHistory,
    encoder_observation,
)
from examples.sonic.reference import FPS, ReferenceStream


def load_model():
    assets = Path(__file__).parent / "assets"
    root = ET.parse(assets / "g1/g1_29dof.xml").getroot()
    root.find("compiler").set("meshdir", str(assets / "g1/meshes"))
    root.extend(ET.parse(assets / "scene.xml").getroot())
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for name in JOINT_NAMES:
        joint = model.joint(name)
        model.dof_damping[joint.dofadr] = 0.05
        model.dof_armature[joint.dofadr] = 0.01
        model.dof_frictionloss[joint.dofadr] = 0.1 if "wrist" in name else 0.2
    return model


def heading(rotation):
    angle = np.arctan2(rotation[1, 0], rotation[0, 0])
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


class SonicSimulator:
    def __init__(self, checkpoint: Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(checkpoint / "model_encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(checkpoint / "model_decoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for session, shape in ((self.encoder, (1, 1247)), (self.decoder, (1, 994))):
            if len(session.get_inputs()) != 1 or tuple(session.get_inputs()[0].shape) != shape:
                raise ValueError(f"Expected matching low-latency Sonic policy pair with input shape {shape}")
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(name).qposadr.item() for name in JOINT_NAMES])
        self.vadr = np.array([self.model.joint(name).dofadr.item() for name in JOINT_NAMES])
        actuator_by_joint = {int(joint): index for index, joint in enumerate(self.model.actuator_trnid[:, 0])}
        self.actuators = np.array([actuator_by_joint[self.model.joint(name).id] for name in JOINT_NAMES])
        self.pelvis = self.model.body("pelvis").id
        self.history = StateHistory()
        self.last_action = np.zeros(29)
        self.heading_alignment = np.eye(3)
        self.physics_steps = 0
        self.renderer = None
        self.torque_low = -EFFORT_LIMIT.astype(float)
        self.torque_high = EFFORT_LIMIT.astype(float)
        limited = self.model.actuator_ctrllimited[self.actuators].astype(bool)
        self.torque_low[limited] = self.model.actuator_ctrlrange[self.actuators[limited], 0]
        self.torque_high[limited] = self.model.actuator_ctrlrange[self.actuators[limited], 1]

    def rotation(self):
        rotation = np.empty(9)
        # mj_step's cached xmat precedes the final integration; qpos is the current state.
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return rotation.reshape(3, 3)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, 1]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qadr] = DEFAULT_ANGLES
        mujoco.mj_forward(self.model, self.data)
        active = (self.model.geom_contype != 0) & (self.model.geom_bodyid > 0)
        lowest = np.min(self.data.geom_xpos[active, 2] - self.model.geom_size[active, 0])
        self.data.qpos[2] = 1 - lowest + 0.002
        mujoco.mj_forward(self.model, self.data)
        self.crane_position = self.data.qpos[:3].copy()
        self.crane_rotation = self.rotation().copy()
        self.history = StateHistory()
        self.last_action[:] = 0
        self.physics_steps = 0

    def _crane(self, scale):
        force = 10000 * (self.crane_position - self.data.qpos[:3]) - 1000 * self.data.qvel[:3]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (self.rotation() @ self.crane_rotation.T).ravel())
        if quaternion[0] < 0:
            quaternion *= -1
        norm = np.linalg.norm(quaternion[1:])
        rotvec = quaternion[1:] * (2 * np.arctan2(norm, quaternion[0]) / norm) if norm > 1e-9 else np.zeros(3)
        angular_velocity = self.rotation() @ self.data.qvel[3:6]
        self.data.xfrc_applied[self.pelvis] = np.concatenate((force, -1000 * rotvec - 10 * angular_velocity)) * scale

    def _physics(self, target, crane_scale):
        for _ in range(4):
            q, dq = self.data.qpos[self.qadr], self.data.qvel[self.vadr]
            self.data.ctrl[self.actuators] = np.clip(KPS * (target - q) - KDS * dq, self.torque_low, self.torque_high)
            self._crane(crane_scale)
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Sonic simulation produced non-finite robot state")

    def warmup(self):
        for _ in range(100):
            self._physics(DEFAULT_ANGLES, 1.0)

    def step(self, reference, frame, crane_scale=0.0):
        rotation = self.rotation().copy()
        self.history.push(
            rotation, self.data.qvel[3:6], self.data.qpos[self.qadr], self.data.qvel[self.vadr], self.last_action
        )
        indices = np.minimum(np.arange(frame, frame + 4), len(reference["joints"]) - 1)
        observation = encoder_observation(
            reference["joints"][indices],
            reference["root_rotation"][indices],
            reference["wrists"][indices],
            rotation,
            self.heading_alignment,
        )
        token = self.encoder.run(None, {self.encoder.get_inputs()[0].name: observation[None]})[0][0]
        decoder_input = self.history.observation(token)
        action = self.decoder.run(None, {self.decoder.get_inputs()[0].name: decoder_input[None]})[0][0]
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Sonic decoder must produce 29 finite joint actions")
        self.last_action = action.astype(float)
        self._physics(DEFAULT_ANGLES + self.last_action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE, crane_scale)

    def snapshot(self, path):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [self.data.qpos[0], self.data.qpos[1], 0.8]
        camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -15
        self.renderer.update_scene(self.data, camera)
        Image.fromarray(self.renderer.render()).save(path, quality=85)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def run(args):
    result_path, video_path = Path(args.out), Path(args.mp4)
    for path in (result_path, video_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    with np.load(args.plan, allow_pickle=False) as plan:
        metadata = json.loads(plan["metadata"].item())
        if (
            metadata["schema_version"] != 1
            or metadata["representation"] != "human_action_138_v1"
            or metadata["fps"] != FPS
        ):
            raise ValueError("Sonic requires a human_action_138_v1 plan at 20 FPS")
        reference = {key: plan[key] for key in ("joints", "root_rotation", "wrist_rotation")}
    stream = ReferenceStream(reference, replan_frames=args.replan_frames, lookahead_frames=args.lookahead_frames)
    simulator = SonicSimulator(Path(args.checkpoint))
    tick, fell_at, heights = 0, None, []
    try:
        simulator.reset()
        simulator.warmup()
        with tempfile.TemporaryDirectory(prefix="sonic-frames-", dir=result_path.parent) as scratch:
            for start, end, chunk in stream:
                if start == 0:
                    simulator.heading_alignment = (
                        heading(simulator.rotation()) @ heading(reference["root_rotation"][0]).T
                    )
                    for settle in range(75):
                        simulator.step(chunk, 0, crane_scale=1 - settle / 75)
                for frame in range(end - start):
                    simulator.step(chunk, frame)
                    if tick % 2 == 0:
                        simulator.snapshot(Path(scratch) / f"frame_{tick:05d}.jpg")
                    tick += 1
                    heights.append(float(simulator.data.qpos[2]))
                    if heights[-1] < 0.35:
                        fell_at = tick / 50
                        break
                if fell_at is not None:
                    break
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-framerate",
                    "25",
                    "-pattern_type",
                    "glob",
                    "-i",
                    str(Path(scratch) / "frame_*.jpg"),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video_path),
                ],
                check=True,
                timeout=120,
            )
        result = {
            "variant": "low_latency",
            "skeleton_profile": "humanoid22_v1",
            "fell_at": fell_at,
            "survived_s": tick / 50,
            "planned_s": stream.frames / FPS,
            "pelvis_min": min(heights),
            "physics_steps": simulator.physics_steps,
        }
        with result_path.open("x") as output:
            json.dump(result, output, allow_nan=False)
        return result
    finally:
        simulator.close()
