"""GEAR-SONIC low-latency observation and G1 actuator contracts, in NumPy."""

from collections import deque

import numpy as np

JOINT_NAMES = tuple(
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)
MUJOCO_TO_ISAACLAB = np.argsort(ISAACLAB_TO_MUJOCO)
DEFAULT_ANGLES = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0] * 2 + [0, 0, 0] + [0.2, 0.2, 0, 0.6, 0, 0, 0] + [0.2, -0.2, 0, 0.6, 0, 0, 0]
)
_MOTORS = np.array([2, 2, 1, 2, 0, 0] * 2 + [1, 0, 0] + [0, 0, 0, 0, 0, 3, 3] * 2)
_ARMATURE = np.array([0.003609725, 0.010177520, 0.025101925, 0.00425])[_MOTORS]
EFFORT_LIMIT = np.array([25, 88, 139, 5])[_MOTORS]
_MULTIPLIER = np.ones(29)
_MULTIPLIER[[4, 5, 10, 11, 13, 14]] = 2
_OMEGA = 20 * np.pi
KPS = _ARMATURE * _OMEGA**2 * _MULTIPLIER
KDS = 4 * _ARMATURE * _OMEGA * _MULTIPLIER
ACTION_SCALE = 0.25 * EFFORT_LIMIT / (_ARMATURE * _OMEGA**2)


def encoder_observation(joints, roots, wrists, base_rotation, heading_alignment):
    if joints.shape != (4, 24, 3) or roots.shape != (4, 3, 3) or wrists.shape != (4, 6):
        raise ValueError("Low-latency Sonic requires exactly four future reference frames")
    observation = np.zeros(1247, np.float32)
    observation[0] = 2
    observation[911:1199] = joints.ravel()
    observation[1199:1223] = (base_rotation.T @ heading_alignment @ roots)[:, :, :2].ravel()
    observation[1223:] = wrists.ravel()
    if not np.isfinite(observation).all():
        raise ValueError("Non-finite Sonic encoder observation")
    return observation


class StateHistory:
    def __init__(self):
        self.entries = deque(maxlen=10)

    def push(self, rotation, angular_velocity, q, dq, last_action):
        self.entries.append(
            (
                angular_velocity.copy(),
                (q - DEFAULT_ANGLES)[MUJOCO_TO_ISAACLAB],
                dq[MUJOCO_TO_ISAACLAB],
                last_action.copy(),
                -rotation[2].copy(),
            )
        )

    def observation(self, token):
        if np.asarray(token).shape != (64,):
            raise ValueError("Sonic token must have shape (64,)")
        observation = np.zeros(994, np.float32)
        observation[:64] = token
        offset = 64
        for component, width in enumerate((3, 29, 29, 29, 3)):
            rows = observation[offset : offset + 10 * width].reshape(10, width)
            if self.entries:
                rows[-len(self.entries) :] = np.stack([entry[component] for entry in self.entries])
            offset += 10 * width
        return observation
