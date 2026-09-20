"""Adapt compact humanoid FK to the Sonic policy's pelvis-relative joint convention."""

import numpy as np

from light_deploy.action_tokenizer.fk import forward_kinematics, rotation_6d_to_matrix
from light_deploy.action_tokenizer.representation import unpack_human_action, validate_human_action


def from_human_action(action: np.ndarray) -> dict[str, np.ndarray]:
    validate_human_action(action)
    fk = forward_kinematics(action)
    root = fk.global_rotations[:, 0]
    joints = np.empty((len(action), 24, 3))
    joints[:, :22] = fk.positions
    for joint, wrist, offset in ((22, 20, 0.08), (23, 21, -0.08)):
        joints[:, joint] = fk.positions[:, wrist] + offset * fk.global_rotations[:, wrist, :, 0]
    canonical = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    y_to_z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    return {
        "joints": (((joints - fk.root_translation[:, None]) @ root) @ canonical.T).astype(np.float32),
        "root_rotation": (y_to_z @ root @ canonical.T).astype(np.float32),
        "wrist_rotation": rotation_6d_to_matrix(unpack_human_action(action).joint_rot6d_local[:, 20:22]).astype(
            np.float32
        ),
    }
