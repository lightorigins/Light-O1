"""HumanAction-138 v1.

Inspired by the heading-local root displacement and 6D joint rotation conventions
from https://github.com/Li-xingXiao/272-dim-Motion-Representation.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

import numpy as np

if TYPE_CHECKING:
    import torch

REPRESENTATION_NAME = "human_action_138_v1"
FEATURE_DIM = 138
FPS = 20
AXIS_UP = "Y"
COORDINATE_SYSTEM = "right_handed"
ROT6D_CONVENTION = "first_two_rows_parent_local"

FIELD_SLICES: Mapping[str, slice] = MappingProxyType(
    {
        "root_delta_xz_local": slice(0, 2),
        "pelvis_height_y": slice(2, 3),
        "yaw_delta_rad_per_frame": slice(3, 4),
        "joint_rot6d_local": slice(4, 136),
        "left_hand_open": slice(136, 137),
        "right_hand_open": slice(137, 138),
    }
)

JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
JOINT_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
NUM_JOINTS = len(JOINT_NAMES)

ActionArray: TypeAlias = "np.ndarray | torch.Tensor"
ActionArrayT = TypeVar("ActionArrayT", np.ndarray, "torch.Tensor")


@dataclass(frozen=True)
class HumanActionFields(Generic[ActionArrayT]):
    """Zero-copy views; mutations alias the source when the input is writable.

    Frozen fields prevent rebinding; they do not make the underlying arrays immutable.
    """

    root_delta_xz_local: ActionArrayT
    pelvis_height_y: ActionArrayT
    yaw_delta_rad_per_frame: ActionArrayT
    joint_rot6d_local: ActionArrayT
    left_hand_open: ActionArrayT
    right_hand_open: ActionArrayT


def validate_human_action(action: ActionArrayT) -> ActionArrayT:
    # A Tensor's caller already imported torch; CPU previews must not load the GPU dependency.
    torch = sys.modules.get("torch")
    if not isinstance(action, np.ndarray) and (torch is None or not isinstance(action, torch.Tensor)):
        raise TypeError("action must be a numpy.ndarray or torch.Tensor")
    if action.ndim != 2:
        raise ValueError(f"action must have rank 2 and shape (T, {FEATURE_DIM}), got {tuple(action.shape)}")
    if action.shape[1] != FEATURE_DIM:
        raise ValueError(f"action last dimension must be {FEATURE_DIM}, got {action.shape[1]}")
    if action.shape[0] < 1:
        raise ValueError("action must contain at least one frame")

    if isinstance(action, np.ndarray):
        if action.dtype != np.float32:
            raise TypeError(f"numpy action dtype must be float32, got {action.dtype}")
        finite_ok = bool(np.isfinite(action).all())
        left_hand = action[:, FIELD_SLICES["left_hand_open"]]
        right_hand = action[:, FIELD_SLICES["right_hand_open"]]
        left_hand_ok = bool(((left_hand >= 0) & (left_hand <= 1)).all())
        right_hand_ok = bool(((right_hand >= 0) & (right_hand <= 1)).all())
    else:
        if action.dtype != torch.float32:
            raise TypeError(f"torch action dtype must be float32, got {action.dtype}")
        left_hand = action[:, FIELD_SLICES["left_hand_open"]]
        right_hand = action[:, FIELD_SLICES["right_hand_open"]]
        finite_ok, left_hand_ok, right_hand_ok = torch.stack(
            (
                torch.isfinite(action).all(),
                ((left_hand >= 0) & (left_hand <= 1)).all(),
                ((right_hand >= 0) & (right_hand <= 1)).all(),
            )
        ).tolist()

    if not finite_ok:
        raise ValueError("action must contain only finite values")
    if not left_hand_ok:
        raise ValueError("left_hand_open must be within [0, 1]")
    if not right_hand_ok:
        raise ValueError("right_hand_open must be within [0, 1]")
    return action


def unpack_human_action(action: ActionArrayT) -> HumanActionFields[ActionArrayT]:
    """Return field views; mutations alias the source when the input is writable."""

    validate_human_action(action)
    joint_rot6d = action[:, FIELD_SLICES["joint_rot6d_local"]]
    if isinstance(joint_rot6d, np.ndarray):
        joint_rot6d = np.lib.stride_tricks.as_strided(
            joint_rot6d,
            shape=(action.shape[0], NUM_JOINTS, 6),
            strides=(joint_rot6d.strides[0], 6 * joint_rot6d.strides[1], joint_rot6d.strides[1]),
            writeable=True,
        )
    else:
        joint_rot6d = joint_rot6d.as_strided(
            size=(action.shape[0], NUM_JOINTS, 6),
            stride=(joint_rot6d.stride(0), 6 * joint_rot6d.stride(1), joint_rot6d.stride(1)),
        )
    return HumanActionFields(
        root_delta_xz_local=action[:, FIELD_SLICES["root_delta_xz_local"]],
        pelvis_height_y=action[:, FIELD_SLICES["pelvis_height_y"]],
        yaw_delta_rad_per_frame=action[:, FIELD_SLICES["yaw_delta_rad_per_frame"]],
        joint_rot6d_local=joint_rot6d,
        left_hand_open=action[:, FIELD_SLICES["left_hand_open"]],
        right_hand_open=action[:, FIELD_SLICES["right_hand_open"]],
    )
