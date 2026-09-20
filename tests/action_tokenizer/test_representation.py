from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

import light_deploy.action_tokenizer as human_action
from light_deploy.action_tokenizer import (
    AXIS_UP,
    COORDINATE_SYSTEM,
    FEATURE_DIM,
    FIELD_SLICES,
    FPS,
    JOINT_NAMES,
    JOINT_PARENTS,
    REPRESENTATION_NAME,
    ROT6D_CONVENTION,
    HumanActionFields,
    unpack_human_action,
    validate_human_action,
)

EXPECTED_JOINT_NAMES = (
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
EXPECTED_JOINT_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
EXPECTED_PUBLIC_EXPORTS = {
    "AXIS_UP",
    "COORDINATE_SYSTEM",
    "FEATURE_DIM",
    "FIELD_SLICES",
    "FPS",
    "JOINT_NAMES",
    "JOINT_PARENTS",
    "NUM_JOINTS",
    "REPRESENTATION_NAME",
    "ROT6D_CONVENTION",
    "HumanActionFields",
    "ActionArray",
    "unpack_human_action",
    "validate_human_action",
}


def _valid_numpy_action(frames: int = 3) -> np.ndarray:
    action = np.zeros((frames, FEATURE_DIM), dtype=np.float32)
    action[:, FIELD_SLICES["left_hand_open"]] = 0.0
    action[:, FIELD_SLICES["right_hand_open"]] = 1.0
    return action


def _valid_torch_action(frames: int = 3) -> torch.Tensor:
    return torch.from_numpy(_valid_numpy_action(frames))


def _action_for_backend(backend: str, frames: int = 3) -> np.ndarray | torch.Tensor:
    return _valid_torch_action(frames) if backend == "torch" else _valid_numpy_action(frames)


def test_contract_metadata_and_exact_field_slices() -> None:
    assert REPRESENTATION_NAME == "human_action_138_v1"
    assert FEATURE_DIM == 138
    assert FPS == 20
    assert AXIS_UP == "Y"
    assert COORDINATE_SYSTEM == "right_handed"
    assert ROT6D_CONVENTION == "first_two_rows_parent_local"
    assert dict(FIELD_SLICES) == {
        "root_delta_xz_local": slice(0, 2),
        "pelvis_height_y": slice(2, 3),
        "yaw_delta_rad_per_frame": slice(3, 4),
        "joint_rot6d_local": slice(4, 136),
        "left_hand_open": slice(136, 137),
        "right_hand_open": slice(137, 138),
    }
    with pytest.raises(TypeError):
        FIELD_SLICES["root_delta_xz_local"] = slice(0, 3)


def test_joint_order_and_parent_tree_are_exact_and_immutable() -> None:
    assert JOINT_NAMES == EXPECTED_JOINT_NAMES
    assert JOINT_PARENTS == EXPECTED_JOINT_PARENTS
    assert len(JOINT_NAMES) == len(JOINT_PARENTS) == 22
    assert isinstance(JOINT_NAMES, tuple)
    assert isinstance(JOINT_PARENTS, tuple)


def test_numpy_action_exposes_named_views_without_copying() -> None:
    action = np.arange(2 * FEATURE_DIM, dtype=np.float32).reshape(2, FEATURE_DIM)
    action[:, FIELD_SLICES["left_hand_open"]] = 0.25
    action[:, FIELD_SLICES["right_hand_open"]] = 0.75

    fields = unpack_human_action(action)

    assert isinstance(fields, HumanActionFields)
    assert fields.root_delta_xz_local.shape == (2, 2)
    assert fields.pelvis_height_y.shape == (2, 1)
    assert fields.yaw_delta_rad_per_frame.shape == (2, 1)
    assert fields.joint_rot6d_local.shape == (2, 22, 6)
    assert fields.left_hand_open.shape == (2, 1)
    assert fields.right_hand_open.shape == (2, 1)
    assert np.shares_memory(fields.root_delta_xz_local, action)
    assert np.shares_memory(fields.joint_rot6d_local, action)
    with pytest.raises(FrozenInstanceError):
        fields.left_hand_open = fields.right_hand_open


def test_numpy_read_only_action_returns_read_only_views() -> None:
    action = _valid_numpy_action()
    action.flags.writeable = False

    assert validate_human_action(action) is action
    fields = unpack_human_action(action)

    assert all(not field.flags.writeable for field in fields.__dict__.values())
    with pytest.raises(ValueError, match="read-only"):
        fields.pelvis_height_y[0, 0] = 1.25


def test_torch_action_exposes_named_views() -> None:
    action = _valid_torch_action(2)

    fields = unpack_human_action(action)

    assert fields.joint_rot6d_local.shape == (2, 22, 6)
    assert fields.root_delta_xz_local.untyped_storage().data_ptr() == action.untyped_storage().data_ptr()
    assert fields.joint_rot6d_local.untyped_storage().data_ptr() == action.untyped_storage().data_ptr()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_named_field_mutations_alias_the_source(backend: str) -> None:
    action = _action_for_backend(backend, frames=2)
    fields = unpack_human_action(action)

    fields.pelvis_height_y[0, 0] = 1.25
    fields.joint_rot6d_local[1, 3, 4] = -0.5

    assert action[0, 2].item() == pytest.approx(1.25)
    assert action[1, 4 + 3 * 6 + 4].item() == pytest.approx(-0.5)


@pytest.mark.parametrize("width", [137, 139])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_validation_rejects_wrong_feature_width(width: int, backend: str) -> None:
    action = np.zeros((2, width), dtype=np.float32)
    value = torch.from_numpy(action) if backend == "torch" else action

    with pytest.raises(ValueError, match=f"last dimension must be {FEATURE_DIM}, got {width}"):
        validate_human_action(value)


@pytest.mark.parametrize("shape", [(FEATURE_DIM,), (2, 3, FEATURE_DIM)])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_validation_rejects_non_matrix_shapes(shape: tuple[int, ...], backend: str) -> None:
    action = np.zeros(shape, dtype=np.float32)
    value = torch.from_numpy(action) if backend == "torch" else action

    with pytest.raises(ValueError, match="action must have rank 2"):
        validate_human_action(value)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_validation_rejects_zero_frames(backend: str) -> None:
    action = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    value = torch.from_numpy(action) if backend == "torch" else action

    with pytest.raises(ValueError, match="action must contain at least one frame"):
        validate_human_action(value)


@pytest.mark.parametrize(
    ("backend", "dtype"),
    [
        ("numpy", np.bool_),
        ("numpy", np.int64),
        ("numpy", np.float64),
        ("numpy", np.float16),
        ("torch", torch.bool),
        ("torch", torch.int64),
        ("torch", torch.float64),
        ("torch", torch.float16),
    ],
)
def test_validation_rejects_non_float32_dtypes(backend: str, dtype: object) -> None:
    if backend == "numpy":
        action = np.zeros((2, FEATURE_DIM), dtype=dtype)
    else:
        action = torch.zeros((2, FEATURE_DIM), dtype=dtype)

    with pytest.raises(TypeError, match=f"{backend} action dtype must be float32"):
        validate_human_action(action)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_validation_accepts_valid_arrays_with_hand_endpoints_and_preserves_identity(backend: str) -> None:
    action = _action_for_backend(backend)

    assert action[0, FIELD_SLICES["left_hand_open"]].item() == 0.0
    assert action[0, FIELD_SLICES["right_hand_open"]].item() == 1.0
    assert validate_human_action(action) is action


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_validation_rejects_non_finite_values(backend: str, bad_value: float) -> None:
    action = _action_for_backend(backend)
    action[1, 12] = bad_value

    with pytest.raises(ValueError, match="must contain only finite values"):
        validate_human_action(action)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left_hand_open", -0.001),
        ("left_hand_open", 1.001),
        ("right_hand_open", -0.001),
        ("right_hand_open", 1.001),
    ],
)
def test_validation_rejects_hand_values_outside_open01(backend: str, field: str, value: float) -> None:
    action = _action_for_backend(backend)
    action[:, FIELD_SLICES[field]] = value

    with pytest.raises(ValueError, match=f"{field} must be within \\[0, 1\\]"):
        validate_human_action(action)


def test_validation_rejects_unsupported_array_types() -> None:
    with pytest.raises(TypeError, match="must be a numpy.ndarray or torch.Tensor"):
        validate_human_action([[0.0] * FEATURE_DIM])


def test_package_exports_the_intended_public_api() -> None:
    assert set(human_action.__all__) == EXPECTED_PUBLIC_EXPORTS
    for name in EXPECTED_PUBLIC_EXPORTS:
        assert getattr(human_action, name) is not None
