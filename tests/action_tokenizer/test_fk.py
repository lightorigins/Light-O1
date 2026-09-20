from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from light_deploy.action_tokenizer.fk import (
    HUMANOID22_V1,
    SkeletonProfile,
    forward_kinematics,
    rotation_6d_to_matrix,
)
from light_deploy.action_tokenizer.representation import JOINT_NAMES, JOINT_PARENTS

IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


class _StringSubclass(str):
    pass


def _human_action(num_frames: int = 1) -> np.ndarray:
    action = np.zeros((num_frames, 138), dtype=np.float32)
    action[:, 4:136] = np.tile(IDENTITY_6D, 22)
    action[:, 136:138] = 0.5
    return action


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[:2].reshape(6).astype(np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]])


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _profile(**overrides: object) -> SkeletonProfile:
    values = {
        "name": "test_profile",
        "joint_names": JOINT_NAMES,
        "parents": JOINT_PARENTS,
        "rest_offsets": ((0.0, 0.0, 0.0),) * len(JOINT_NAMES),
    }
    values.update(overrides)
    return SkeletonProfile(**values)


def test_rotation_6d_identity_uses_first_two_rows() -> None:
    matrix = rotation_6d_to_matrix(IDENTITY_6D)

    assert matrix.dtype == np.float64
    assert np.allclose(matrix, np.eye(3))


@pytest.mark.parametrize(
    ("rotation", "message"),
    [
        (np.zeros(6), "first basis vector"),
        (np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]), "collinear"),
        (np.zeros(5), "last dimension"),
    ],
)
def test_rotation_6d_rejects_invalid_input(rotation: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rotation_6d_to_matrix(rotation)


def test_humanoid_profile_is_immutable_and_topological() -> None:
    assert HUMANOID22_V1.name == "humanoid22_v1"
    assert HUMANOID22_V1.joint_names is JOINT_NAMES
    assert HUMANOID22_V1.parents is JOINT_PARENTS
    assert len(HUMANOID22_V1.rest_offsets) == len(JOINT_NAMES)
    assert HUMANOID22_V1.parents[0] == -1
    assert all(parent < joint for joint, parent in enumerate(HUMANOID22_V1.parents) if joint > 0)
    assert HUMANOID22_V1.rest_offsets[0] == (0.0, 0.0, 0.0)
    assert HUMANOID22_V1.rest_offsets[1][0] > 0.0
    assert HUMANOID22_V1.rest_offsets[2][0] < 0.0
    with pytest.raises(FrozenInstanceError):
        HUMANOID22_V1.name = "changed"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "non-empty"),
        ({"joint_names": JOINT_NAMES[:-1]}, "joint count"),
        ({"joint_names": ("root", *JOINT_NAMES[1:])}, "joint_names must match"),
        ({"parents": (-1, 0, 0, 1, *JOINT_PARENTS[4:])}, "parents must match"),
        ({"parents": (0, *JOINT_PARENTS[1:])}, "root parent"),
        ({"parents": (-1, 2, *JOINT_PARENTS[2:])}, "topological"),
        (
            {"rest_offsets": ((1.0, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * (len(JOINT_NAMES) - 1)},
            "root offset",
        ),
        (
            {"rest_offsets": ((0.0, 0.0, 0.0),) * (len(JOINT_NAMES) - 1) + ((np.nan, 0.0, 0.0),)},
            "finite",
        ),
    ],
)
def test_profile_rejects_invalid_skeleton(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _profile(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": _StringSubclass("profile")}, "exact str"),
        ({"name": 1}, "exact str"),
        ({"parents": (-1, True, *JOINT_PARENTS[2:])}, "exact ints"),
        ({"parents": (-1, np.int64(0), *JOINT_PARENTS[2:])}, "exact ints"),
        ({"rest_offsets": ((0.0, 0.0, 0.0),) * 21 + ((True, 0.0, 0.0),)}, "real numbers"),
        ({"rest_offsets": ((0.0, 0.0, 0.0),) * 21 + (("0", 0.0, 0.0),)}, "real numbers"),
        ({"rest_offsets": ((0.0, 0.0, 0.0),) * 21 + ((1j, 0.0, 0.0),)}, "real numbers"),
    ],
)
def test_profile_rejects_invalid_schema_types(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(TypeError, match=message):
        _profile(**overrides)


def test_fk_stationary_identity_pose_uses_pelvis_height_and_offsets() -> None:
    action = _human_action()
    action[0, 2] = 1.0

    result = forward_kinematics(action)

    assert result.positions.shape == (1, 22, 3)
    assert result.global_rotations.shape == (1, 22, 3, 3)
    assert result.quaternions_wxyz.shape == (1, 22, 4)
    assert np.allclose(result.root_translation, [[0.0, 1.0, 0.0]])
    assert np.allclose(result.global_rotations, np.eye(3))
    assert np.allclose(result.positions[0, 0], [0.0, 1.0, 0.0])
    assert np.allclose(result.positions[0, 4], [0.09, 0.5, 0.0])
    assert np.allclose(result.quaternions_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert not result.positions.flags.writeable
    assert not result.global_rotations.flags.writeable
    assert not result.root_translation.flags.writeable
    assert not result.quaternions_wxyz.flags.writeable


def test_fk_applies_parent_rotation_to_articulated_chain() -> None:
    action = _human_action()
    left_hip = JOINT_NAMES.index("left_hip")
    left_knee = JOINT_NAMES.index("left_knee")
    left_ankle = JOINT_NAMES.index("left_ankle")
    hip_rotation = _rotation_z(np.pi / 2)
    action[0, 4 + left_hip * 6 : 4 + (left_hip + 1) * 6] = _rotation_6d(hip_rotation)

    result = forward_kinematics(action)

    hip_position = np.array([0.09, -0.08, 0.0])
    expected_knee = hip_position + hip_rotation @ np.array([0.0, -0.42, 0.0])
    expected_ankle = expected_knee + hip_rotation @ np.array([0.0, -0.41, 0.0])
    assert np.allclose(result.positions[0, left_hip], hip_position)
    assert np.allclose(result.positions[0, left_knee], expected_knee)
    assert np.allclose(result.positions[0, left_ankle], expected_ankle)
    assert np.allclose(result.global_rotations[0, left_knee], hip_rotation)


def test_fk_integrates_root_delta_before_current_yaw_and_applies_yaw_to_pose() -> None:
    action = _human_action(3)
    action[:, 0] = 1.0
    action[:, 2] = [0.8, 0.9, 1.0]
    action[0, 3] = np.pi / 2

    result = forward_kinematics(action)

    assert np.allclose(result.root_translation, [[1.0, 0.8, 0.0], [1.0, 0.9, 1.0], [1.0, 1.0, 2.0]])
    assert np.allclose(result.global_rotations[0, 0], _rotation_y(np.pi / 2), atol=1e-6)


def test_fk_composes_local_rotations_down_parent_tree() -> None:
    action = _human_action()
    pelvis_rotation = _rotation_y(np.pi / 4)
    hip_rotation = _rotation_z(np.pi / 3)
    left_hip = JOINT_NAMES.index("left_hip")
    left_knee = JOINT_NAMES.index("left_knee")
    action[0, 4:10] = _rotation_6d(pelvis_rotation)
    action[0, 4 + left_hip * 6 : 4 + (left_hip + 1) * 6] = _rotation_6d(hip_rotation)

    result = forward_kinematics(action)

    assert np.allclose(result.global_rotations[0, 0], pelvis_rotation)
    assert np.allclose(result.global_rotations[0, left_hip], pelvis_rotation @ hip_rotation)
    assert np.allclose(result.global_rotations[0, left_knee], pelvis_rotation @ hip_rotation)


def test_fk_returns_wxyz_quaternion_for_non_identity_rotation() -> None:
    action = _human_action()
    action[0, 4:10] = _rotation_6d(_rotation_z(np.pi / 2))

    quaternion = forward_kinematics(action).quaternions_wxyz[0, 0]

    expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    assert np.allclose(quaternion, expected, atol=1e-6)
    assert np.isfinite(quaternion).all()
    assert np.isclose(np.linalg.norm(quaternion), 1.0)


def test_fk_preserves_global_quaternion_sign_continuity() -> None:
    action = _human_action(2)
    left_hip = JOINT_NAMES.index("left_hip")
    feature = slice(4 + left_hip * 6, 4 + (left_hip + 1) * 6)
    action[0, feature] = _rotation_6d(_rotation_y(np.deg2rad(170.0)))
    action[1, feature] = _rotation_6d(_rotation_y(np.deg2rad(-170.0)))

    quaternions = forward_kinematics(action).quaternions_wxyz

    assert np.dot(quaternions[0, left_hip], quaternions[1, left_hip]) >= 0.0


def test_fk_does_not_mutate_input() -> None:
    action = _human_action(2)
    before = action.copy()

    forward_kinematics(action)

    assert np.array_equal(action, before)


@pytest.mark.parametrize(
    ("action", "exception", "message"),
    [
        (_human_action().astype(np.float64), TypeError, "float32"),
        (np.zeros((1, 274), dtype=np.float32), ValueError, "138"),
    ],
)
def test_fk_enforces_compact_numpy_contract(action: np.ndarray, exception: type[Exception], message: str) -> None:
    with pytest.raises(exception, match=message):
        forward_kinematics(action)


def test_fk_rejects_degenerate_joint_rotation() -> None:
    action = _human_action()
    action[0, 4:10] = 0.0

    with pytest.raises(ValueError, match="first basis vector"):
        forward_kinematics(action)


def test_fk_requires_skeleton_profile_instance() -> None:
    with pytest.raises(TypeError, match="profile must be a SkeletonProfile"):
        forward_kinematics(_human_action(), profile=object())


def test_fk_rejects_finite_offsets_that_overflow_the_parent_chain() -> None:
    huge = np.finfo(np.float64).max
    offsets = list(HUMANOID22_V1.rest_offsets)
    offsets[1] = (huge, 0.0, 0.0)
    offsets[4] = (huge, 0.0, 0.0)
    profile = SkeletonProfile(
        name="overflow_profile",
        joint_names=JOINT_NAMES,
        parents=JOINT_PARENTS,
        rest_offsets=tuple(offsets),
    )

    with pytest.raises(ValueError, match="non-finite"):
        forward_kinematics(_human_action(), profile=profile)
