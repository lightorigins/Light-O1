from dataclasses import dataclass
from numbers import Real

import numpy as np

from light_deploy.action_tokenizer.representation import (
    JOINT_NAMES,
    JOINT_PARENTS,
    NUM_JOINTS,
    REPRESENTATION_NAME,
    unpack_human_action,
    validate_human_action,
)

ROTATION_EPSILON = 1e-8


@dataclass(frozen=True)
class SkeletonProfile:
    """Parent-local offsets in meters for a right-handed, Y-up skeleton.

    Positive X points to the character's left and positive Z is forward at zero heading. Positive yaw rotates local
    positive X toward world positive Z.
    """

    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_offsets: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("skeleton profile name must be an exact str")
        if not self.name:
            raise ValueError("skeleton profile name must be non-empty")
        if not isinstance(self.joint_names, tuple):
            raise TypeError("joint_names must be an immutable tuple")
        if not isinstance(self.parents, tuple):
            raise TypeError("parents must be an immutable tuple")
        if not isinstance(self.rest_offsets, tuple) or any(
            not isinstance(offset, tuple) for offset in self.rest_offsets
        ):
            raise TypeError("rest_offsets must be immutable tuples")
        if len(self.joint_names) != NUM_JOINTS:
            raise ValueError(f"skeleton profile joint count must be {NUM_JOINTS}, got {len(self.joint_names)}")
        if len(self.parents) != NUM_JOINTS:
            raise ValueError(f"skeleton profile parent count must be {NUM_JOINTS}, got {len(self.parents)}")
        if len(self.rest_offsets) != NUM_JOINTS:
            raise ValueError(f"skeleton profile offset count must be {NUM_JOINTS}, got {len(self.rest_offsets)}")
        if any(type(parent) is not int for parent in self.parents):
            raise TypeError("skeleton profile parents must contain exact ints")
        if self.parents[0] != -1:
            raise ValueError(f"skeleton profile root parent must be -1, got {self.parents[0]}")
        for joint_index, parent in enumerate(self.parents[1:], start=1):
            if parent < 0 or parent >= joint_index:
                raise ValueError(
                    f"skeleton profile parents must be topological: joint {joint_index} has parent {parent}"
                )
        if self.joint_names != JOINT_NAMES:
            raise ValueError(
                f"skeleton profile joint_names must match the versioned {REPRESENTATION_NAME} joint order"
            )
        if self.parents != JOINT_PARENTS:
            raise ValueError(f"skeleton profile parents must match the versioned {REPRESENTATION_NAME} topology")
        if any(len(offset) != 3 for offset in self.rest_offsets):
            raise ValueError(f"skeleton profile offsets must have shape ({NUM_JOINTS}, 3)")
        if any(type(value) is bool or not isinstance(value, Real) for offset in self.rest_offsets for value in offset):
            raise TypeError("skeleton profile offset components must be real numbers, excluding bool")
        offsets = np.asarray(self.rest_offsets, dtype=np.float64)
        if offsets.shape != (NUM_JOINTS, 3):
            raise ValueError(f"skeleton profile offsets must have shape ({NUM_JOINTS}, 3), got {offsets.shape}")
        if not np.isfinite(offsets).all():
            raise ValueError("skeleton profile offsets must contain only finite values")
        if not np.array_equal(offsets[0], np.zeros(3)):
            raise ValueError("skeleton profile root offset must be zero")


HUMANOID22_V1 = SkeletonProfile(
    name="humanoid22_v1",
    joint_names=JOINT_NAMES,
    parents=JOINT_PARENTS,
    rest_offsets=(
        (0.0, 0.0, 0.0),
        (0.09, -0.08, 0.0),
        (-0.09, -0.08, 0.0),
        (0.0, 0.13, 0.0),
        (0.0, -0.42, 0.0),
        (0.0, -0.42, 0.0),
        (0.0, 0.14, 0.0),
        (0.0, -0.41, 0.0),
        (0.0, -0.41, 0.0),
        (0.0, 0.06, 0.0),
        (0.0, -0.06, 0.13),
        (0.0, -0.06, 0.13),
        (0.0, 0.21, 0.0),
        (0.08, 0.06, 0.0),
        (-0.08, 0.06, 0.0),
        (0.0, 0.09, 0.05),
        (0.14, 0.0, 0.0),
        (-0.14, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (-0.25, 0.0, 0.0),
        (0.24, 0.0, 0.0),
        (-0.24, 0.0, 0.0),
    ),
)


@dataclass(frozen=True)
class ForwardKinematicsResult:
    positions: np.ndarray
    root_translation: np.ndarray
    global_rotations: np.ndarray
    quaternions_wxyz: np.ndarray


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    array = np.asarray(rotation_6d, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 6:
        raise ValueError(f"6D rotation last dimension must be 6, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("6D rotation must contain only finite values")
    first = array[..., :3]
    second = array[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < ROTATION_EPSILON):
        raise ValueError("degenerate 6D rotation: first basis vector has zero norm")
    basis_x = first / first_norm
    second_orthogonal = second - np.sum(basis_x * second, axis=-1, keepdims=True) * basis_x
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < ROTATION_EPSILON):
        raise ValueError("degenerate 6D rotation: basis vectors are collinear")
    basis_y = second_orthogonal / second_norm
    basis_z = np.cross(basis_x, basis_y, axis=-1)
    return np.stack((basis_x, basis_y, basis_z), axis=-2)


def _heading_matrix(yaw: float) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array([[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]])


def _matrices_to_quaternions_wxyz(matrices: np.ndarray) -> np.ndarray:
    flat_matrices = matrices.reshape(-1, 3, 3)
    quaternions = np.empty((len(flat_matrices), 4), dtype=np.float64)
    for index, matrix in enumerate(flat_matrices):
        trace = np.trace(matrix)
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif matrix[1, 1] > matrix[2, 2]:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("forward kinematics produced a non-finite or zero-norm quaternion")
        quaternions[index] = quaternion / norm
    return quaternions.reshape(*matrices.shape[:-2], 4)


def _preserve_quaternion_sign_continuity(quaternions: np.ndarray) -> np.ndarray:
    continuous = quaternions.copy()
    for joint_index in range(continuous.shape[1]):
        for frame_index in range(1, continuous.shape[0]):
            if np.dot(continuous[frame_index - 1, joint_index], continuous[frame_index, joint_index]) < 0.0:
                continuous[frame_index, joint_index] *= -1.0
    return continuous


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def forward_kinematics(action: np.ndarray, profile: SkeletonProfile = HUMANOID22_V1) -> ForwardKinematicsResult:
    if not isinstance(profile, SkeletonProfile):
        raise TypeError("profile must be a SkeletonProfile")
    if not isinstance(action, np.ndarray):
        raise TypeError("forward kinematics requires action as a numpy.ndarray")
    validate_human_action(action)
    fields = unpack_human_action(action)
    local_rotations = rotation_6d_to_matrix(fields.joint_rot6d_local)
    offsets = np.asarray(profile.rest_offsets, dtype=np.float64)
    num_frames = len(action)
    positions = np.empty((num_frames, NUM_JOINTS, 3), dtype=np.float64)
    global_rotations = np.empty((num_frames, NUM_JOINTS, 3, 3), dtype=np.float64)
    root_translation = np.empty((num_frames, 3), dtype=np.float64)
    root_position = np.zeros(3, dtype=np.float64)
    accumulated_yaw = 0.0

    try:
        with np.errstate(over="raise", invalid="raise"):
            for frame_index in range(num_frames):
                heading_before_delta = _heading_matrix(accumulated_yaw)
                local_delta = fields.root_delta_xz_local[frame_index]
                root_position += heading_before_delta @ np.array([local_delta[0], 0.0, local_delta[1]])
                root_position[1] = fields.pelvis_height_y[frame_index, 0]
                root_translation[frame_index] = root_position

                accumulated_yaw += fields.yaw_delta_rad_per_frame[frame_index, 0]
                heading_after_delta = _heading_matrix(accumulated_yaw)
                positions[frame_index, 0] = root_position
                global_rotations[frame_index, 0] = heading_after_delta @ local_rotations[frame_index, 0]
                for joint_index in range(1, NUM_JOINTS):
                    parent = profile.parents[joint_index]
                    positions[frame_index, joint_index] = (
                        positions[frame_index, parent] + global_rotations[frame_index, parent] @ offsets[joint_index]
                    )
                    global_rotations[frame_index, joint_index] = (
                        global_rotations[frame_index, parent] @ local_rotations[frame_index, joint_index]
                    )

            quaternions = _matrices_to_quaternions_wxyz(global_rotations)
            quaternions = _preserve_quaternion_sign_continuity(quaternions)
    except FloatingPointError as error:
        raise ValueError("forward kinematics produced non-finite values") from error

    outputs = (positions, root_translation, global_rotations, quaternions)
    quaternion_norms = np.linalg.norm(quaternions, axis=-1)
    if not all(np.isfinite(output).all() for output in outputs):
        raise ValueError("forward kinematics produced non-finite values")
    if not np.isfinite(quaternion_norms).all() or np.any(quaternion_norms <= 0.0):
        raise ValueError("forward kinematics produced a non-finite or zero-norm quaternion")
    return ForwardKinematicsResult(
        positions=_readonly(positions),
        root_translation=_readonly(root_translation),
        global_rotations=_readonly(global_rotations),
        quaternions_wxyz=_readonly(quaternions),
    )
