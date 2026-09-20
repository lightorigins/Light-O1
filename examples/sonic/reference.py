"""Compact skeleton to the Sonic low-latency policy's reference convention."""

import numpy as np

FPS = 20.0
CONTROL_HZ = 50.0
FUTURE_FRAMES = 4


def validate_reference(reference: dict[str, np.ndarray]) -> int:
    frames = len(reference["joints"])
    for key, shape in (
        ("joints", (frames, 24, 3)),
        ("root_rotation", (frames, 3, 3)),
        ("wrist_rotation", (frames, 2, 3, 3)),
    ):
        array = reference[key]
        if frames < 1 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid Sonic reference {key}: expected nonempty finite {shape}")
        if key != "joints":
            if not np.allclose(array @ np.swapaxes(array, -1, -2), np.eye(3), atol=1e-5):
                raise ValueError(f"Invalid Sonic reference {key}: rotations must be orthonormal")
            if not np.allclose(np.linalg.det(array), 1, atol=1e-5):
                raise ValueError(f"Invalid Sonic reference {key}: rotations must have determinant 1")
    return frames


def wrist_angles(rotations: np.ndarray) -> np.ndarray:
    # Align each forearm's outward axis to G1 wrist +X, and human +Y up to wrist +Z.
    bases = np.array([[[1, 0, 0], [0, 0, -1], [0, 1, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]])
    matrices = bases @ rotations @ bases.transpose(0, 2, 1)
    pitch = np.arcsin(np.clip(matrices[..., 0, 2], -1, 1))
    roll = np.arctan2(-matrices[..., 1, 2], matrices[..., 2, 2])
    yaw = np.arctan2(-matrices[..., 0, 1], matrices[..., 0, 0])
    singular = np.abs(np.cos(pitch)) < 1e-6
    roll = np.where(singular, np.arctan2(matrices[..., 2, 1], matrices[..., 1, 1]), roll)
    yaw = np.where(singular, 0, yaw)
    angles = np.stack((roll, pitch, yaw), axis=-1)
    limits = np.array([1.97222, 1.61443, 1.61443])
    return np.clip(angles, -limits, limits).transpose(0, 2, 1).reshape(-1, 6)


class ReferenceStream:
    def __init__(self, reference: dict[str, np.ndarray], *, replan_frames: int, lookahead_frames: int):
        self.frames = validate_reference(reference)
        if replan_frames < 1 or lookahead_frames < 1:
            raise ValueError("Replan and lookahead frames must be positive")
        self.reference = reference
        self.replan_frames = replan_frames
        self.lookahead_frames = lookahead_frames

    def __iter__(self):
        for start in range(0, self.frames, self.replan_frames):
            end = min(start + self.replan_frames, self.frames)
            first_tick, end_tick = round(start * 2.5), round(end * 2.5)
            horizon = max(
                end_tick - first_tick + FUTURE_FRAMES - 1, round((end - start + self.lookahead_frames) * 2.5)
            )
            # Absolute tick time avoids accumulating rounding drift for odd-sized replans.
            positions = np.clip(
                (np.arange(first_tick, first_tick + horizon) + 1) / 2.5 - 1,
                max(0, start - 1),
                min(end + self.lookahead_frames - 1, self.frames - 1),
            )
            left = np.floor(positions).astype(int)
            right = np.minimum(left + 1, self.frames - 1)
            weight = (positions - left)[:, None, None]
            joints = self.reference["joints"]
            root = self.reference["root_rotation"]
            blended = (1 - weight) * root[left] + weight * root[right]
            u, _, vt = np.linalg.svd(blended)
            correction = np.tile(np.eye(3), (horizon, 1, 1))
            correction[:, 2, 2] = np.linalg.det(u @ vt)
            wrists = wrist_angles(self.reference["wrist_rotation"])
            yield (
                first_tick,
                end_tick,
                {
                    "joints": ((1 - weight) * joints[left] + weight * joints[right]).astype(np.float32),
                    "root_rotation": (u @ correction @ vt).astype(np.float32),
                    "wrists": ((1 - weight[:, :, 0]) * wrists[left] + weight[:, :, 0] * wrists[right]).astype(
                        np.float32
                    ),
                },
            )
