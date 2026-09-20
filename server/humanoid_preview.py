from typing import Any

import numpy as np

from light_deploy.action_contract import ACTION_REPRESENTATION_NAME
from light_deploy.action_tokenizer.fk import HUMANOID22_V1, SkeletonProfile, forward_kinematics
from light_deploy.action_tokenizer.representation import FPS as COMPACT_ACTION_FPS


def _round_compact_array(array: np.ndarray, decimals: int) -> list[Any]:
    try:
        with np.errstate(over="raise", invalid="raise"):
            rounded = np.round(array.astype(np.float64), decimals)
    except FloatingPointError as error:
        raise ValueError("compact action web payload produced non-finite values") from error
    if not np.isfinite(rounded).all():
        raise ValueError("compact action web payload produced non-finite values")
    return rounded.tolist()


def decode_human_action_for_web(
    action: np.ndarray,
    *,
    fps: int = COMPACT_ACTION_FPS,
    profile: SkeletonProfile = HUMANOID22_V1,
) -> dict[str, Any]:
    if fps != COMPACT_ACTION_FPS:
        raise ValueError(f"compact action fps must equal {COMPACT_ACTION_FPS}, got {fps}")
    result = forward_kinematics(action, profile)
    positions = result.positions
    return {
        "representation": ACTION_REPRESENTATION_NAME,
        "skeleton_profile": profile.name,
        "joint_names": list(profile.joint_names),
        "parents": list(profile.parents),
        "fps": int(fps),
        "num_frames": len(action),
        "positions": _round_compact_array(positions, 5),
        "rootTranslation": _round_compact_array(result.root_translation, 5),
        "rotations": _round_compact_array(result.quaternions_wxyz, 5),
        "bounds": {
            "min": _round_compact_array(positions.min(axis=(0, 1)), 5),
            "max": _round_compact_array(positions.max(axis=(0, 1)), 5),
        },
        "decode_source": f"compact_fk_{profile.name}",
    }
