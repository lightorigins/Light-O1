import numpy as np
import pytest

from light_deploy.action_contract import ACTION_REPRESENTATION_NAME
from server.humanoid_preview import decode_human_action_for_web


def _action(frames: int = 2) -> np.ndarray:
    action = np.zeros((frames, 138), dtype=np.float32)
    action[:, 2] = 1.0
    rotations = action[:, 4:136].reshape(frames, 22, 6)
    rotations[..., 0] = 1.0
    rotations[..., 4] = 1.0
    action[:, 136:138] = 0.5
    return action


def test_decode_human_action_returns_procedural_skeleton_without_grippers() -> None:
    action = _action()
    action[1, 0] = 0.1

    payload = decode_human_action_for_web(action)

    assert payload["representation"] == ACTION_REPRESENTATION_NAME
    assert payload["skeleton_profile"] == "humanoid22_v1"
    assert payload["fps"] == 20
    assert payload["num_frames"] == 2
    assert len(payload["joint_names"]) == 22
    assert len(payload["positions"]) == 2
    assert payload["rootTranslation"][1][0] == pytest.approx(0.1)
    assert "gripper" not in payload


@pytest.mark.parametrize("fps", [0, 19, 21, 30])
def test_decode_human_action_requires_canonical_fps(fps: int) -> None:
    with pytest.raises(ValueError, match="fps must equal 20"):
        decode_human_action_for_web(_action(), fps=fps)


@pytest.mark.parametrize(
    ("action", "error"),
    [
        (np.zeros((1, 137), dtype=np.float32), ValueError),
        (np.zeros((1, 138), dtype=np.float64), TypeError),
        (np.full((1, 138), np.nan, dtype=np.float32), ValueError),
    ],
)
def test_decode_human_action_rejects_noncanonical_input(action: np.ndarray, error: type[Exception]) -> None:
    with pytest.raises(error):
        decode_human_action_for_web(action)
