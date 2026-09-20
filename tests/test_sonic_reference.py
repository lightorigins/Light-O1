import importlib
import json

import numpy as np
import pytest

from examples.sonic.adapter import from_human_action


def action(frames=5):
    value = np.zeros((frames, 138), np.float32)
    value[:, 2] = 0.95
    value[:, 4:136] = np.tile([1, 0, 0, 0, 1, 0], 22)
    return value


def reference_module():
    return importlib.import_module("examples.sonic.reference")


def test_canonical_reference_preserves_turning_and_local_joint_shape():
    straight = action()
    turning = straight.copy()
    turning[:, :2] = [0.1, 0.2]
    turning[:, 3] = 0.2
    a, b = from_human_action(straight), from_human_action(turning)
    np.testing.assert_allclose(a["joints"], b["joints"], atol=1e-7)
    assert b["joints"].shape == (5, 24, 3)
    assert b["root_rotation"].shape == (5, 3, 3)
    assert not np.allclose(b["root_rotation"][0], b["root_rotation"][-1])
    np.testing.assert_allclose(b["joints"][:, 0], 0, atol=1e-7)


def test_grippers_do_not_change_sonic_reference():
    value = action()
    changed = value.copy()
    changed[:, -2:] = 1
    for key, array in from_human_action(value).items():
        np.testing.assert_array_equal(array, from_human_action(changed)[key])


def test_geometric_wrists_map_forearm_twist_and_vertical_bend():
    reference = reference_module()
    angle = 0.3
    c, s = np.cos(angle), np.sin(angle)
    twist = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    rotations = np.tile(twist, (1, 2, 1, 1))
    np.testing.assert_allclose(reference.wrist_angles(rotations)[0], [angle, -angle, 0, 0, 0, 0], atol=1e-7)
    bend = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotations[:] = bend
    np.testing.assert_allclose(reference.wrist_angles(rotations)[0], [0, 0, -angle, angle, 0, 0], atol=1e-7)


def test_reference_rejects_non_rotations():
    reference = reference_module()
    plan = from_human_action(action())
    plan["root_rotation"][:] = 0
    with pytest.raises(ValueError, match="orthonormal"):
        reference.ReferenceStream(plan, replan_frames=8, lookahead_frames=12)


@pytest.mark.parametrize("frames", [1, 3, 5, 109])
def test_reference_resampling_has_global_tick_count_and_finite_windows(frames):
    reference = reference_module()
    plan = from_human_action(action(frames))
    stream = reference.ReferenceStream(plan, replan_frames=3, lookahead_frames=2)
    chunks = list(stream)
    ticks = [tick for start, end, _ in chunks for tick in range(start, end)]
    assert ticks == list(range(round(frames * 2.5)))
    for start, end, chunk in chunks:
        assert len(chunk["joints"]) >= end - start + 3
        assert np.isfinite(chunk["root_rotation"]).all()
        np.testing.assert_allclose(
            chunk["root_rotation"] @ chunk["root_rotation"].transpose(0, 2, 1),
            np.broadcast_to(np.eye(3), chunk["root_rotation"].shape),
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((0, 138), np.float32),
        np.zeros((3, 274), np.float32),
        np.full((3, 138), np.nan, np.float32),
        np.zeros((3, 138), np.float32),
    ],
)
def test_invalid_action_rejected(value):
    with pytest.raises((ValueError, RuntimeError)):
        from_human_action(value)


def test_plan_is_exclusive_compact_only_and_roundtrips(tmp_path):
    sonic = importlib.import_module("server.sonic")
    path = tmp_path / "plan.npz"
    sonic.write_sonic_plan(action(), path, provenance={"generation_id": "abc"})
    with np.load(path, allow_pickle=False) as plan:
        header = json.loads(plan["metadata"].item())
        assert header["representation"] == "human_action_138_v1"
        assert header["provenance"]["generation_id"] == "abc"
        assert set(plan.files) == {"metadata", "joints", "root_rotation", "wrist_rotation"}
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        sonic.write_sonic_plan(action(), path, provenance={})
    assert path.read_bytes() == before
