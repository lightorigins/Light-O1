import pytest

from light_deploy.sampling import action_token_limits, generation_token_limit


def test_action_seconds_convert_to_strict_twenty_fps_bounds() -> None:
    assert action_token_limits(2.0, 20.0) == (40, 400)
    assert action_token_limits(2.01, 8.04) == (41, 160)


@pytest.mark.parametrize(
    ("min_seconds", "max_seconds", "message"),
    [
        (1.9, 20.0, "action_min_seconds"),
        (2.0, 20.1, "action_max_seconds"),
        (10.0, 4.0, "must not exceed"),
    ],
)
def test_action_seconds_reject_invalid_ranges(min_seconds: float, max_seconds: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        action_token_limits(min_seconds, max_seconds)


def test_total_generation_limit_only_reserves_reasoning_in_thinking_mode() -> None:
    assert generation_token_limit(False, 384, 400) == 400
    assert generation_token_limit(True, 384, 400) == 785
    with pytest.raises(ValueError, match="non-negative"):
        generation_token_limit(True, -1, 400)
