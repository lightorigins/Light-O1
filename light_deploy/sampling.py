from __future__ import annotations

import math

ACTION_FPS = 20
MIN_ACTION_SECONDS = 2.0
MAX_ACTION_SECONDS = 20.0
DEFAULT_REASONING_TOKEN_BUDGET = 384
DEFAULT_REASONING_TEMPERATURE = 0.6
DEFAULT_REASONING_TOP_P = 0.95
DEFAULT_ACTION_TEMPERATURE = 0.6
DEFAULT_ACTION_TOP_P = 0.95


def action_token_limits(min_seconds: float, max_seconds: float) -> tuple[int, int]:
    if not MIN_ACTION_SECONDS <= min_seconds <= MAX_ACTION_SECONDS:
        raise ValueError(f"action_min_seconds must be between {MIN_ACTION_SECONDS:g} and {MAX_ACTION_SECONDS:g}")
    if not MIN_ACTION_SECONDS <= max_seconds <= MAX_ACTION_SECONDS:
        raise ValueError(f"action_max_seconds must be between {MIN_ACTION_SECONDS:g} and {MAX_ACTION_SECONDS:g}")
    if min_seconds > max_seconds:
        raise ValueError("action_min_seconds must not exceed action_max_seconds")
    min_tokens = math.ceil(min_seconds * ACTION_FPS)
    max_tokens = math.floor(max_seconds * ACTION_FPS)
    if min_tokens > max_tokens:
        raise ValueError("action duration range contains no complete 20 FPS frame")
    return min_tokens, max_tokens


def generation_token_limit(enable_thinking: bool, reasoning_token_budget: int, max_action_tokens: int) -> int:
    if reasoning_token_budget < 0:
        raise ValueError("reasoning_token_budget must be non-negative")
    return max_action_tokens + (reasoning_token_budget + 1 if enable_thinking else 0)
