import importlib.util

import pytest

from light_deploy.protocol import GenerateRequest


def test_action_tokenizer_is_the_only_public_package():
    assert importlib.util.find_spec("light_deploy.action_tokenizer") is not None
    assert importlib.util.find_spec("light_deploy.compact_motion") is None


def test_public_request_uses_action_sampling_fields():
    request = GenerateRequest(prompt="wave", action_temperature=0.6, action_top_p=0.95)
    assert request.action_min_seconds == 2
    assert request.action_max_seconds == 20
    assert not any("motion" in key.lower() for key in request.model_dump())
    with pytest.raises(ValueError):
        GenerateRequest(prompt="wave", motion_temperature=0.6)
