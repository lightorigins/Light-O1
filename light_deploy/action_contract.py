"""Validate the NumPy result returned by inference."""

import numpy as np

from light_deploy.action_tokenizer.representation import (
    REPRESENTATION_NAME as ACTION_REPRESENTATION_NAME,
)
from light_deploy.action_tokenizer.representation import validate_human_action


def validate_action_output(action: object, *, representation: str) -> np.ndarray:
    if representation != ACTION_REPRESENTATION_NAME:
        raise ValueError(f"unsupported action representation: {representation}")
    if type(action) is not np.ndarray:
        raise RuntimeError(f"action output for {representation} must be a numpy.ndarray")
    try:
        return validate_human_action(action)
    except (TypeError, ValueError) as error:
        raise RuntimeError(str(error)) from error
