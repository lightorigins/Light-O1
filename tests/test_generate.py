import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from light_deploy.action_contract import ACTION_REPRESENTATION_NAME
from light_deploy.generate import Inference, generate_action


class FakeRunner:
    def __init__(self) -> None:
        self.input_processor = SimpleNamespace(motion_end_id=99, motion_id_bias=10)
        self.kwargs = None
        self.finish_reason = "motion_end"

    def generate(self, prompt: str, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            motion_token_ids=[12, 13, 99],
            thinking_text="reasoning",
            finish_reason=self.finish_reason,
            raw_token_ids=[1, 2, 3],
        )


class FakeDecoder:
    def __init__(self) -> None:
        self.tokens = None

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self.tokens = tokens
        action = np.zeros((2, 138), dtype=np.float32)
        action[:, 2] = 1.0
        rotations = action[:, 4:136].reshape(2, 22, 6)
        rotations[..., 0] = 1.0
        rotations[..., 4] = 1.0
        return action


def test_generate_action_has_one_compact_output_contract_and_disables_top_k() -> None:
    runner = FakeRunner()
    decoder = FakeDecoder()

    output = generate_action(runner, decoder, "wave", enable_thinking=True)

    assert output.representation == ACTION_REPRESENTATION_NAME
    assert output.action.shape == (2, 138)
    assert output.action.dtype == np.float32
    assert output.action_token_ids == [12, 13]
    assert output.action_codebook_ids == [2, 3]
    np.testing.assert_array_equal(decoder.tokens, np.array([2, 3], dtype=np.int64))
    assert runner.kwargs["top_k"] == 0
    assert "min_motion_tokens" in runner.kwargs
    assert "min_action_tokens" not in runner.kwargs
    assert output.finish_reason == "action_end"


def test_generate_action_translates_native_token_limit_finish_reason() -> None:
    runner = FakeRunner()
    runner.finish_reason = "max_tokens_during_motion"

    output = generate_action(runner, FakeDecoder(), "wave", max_action_tokens=2)

    assert output.finish_reason == "max_tokens_during_action"
    assert runner.kwargs["max_motion_tokens"] == 2


def test_generate_action_rejects_empty_prompt_before_runner_call() -> None:
    runner = FakeRunner()

    try:
        generate_action(runner, FakeDecoder(), "   ")
    except ValueError as error:
        assert "prompt" in str(error)
    else:
        raise AssertionError("empty prompt must fail")

    assert runner.kwargs is None


@pytest.mark.parametrize("override", [None, Path("/models/override")])
def test_inference_resolves_packaged_decoder(monkeypatch: pytest.MonkeyPatch, override: Path | None) -> None:
    observed = {}

    class Runner:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            observed["model"] = model
            return cls()

        def close(self):
            observed["closed"] = True

    def load_decoder(path, **kwargs):
        observed["decoder"] = Path(path)
        return FakeDecoder()

    monkeypatch.setitem(sys.modules, "light_deploy.runner", SimpleNamespace(ARModelRunner=Runner))
    monkeypatch.setattr("light_deploy.action_tokenizer.decoder.load_action_decoder", load_decoder)
    kwargs = {} if override is None else {"decoder_path": override}
    with Inference(Path("/models/model"), **kwargs):
        assert observed["decoder"] == (override or Path("/models/model/action_tokenizer"))
    assert observed["closed"]
