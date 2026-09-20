"""Unit tests for the transformers backend's vLLM-free protocol and sampling.

The end-to-end generation is covered by a gpu_e2e test that needs real model
artifacts; these tests exercise the pure pieces without torch CUDA, vLLM, or a
checkpoint.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from light_deploy.processors import MMMotionInputProcessor, MMMotionOutputProcessor  # noqa: E402
from light_deploy.transformers_runner import build_union_logits, sample_token  # noqa: E402


class _Config:
    motion_start_id = 100
    motion_end_id = 116
    motion_id_bias = 102
    motion_pad_id = 101
    split_embeddings = True


class _FakeTokenizer:
    """Minimal tokenizer stand-in for the protocol-only code paths."""

    def __init__(self):
        self._think = [200, 201]
        self._specials = {"<|im_end|>": 5}

    def encode(self, text, add_special_tokens=False):
        assert text == "<think>\n"
        return list(self._think)

    def convert_tokens_to_ids(self, token):
        return self._specials[token]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, return_dict, enable_thinking):
        assert add_generation_prompt and tokenize and not return_dict
        base = [1, 2, 3]
        return base + self._think if enable_thinking else base

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(t) for t in token_ids)


def test_build_union_places_heads_by_global_id():
    text = torch.arange(0, 102, dtype=torch.float32)  # ids 0..101
    motion = torch.arange(1000, 1015, dtype=torch.float32)  # 15 local motion ids
    union = build_union_logits(text, motion, motion_id_bias=102)
    assert union.shape[0] == 102 + 15
    assert torch.equal(union[:102], text[:102])
    # local motion id k lands at global id motion_id_bias + k
    assert union[102].item() == 1000.0
    assert union[102 + 14].item() == 1014.0


def test_build_union_pads_short_text_head():
    text = torch.zeros(50, dtype=torch.float32)
    motion = torch.zeros(15, dtype=torch.float32)
    union = build_union_logits(text, motion, motion_id_bias=102)
    assert union.shape[0] == 102 + 15
    assert math.isinf(union[60].item()) and union[60].item() < 0


def test_sample_token_greedy_and_respects_mask():
    row = torch.tensor([1.0, 5.0, 2.0, 9.0])
    gen = torch.Generator()
    assert sample_token(row, temperature=0.0, top_p=1.0, top_k=0, generator=gen) == 3
    masked = torch.tensor([1.0, 5.0, 2.0, -math.inf])
    for _ in range(64):
        assert sample_token(masked, temperature=1.0, top_p=1.0, top_k=0, generator=gen) != 3


def test_prepare_text_generation_think_opens_think_block():
    proc = MMMotionInputProcessor(_FakeTokenizer(), _Config())
    ids, stop = proc.prepare_text_generation("wave", enable_thinking=True)
    assert ids == [1, 2, 3, 200, 201]  # think opening appended once
    assert stop == {_Config.motion_end_id, 5}


def test_prepare_text_generation_direct_prefeeds_motion_start():
    proc = MMMotionInputProcessor(_FakeTokenizer(), _Config())
    ids, stop = proc.prepare_text_generation("wave", enable_thinking=False)
    assert ids == [1, 2, 3, _Config.motion_start_id]
    assert stop == {_Config.motion_end_id}


def test_output_parse_splits_thinking_and_motion():
    proc = MMMotionOutputProcessor(_FakeTokenizer(), _Config())
    raw = [7, 8, _Config.motion_start_id, 103, 104, _Config.motion_end_id]
    result = proc.parse(raw, enable_thinking=True)
    assert result.motion_token_ids == [103, 104, _Config.motion_end_id]
    assert result.finish_reason == "motion_end"
    assert result.thinking_text == "7 8"


def test_output_parse_direct_is_all_motion():
    proc = MMMotionOutputProcessor(_FakeTokenizer(), _Config())
    raw = [103, 104, _Config.motion_end_id]
    result = proc.parse(raw, enable_thinking=False)
    assert result.thinking_text is None
    assert result.motion_token_ids == raw
    assert result.finish_reason == "motion_end"
