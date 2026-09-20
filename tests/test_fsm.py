from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
fsm = pytest.importorskip("light_deploy.fsm")
logits_processor_interface = pytest.importorskip("vllm.v1.sample.logits_processor.interface")
BatchUpdate = logits_processor_interface.BatchUpdate
MotionFsmLogitsProcessor = fsm.MotionFsmLogitsProcessor


def _processor() -> MotionFsmLogitsProcessor:
    config = SimpleNamespace(
        motion_start_id=10,
        motion_id_bias=100,
        motion_end_id=200,
        split_embeddings=True,
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=config))
    return MotionFsmLogitsProcessor(vllm_config, torch.device("cpu"), False)


def _track(
    processor: MotionFsmLogitsProcessor,
    prompt_ids: list[int],
    output_ids: list[int],
    *,
    reasoning_token_budget: int | None = None,
    min_motion_tokens: int = 0,
) -> None:
    params = vllm.SamplingParams(
        extra_args={
            "reasoning_token_budget": reasoning_token_budget,
            "min_motion_tokens": min_motion_tokens,
        }
    )
    processor.update_state(BatchUpdate(1, [], [(0, params, prompt_ids, output_ids)], []))


@pytest.mark.parametrize(("budget", "output_ids"), [(0, []), (2, [2, 3])])
def test_reasoning_budget_forces_motion_start(budget: int, output_ids: list[int]) -> None:
    processor = _processor()
    _track(processor, [1], output_ids, reasoning_token_budget=budget)

    logits = processor.apply(torch.zeros((1, 202)))

    assert torch.isfinite(logits[0, 10])
    assert torch.isneginf(logits[0, :10]).all()
    assert torch.isneginf(logits[0, 11:]).all()


def test_motion_end_is_masked_until_minimum_motion_tokens() -> None:
    processor = _processor()
    output_ids = [101]
    _track(processor, [1, 10], output_ids, min_motion_tokens=2)

    before_minimum = processor.apply(torch.zeros((1, 202)))
    assert torch.isneginf(before_minimum[0, 200])
    assert torch.isfinite(before_minimum[0, 102])

    output_ids.append(102)
    at_minimum = processor.apply(torch.zeros((1, 202)))
    assert torch.isfinite(at_minimum[0, 200])
