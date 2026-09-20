from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
Image = pytest.importorskip("PIL.Image")
ar_model_runner = pytest.importorskip("light_deploy.runner")
processors = pytest.importorskip("light_deploy.processors")
SamplingParams = vllm.SamplingParams
ARModelRunner = ar_model_runner.ARModelRunner
MMMotionInputProcessor = processors.MMMotionInputProcessor


def test_runner_uses_action_architecture() -> None:
    assert ARModelRunner._ARCHITECTURE == "Qwen3_5ActionForConditionalGeneration"


def test_action_architecture_resolves_registered_model(monkeypatch) -> None:
    registry = vllm.ModelRegistry
    monkeypatch.setattr(registry, "models", dict(registry.models))
    ar_model_runner.register_model()
    architecture = "Qwen3_5ActionForConditionalGeneration"

    assert architecture in registry.get_supported_archs()
    model_class, resolved = registry.resolve_model_cls(architecture, SimpleNamespace(model_impl="vllm"))
    assert resolved == architecture
    assert model_class.__name__ == architecture
    assert model_class.__module__ == "light_deploy.model"


class _FakeGpuRunner:
    use_async_scheduling = True

    def __init__(self) -> None:
        self.scheduler_outputs = []
        self.reset_encoder_cache_calls = 0
        self.sampled_tokens = []
        self.sampled_settings = []
        self.sampled_runtime_settings = []
        self.requests = {}
        self.input_batch = SimpleNamespace(
            req_id_to_index={},
            temperature_cpu=np.zeros(1, dtype=np.float32),
            top_p_cpu=np.ones(1, dtype=np.float32),
            top_k_cpu=np.full(1, 256, dtype=np.int32),
            greedy_reqs=set(),
            random_reqs=set(),
            top_p_reqs=set(),
            top_k_reqs=set(),
            generators={},
            vocab_size=256,
            sampling_metadata=None,
        )
        self.input_batch._make_sampling_metadata = self._make_sampling_metadata

    def execute_model(self, scheduler_output) -> None:
        self.scheduler_outputs.append(scheduler_output)
        if scheduler_output.scheduled_new_reqs:
            params = scheduler_output.scheduled_new_reqs[0].sampling_params
            self.requests[ARModelRunner._SLOT_ID] = SimpleNamespace(sampling_params=params)
            self.input_batch.req_id_to_index[ARModelRunner._SLOT_ID] = 0
            self.input_batch.temperature_cpu[0] = params.temperature
            self.input_batch.top_p_cpu[0] = params.top_p
            self.input_batch.top_k_cpu[0] = params.top_k or self.input_batch.vocab_size
            target = self.input_batch.greedy_reqs if params.temperature == 0 else self.input_batch.random_reqs
            target.add(ARModelRunner._SLOT_ID)
            if params.top_p < 1:
                self.input_batch.top_p_reqs.add(ARModelRunner._SLOT_ID)

    def sample_tokens(self, _grammar_output):
        request = self.requests.get(ARModelRunner._SLOT_ID)
        if request is not None:
            params = request.sampling_params
            self.sampled_runtime_settings.append(
                (
                    float(self.input_batch.temperature_cpu[0]),
                    float(self.input_batch.top_p_cpu[0]),
                    int(self.input_batch.top_k_cpu[0]),
                )
            )
            self.sampled_settings.append((params.temperature, params.top_p, params.top_k))
        token_id = self.sampled_tokens.pop(0) if self.sampled_tokens else 11
        return SimpleNamespace(sampled_token_ids=[[token_id]])

    def queue_tokens(self, *token_ids: int) -> None:
        self.sampled_tokens.extend(token_ids)

    def _make_sampling_metadata(self):
        return SimpleNamespace(
            temperature=float(self.input_batch.temperature_cpu[0]),
            top_p=float(self.input_batch.top_p_cpu[0]),
            top_k=int(self.input_batch.top_k_cpu[0]),
        )

    def reset_encoder_cache(self) -> None:
        self.reset_encoder_cache_calls += 1


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_content = None
        self.prompt_token_ids = [1, 2, 32]

    def apply_chat_template(self, messages, *, tokenize, **_kwargs):
        self.last_content = messages[0]["content"]
        return list(self.prompt_token_ids) if tokenize else "rendered-visual-prompt"

    @staticmethod
    def convert_tokens_to_ids(_token):
        return 30

    @staticmethod
    def encode(text, **_kwargs):
        return [32] if text == "<think>\n" else [31]

    @staticmethod
    def decode(token_ids, **_kwargs):
        pieces = {12: "plan", 31: "</think>"}
        return "".join(pieces.get(token_id, "") for token_id in token_ids)


class _FakeVllmInputProcessor:
    def __init__(self) -> None:
        self.rendered_prompts = []
        self.renderer = self

    def render_cmpl(self, prompts):
        self.rendered_prompts.extend(prompts)
        return ["engine-input"]

    @staticmethod
    def process_inputs(_request_id, _engine_input, sampling_params, **_kwargs):
        return SimpleNamespace(
            prompt_token_ids=[3, 4, 32],
            sampling_params=sampling_params,
            mm_features=[SimpleNamespace(data="vision")],
        )


def test_runner_with_fake_text_and_visual_inputs(monkeypatch) -> None:
    gpu_runner = _FakeGpuRunner()
    tokenizer = _FakeTokenizer()
    hf_config = SimpleNamespace(motion_start_id=10, motion_end_id=20, motion_id_bias=100)
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32, hf_config=hf_config))
    input_processor = MMMotionInputProcessor(tokenizer, vllm_config)
    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    thinking_input = input_processor.prepare_generation("test", "move", (), sampling_params, enable_thinking=True)
    motion_input = input_processor.prepare_generation("test", "move", (), sampling_params, enable_thinking=False)
    tokenizer.prompt_token_ids = [1, 2, 10]
    existing_motion_start = input_processor.prepare_generation(
        "test", "move", (), sampling_params, enable_thinking=False
    )
    assert thinking_input[0] == [1, 2, 32]
    assert thinking_input[3] == {20, 30}
    assert motion_input[0] == [1, 2, 10]
    assert motion_input[3] == {20}
    assert existing_motion_start[0] == [1, 2, 10]
    tokenizer.prompt_token_ids = [1, 2, 32]

    def initialize_kv_cache(runner, _kv_cache_memory_bytes) -> None:
        runner._block_ids = ([1],)
        runner._new_block_ids = [1]

    monkeypatch.setattr(ARModelRunner, "initialize_kv_cache", initialize_kv_cache)
    monkeypatch.setattr(ar_model_runner, "set_current_vllm_config", lambda _config: nullcontext())

    runner = ARModelRunner(
        model_path=Path("fake-model"),
        runner=gpu_runner,
        input_processor=input_processor,
        vllm_config=vllm_config,
        kv_cache_memory_bytes=1,
        device=torch.device("cpu"),
    )

    assert runner.run("move", max_new_tokens=2, temperature=0.0) == [11, 11]

    prefill, decode, reset = gpu_runner.scheduler_outputs
    assert prefill.scheduled_new_reqs[0].prompt_token_ids == [1, 2, 10]
    assert decode.scheduled_cached_reqs.num_computed_tokens == [3]
    assert reset.finished_req_ids == {runner._SLOT_ID}

    image = Image.new("RGB", (56, 56), color="white")
    fake_vllm_input_processor = _FakeVllmInputProcessor()
    monkeypatch.setattr(input_processor, "_get_vllm_input_processor", lambda: fake_vllm_input_processor)
    gpu_runner.scheduler_outputs.clear()
    assert runner.run("play the instrument", images=[image], max_new_tokens=2, temperature=0.0) == [11, 11]

    prefill, decode, reset = gpu_runner.scheduler_outputs
    assert tokenizer.last_content == "play the instrument<|vision_start|><|image_pad|><|vision_end|>"
    assert prefill.scheduled_new_reqs[0].prompt_token_ids == [3, 4, 10]
    assert prefill.scheduled_encoder_inputs == {runner._SLOT_ID: [0]}
    assert decode.scheduled_cached_reqs.num_computed_tokens == [3]
    assert reset.finished_req_ids == {runner._SLOT_ID}
    assert gpu_runner.reset_encoder_cache_calls == 1

    gpu_runner.queue_tokens(101, 20)
    direct = runner.generate("do a cartwheel", enable_thinking=False, max_new_tokens=2, temperature=0.0)
    assert direct.thinking_text is None
    assert direct.motion_token_ids == [101, 20]
    assert direct.finish_reason == "motion_end"

    gpu_runner.queue_tokens(12, 31, 10, 101, 20)
    thinking = runner.generate("do a cartwheel", enable_thinking=True, max_new_tokens=5, temperature=0.0)
    assert thinking.thinking_text == "plan"
    assert thinking.motion_token_ids == [101, 20]
    assert thinking.raw_token_ids == [12, 31, 10, 101, 20]
    assert thinking.finish_reason == "motion_end"

    gpu_runner.queue_tokens(12, 30)
    missing_motion = runner.generate(
        "play the instrument",
        images=[image],
        enable_thinking=True,
        max_new_tokens=2,
        temperature=0.0,
    )
    assert missing_motion.thinking_text == "plan"
    assert missing_motion.motion_token_ids == []
    assert missing_motion.finish_reason == "im_end_before_motion"


def _make_runner(monkeypatch) -> tuple[ARModelRunner, _FakeGpuRunner]:
    gpu_runner = _FakeGpuRunner()
    tokenizer = _FakeTokenizer()
    hf_config = SimpleNamespace(motion_start_id=10, motion_end_id=20, motion_id_bias=100)
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32, hf_config=hf_config))
    input_processor = MMMotionInputProcessor(tokenizer, vllm_config)

    def initialize_kv_cache(runner, _kv_cache_memory_bytes) -> None:
        runner._block_ids = ([1],)
        runner._new_block_ids = [1]

    monkeypatch.setattr(ARModelRunner, "initialize_kv_cache", initialize_kv_cache)
    monkeypatch.setattr(ar_model_runner, "set_current_vllm_config", lambda _config: nullcontext())
    return (
        ARModelRunner(
            model_path=Path("fake-model"),
            runner=gpu_runner,
            input_processor=input_processor,
            vllm_config=vllm_config,
            kv_cache_memory_bytes=1,
            device=torch.device("cpu"),
        ),
        gpu_runner,
    )


def test_thinking_switches_sampling_before_first_motion_token(monkeypatch) -> None:
    runner, gpu_runner = _make_runner(monkeypatch)
    gpu_runner.queue_tokens(2, 10, 101, 102, 103)

    result = runner.generate(
        "move",
        enable_thinking=True,
        reasoning_token_budget=2,
        reasoning_temperature=0.2,
        reasoning_top_p=0.8,
        motion_temperature=0.7,
        motion_top_p=0.9,
        max_motion_tokens=2,
        seed=7,
    )

    assert result.raw_token_ids == [2, 10, 101, 102]
    assert gpu_runner.sampled_runtime_settings[:4] == [
        pytest.approx((0.2, 0.8, 256)),
        pytest.approx((0.2, 0.8, 256)),
        pytest.approx((0.7, 0.9, 256)),
        pytest.approx((0.7, 0.9, 256)),
    ]
    assert result.motion_token_ids == [101, 102]
    assert result.finish_reason == "max_tokens_during_motion"
    assert gpu_runner.sampled_settings[:4] == [
        (0.2, 0.8, 0),
        (0.2, 0.8, 0),
        (0.7, 0.9, 0),
        (0.7, 0.9, 0),
    ]


def test_direct_generation_hard_stops_at_maximum_motion_tokens(monkeypatch) -> None:
    runner, gpu_runner = _make_runner(monkeypatch)
    gpu_runner.queue_tokens(101, 102, 103)

    result = runner.generate("move", motion_temperature=0.6, motion_top_p=0.95, max_motion_tokens=2)

    assert result.raw_token_ids == [101, 102]
    assert result.motion_token_ids == [101, 102]
    assert result.finish_reason == "max_tokens_during_motion"
