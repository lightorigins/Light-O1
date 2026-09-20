"""Batch-1 AR model runner backed by vLLM Modeling."""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Self

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams, ir
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.config.model import ModelDType
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs
from vllm.multimodal.inputs import ImageItem, MultiModalFeatureSpec
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingType
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, get_kv_cache_groups
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager

from light_deploy.errors import ContextLengthError
from light_deploy.fsm import MotionFsmLogitsProcessor, in_open_motion_span
from light_deploy.model import register_model
from light_deploy.processors import (
    GenerationResult,
    MMMotionInputProcessor,
    MMMotionOutputProcessor,
)


class ARModelRunner:
    """Run one Qwen3.5 Action AR model without a vLLM Engine or Scheduler.

    The model is the vLLM-native ``Qwen3_5ActionForConditionalGeneration``
    registered by Light Infer. vLLM's ``GPUModelRunner`` owns weight loading,
    attention metadata, hybrid KV/GDN cache, forward, and sampling. This class
    supplies only the fixed single-request prefill/decode schedule.

    The runner owns vLLM's process-global distributed state and must be closed
    before another in-process vLLM runtime is created.
    """

    _ARCHITECTURE = "Qwen3_5ActionForConditionalGeneration"
    _SLOT_ID = "qwen35-ar"

    def __init__(
        self,
        *,
        model_path: Path,
        runner: GPUModelRunner,
        input_processor: MMMotionInputProcessor,
        vllm_config: VllmConfig,
        kv_cache_memory_bytes: int,
        device: torch.device,
    ) -> None:
        self.model_path = model_path
        self._runner: GPUModelRunner | None = runner
        self.input_processor = input_processor
        self.output_processor = MMMotionOutputProcessor(input_processor.tokenizer, vllm_config.model_config.hf_config)
        self._vllm_config = vllm_config
        self.max_model_len = vllm_config.model_config.max_model_len
        self.device = device
        self.initialize_kv_cache(kv_cache_memory_bytes)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dtype: ModelDType = "bfloat16",
        max_model_len: int = 320,
        kv_cache_memory_bytes: int = 1 << 30,
        deterministic: bool = False,
    ) -> Self:
        """Load the checkpoint and prepare the AR runtime.

        Production uses compile, single-token CUDA Graph, and asynchronous
        scheduling. ``deterministic=True`` selects the eager synchronous
        reference path used by golden tests.
        """
        checkpoint = Path(model_path).expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Checkpoint is missing config.json: {checkpoint}")
        target_device = torch.device(device)
        torch.cuda.set_device(target_device)
        target_device = torch.device("cuda", torch.cuda.current_device())

        register_model()
        # EngineArgs is used only as vLLM's validated VllmConfig factory. No
        # LLMEngine, EngineCore, Executor, Worker, or Scheduler is instantiated.
        compilation_config = (
            CompilationConfig(
                mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.NONE,
            )
            if deterministic
            else CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                cudagraph_capture_sizes=[1],
                max_cudagraph_capture_size=1,
            )
        )
        vllm_config = EngineArgs(
            model=str(checkpoint),
            tokenizer=str(checkpoint),
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=max_model_len,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            # Session and cross-request ViT caching are intentionally outside
            # this model executor. Each call preprocesses and encodes its own
            # images, then releases the temporary encoder outputs.
            mm_processor_cache_gb=0,
            enforce_eager=deterministic,
            enable_prefix_caching=False,
            async_scheduling=not deterministic,
            disable_custom_all_reduce=True,
            logits_processors=[MotionFsmLogitsProcessor],
            compilation_config=compilation_config,
        ).create_engine_config()
        config = vllm_config.model_config.hf_config
        if (
            cls._ARCHITECTURE not in (config.architectures or ())
            or not config.split_embeddings
            or getattr(config, "action_config", None) is not None
        ):
            raise ValueError("Checkpoint is not a split-embedding Qwen3.5 Action AR model")
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            local_files_only=True,
            padding_side="right",
        )

        runner: GPUModelRunner | None = None
        instance: Self
        try:
            with set_current_vllm_config(vllm_config):
                # GPUModelRunner is normally created by WorkerBase, which
                # installs these process-wide execution policies first.
                vllm_config.kernel_config.ir_op_priority.set_default()
                ir.set_default_torch_wrap(vllm_config.compilation_config.ir_enable_torch_wrap)
                init_worker_distributed_environment(
                    vllm_config,
                    rank=0,
                    distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
                    local_rank=target_device.index or 0,
                    backend=current_platform.dist_backend,
                )
                set_random_seed(vllm_config.model_config.seed)
                init_workspace_manager(target_device, 1)
                runner = GPUModelRunner(vllm_config, target_device)
                runner.load_model()
                instance = cls(
                    model_path=checkpoint,
                    runner=runner,
                    input_processor=MMMotionInputProcessor(
                        tokenizer, vllm_config.model_config.hf_config, vllm_config=vllm_config
                    ),
                    vllm_config=vllm_config,
                    kv_cache_memory_bytes=kv_cache_memory_bytes,
                    device=target_device,
                )
                cls._prepare_runtime(runner, vllm_config)
        except Exception:
            if runner is not None:
                with suppress(Exception):
                    cls._shutdown_vllm_runner(runner)
                del runner
            cleanup_dist_env_and_memory()
            raise

        return instance

    @property
    def modeling_class(self) -> str:
        """Fully qualified vLLM Modeling class loaded by this runner."""
        model_type = type(self._require_runner().get_model())
        return f"{model_type.__module__}.{model_type.__name__}"

    @property
    def cache_num_blocks(self) -> int:
        """Number of physical GPU cache blocks, including vLLM's null block."""
        return self._kv_cache_config.num_blocks

    @property
    def cache_memory_bytes(self) -> int:
        """Physical GPU bytes described by vLLM's non-aliased cache tensors."""
        return sum(tensor.size for tensor in self._kv_cache_config.kv_cache_tensors)

    @property
    def closed(self) -> bool:
        return self._runner is None

    def initialize_kv_cache(self, kv_cache_memory_bytes: int) -> None:
        """Initialize the fixed Batch-1 Attention and GDN cache."""
        runner = self._require_runner()
        current_platform.update_block_size_for_backend(self._vllm_config)
        register_all_kvcache_specs(self._vllm_config)
        kv_cache_specs = runner.get_kv_cache_spec()
        required_num_blocks = self._required_num_blocks(
            get_kv_cache_groups(self._vllm_config, kv_cache_specs),
            self._vllm_config,
            self.max_model_len,
        )
        self._vllm_config.cache_config.num_gpu_blocks_override = required_num_blocks
        kv_cache_config = get_kv_cache_configs(
            self._vllm_config,
            [kv_cache_specs],
            [kv_cache_memory_bytes],
        )[0]
        if kv_cache_config.num_blocks != required_num_blocks:
            raise RuntimeError(
                "vLLM did not honor the static cache block override: "
                f"expected {required_num_blocks}, got {kv_cache_config.num_blocks}"
            )

        self._vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        # TODO: Replace vLLM allocation with a Light-owned Qwen3.5 Attention/GDN cache implementation.
        runner.initialize_kv_cache(kv_cache_config)
        if kv_cache_config.needs_kv_cache_zeroing:
            runner._init_kv_zero_meta()

        self._kv_cache_config = kv_cache_config
        self._block_ids, self._new_block_ids = self._build_static_blocks(
            kv_cache_config,
            self._vllm_config,
            self.max_model_len,
        )

    def run(
        self,
        prompt: str,
        *,
        images: Sequence[ImageItem] | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Generate global motion token IDs for one text-and-image prompt."""
        return self.generate(
            prompt,
            images=images,
            enable_thinking=False,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            on_token=on_token,
        ).motion_token_ids

    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageItem] | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int | None = None,
        reasoning_token_budget: int | None = None,
        reasoning_temperature: float | None = None,
        reasoning_top_p: float | None = None,
        min_motion_tokens: int = 0,
        max_motion_tokens: int | None = None,
        motion_temperature: float | None = None,
        motion_top_p: float | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        """Generate one direct-Motion or autonomous Thinking-to-Motion result."""
        if reasoning_token_budget is not None and reasoning_token_budget < 0:
            raise ValueError("reasoning_token_budget must be non-negative")
        if min_motion_tokens < 0:
            raise ValueError("min_motion_tokens must be non-negative")
        if max_motion_tokens is not None and max_motion_tokens <= 0:
            raise ValueError("max_motion_tokens must be positive")
        if max_motion_tokens is not None and min_motion_tokens > max_motion_tokens:
            raise ValueError("min_motion_tokens must not exceed max_motion_tokens")
        if enable_thinking and max_motion_tokens is not None and reasoning_token_budget is None:
            raise ValueError("reasoning_token_budget is required with max_motion_tokens in thinking mode")

        phase_sampling = any(
            value is not None for value in (reasoning_temperature, reasoning_top_p, motion_temperature, motion_top_p)
        )
        reasoning_temperature = temperature if reasoning_temperature is None else reasoning_temperature
        reasoning_top_p = top_p if reasoning_top_p is None else reasoning_top_p
        motion_temperature = temperature if motion_temperature is None else motion_temperature
        motion_top_p = top_p if motion_top_p is None else motion_top_p
        phase_top_k = 0 if phase_sampling else top_k
        total_token_limit = max_new_tokens
        if max_motion_tokens is not None:
            total_token_limit = max_motion_tokens
            if enable_thinking:
                assert reasoning_token_budget is not None
                total_token_limit += reasoning_token_budget + 1
        fsm_args = {
            "reasoning_token_budget": reasoning_token_budget if enable_thinking else None,
            "min_motion_tokens": min_motion_tokens,
        }
        initial_temperature = reasoning_temperature if enable_thinking else motion_temperature
        initial_top_p = reasoning_top_p if enable_thinking else motion_top_p
        sampling_params = SamplingParams(
            max_tokens=total_token_limit,
            temperature=initial_temperature,
            top_p=initial_top_p,
            top_k=phase_top_k,
            seed=seed,
            extra_args=fsm_args,
        )
        motion_sampling_params = SamplingParams(
            max_tokens=total_token_limit,
            temperature=motion_temperature,
            top_p=motion_top_p,
            top_k=phase_top_k,
            seed=seed,
            extra_args=fsm_args,
        )
        if not enable_thinking or (
            sampling_params.temperature == motion_sampling_params.temperature
            and sampling_params.top_p == motion_sampling_params.top_p
            and sampling_params.top_k == motion_sampling_params.top_k
        ):
            motion_sampling_params = None
        prompt_token_ids, sampling_params, mm_features, stop_token_ids = self.input_processor.prepare_generation(
            self._SLOT_ID,
            prompt,
            images or (),
            sampling_params,
            enable_thinking=enable_thinking,
        )
        raw_token_ids = self.generate_tokens(
            prompt_token_ids,
            sampling_params,
            mm_features,
            stop_token_ids,
            max_new_tokens=total_token_limit,
            has_visual_input=bool(images),
            motion_sampling_params=motion_sampling_params,
            max_motion_tokens=max_motion_tokens,
            on_token=on_token,
        )
        return self.output_processor.parse(raw_token_ids, enable_thinking=enable_thinking)

    def generate_tokens(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        mm_features: list[MultiModalFeatureSpec],
        stop_token_ids: set[int],
        *,
        max_new_tokens: int,
        has_visual_input: bool,
        motion_sampling_params: SamplingParams | None = None,
        max_motion_tokens: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Execute one prepared Batch-1 token stream without interpreting its protocol."""
        runner = self._require_runner()
        self._validate_context_length(len(prompt_token_ids) + max_new_tokens)

        generated: list[int] = []
        motion_started = in_open_motion_span(
            prompt_token_ids,
            generated,
            motion_start_id=self.input_processor.motion_start_id,
            motion_end_id=self.input_processor.motion_end_id,
        )
        motion_tokens = 0
        motion_sampling_applied = motion_sampling_params is None
        execution_error: BaseException | None = None
        try:
            with torch.inference_mode(), set_current_vllm_config(self._vllm_config):
                output = self._prefill(
                    prompt_token_ids,
                    sampling_params,
                    mm_features,
                    list(range(len(mm_features))),
                )
                for num_output_tokens in range(max_new_tokens):
                    can_schedule_ahead = (
                        runner.use_async_scheduling
                        and motion_sampling_applied
                        and num_output_tokens + 1 < max_new_tokens
                    )
                    next_output = (
                        self._decode(len(prompt_token_ids), num_output_tokens + 1) if can_schedule_ahead else None
                    )
                    token_id = self._extract_sampled_token(output)
                    generated.append(token_id)
                    if on_token is not None:
                        on_token(token_id)

                    if token_id == self.input_processor.motion_start_id:
                        motion_started = True
                        if motion_sampling_params is not None and not motion_sampling_applied:
                            self._set_active_sampling_params(motion_sampling_params)
                            motion_sampling_applied = True
                    elif token_id == self.input_processor.motion_end_id:
                        motion_started = False
                    elif motion_started and token_id >= self.input_processor.motion_id_bias:
                        motion_tokens += 1

                    reached_motion_limit = max_motion_tokens is not None and motion_tokens >= max_motion_tokens
                    if token_id in stop_token_ids or reached_motion_limit:
                        if next_output is not None:
                            self._extract_sampled_token(next_output)
                        break
                    if num_output_tokens + 1 < max_new_tokens:
                        output = next_output or self._decode(len(prompt_token_ids), num_output_tokens + 1)
        except BaseException as error:
            execution_error = error
            raise
        finally:
            try:
                self._reset(has_visual_input=has_visual_input)
            except Exception as cleanup_error:
                close_error: Exception | None = None
                try:
                    self.close()
                except Exception as error:
                    close_error = error
                if execution_error is None:
                    if close_error is not None:
                        cleanup_error.add_note(f"Runner close also failed: {close_error!r}")
                    raise
                note = f"Inference cleanup also failed and the runner was closed: {cleanup_error!r}"
                if close_error is not None:
                    note += f"; runner close also failed: {close_error!r}"
                execution_error.add_note(note)

        return generated

    @classmethod
    def _prepare_runtime(
        cls,
        runner: GPUModelRunner,
        vllm_config: VllmConfig,
    ) -> None:
        """Apply the fixed production startup lifecycle without a Worker."""
        from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune
        from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
        from vllm.model_executor.warmup.v1_block_table_warmup import (
            warm_v1_block_table_kernels,
        )

        cls._compile_model(runner, vllm_config)
        warm_v1_block_table_kernels(
            runner.device,
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        qwen_triton_warmup(runner, vllm_config.model_config)
        flashinfer_autotune(runner)
        runner.capture_model()

        hidden_states, last_hidden_states = runner._dummy_run(
            num_tokens=1,
            skip_eplb=True,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        if runner.is_pooling_model:
            runner._dummy_pooler_run(hidden_states)
        else:
            runner._dummy_sampler_run(hidden_states=last_hidden_states)

    @staticmethod
    def _compile_model(runner: GPUModelRunner, vllm_config: VllmConfig) -> None:
        """Compile every configured token range once, following GPUWorker."""
        compilation_config = vllm_config.compilation_config
        warmup_sizes = list(compilation_config.compile_sizes or [])
        capture_sizes = set(compilation_config.cudagraph_capture_sizes or [])
        if compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            warmup_sizes = [size for size in warmup_sizes if size not in capture_sizes]

        covered_sizes = capture_sizes | set(warmup_sizes)
        for compile_range in compilation_config.get_compile_ranges():
            if not any(size in compile_range for size in covered_sizes):
                warmup_sizes.append(compile_range.end)

        for size in sorted(warmup_sizes, reverse=True):
            runner._dummy_run(int(size), skip_eplb=True, remove_lora=False)
        runner.maybe_remove_all_loras(runner.lora_config)

    def close(self) -> None:
        """Release the vLLM runner and its process-global distributed state."""
        if self._runner is None:
            return
        runner = self._runner
        self._runner = None
        try:
            self.input_processor.close()
        finally:
            try:
                self._shutdown_vllm_runner(runner)
            finally:
                del runner
                gc.collect()
                cleanup_dist_env_and_memory()

    @staticmethod
    def _shutdown_vllm_runner(runner: GPUModelRunner) -> None:
        """Release tensors owned by the vLLM runner."""
        model = getattr(runner, "model", None)
        if model is not None:
            # vLLM 0.26 caches SupportsMultiModal.get_language_model() in a
            # process-global plain dict. GPUModelRunner.shutdown() clears
            # runner.model but not this reference, so remove it before cleanup.
            from vllm.model_executor.models import interfaces as model_interfaces

            language_model_cache = getattr(model_interfaces, "_language_model_by_module", None)
            if language_model_cache is not None:
                language_model_cache.pop(model, None)
        runner.shutdown()

    def _require_runner(self) -> GPUModelRunner:
        if self._runner is None:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return self._runner

    def _set_active_sampling_params(self, sampling_params: SamplingParams) -> None:
        runner = self._require_runner()
        request = runner.requests[self._SLOT_ID]
        input_batch = runner.input_batch
        index = input_batch.req_id_to_index[self._SLOT_ID]
        request.sampling_params = sampling_params

        input_batch.greedy_reqs.discard(self._SLOT_ID)
        input_batch.random_reqs.discard(self._SLOT_ID)
        if sampling_params.sampling_type == SamplingType.GREEDY:
            input_batch.temperature_cpu[index] = 0.0
            input_batch.greedy_reqs.add(self._SLOT_ID)
        else:
            input_batch.temperature_cpu[index] = sampling_params.temperature
            input_batch.random_reqs.add(self._SLOT_ID)

        input_batch.top_p_cpu[index] = sampling_params.top_p
        input_batch.top_p_reqs.discard(self._SLOT_ID)
        if sampling_params.top_p < 1:
            input_batch.top_p_reqs.add(self._SLOT_ID)

        top_k = sampling_params.top_k
        input_batch.top_k_reqs.discard(self._SLOT_ID)
        if 0 < top_k < input_batch.vocab_size:
            input_batch.top_k_reqs.add(self._SLOT_ID)
        else:
            top_k = input_batch.vocab_size
        input_batch.top_k_cpu[index] = top_k

        if sampling_params.sampling_type == SamplingType.RANDOM_SEED and index not in input_batch.generators:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
            input_batch.generators[index] = generator
        input_batch.sampling_metadata = input_batch._make_sampling_metadata()

    def _validate_context_length(self, total_tokens: int) -> None:
        if total_tokens > self.max_model_len:
            raise ContextLengthError(total_tokens, self.max_model_len)

    @staticmethod
    def _build_static_blocks(
        kv_cache_config: KVCacheConfig,
        vllm_config: VllmConfig,
        capacity: int,
    ) -> tuple[tuple[list[int], ...], list[int]]:
        groups: list[list[int]] = []
        allocated: list[int] = []
        next_block_id = 1  # vLLM reserves block zero as its null/padding block.
        for group in kv_cache_config.kv_cache_groups:
            num_blocks = group.kv_cache_spec.max_num_blocks_per_req(vllm_config, capacity)
            block_ids = list(range(next_block_id, next_block_id + num_blocks))
            groups.append(block_ids)
            allocated.extend(block_ids)
            next_block_id += num_blocks
        return tuple(groups), allocated

    @staticmethod
    def _required_num_blocks(
        groups: list[KVCacheGroupSpec],
        vllm_config: VllmConfig,
        capacity: int,
    ) -> int:
        return 1 + sum(group.kv_cache_spec.max_num_blocks_per_req(vllm_config, capacity) for group in groups)

    def _prefill(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        mm_features: list[MultiModalFeatureSpec],
        scheduled_encoder_inputs: list[int],
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Run the fixed Batch-1 prefill step."""
        num_prompt_tokens = len(prompt_token_ids)
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id=self._SLOT_ID,
                    prompt_token_ids=prompt_token_ids,
                    mm_features=mm_features,
                    sampling_params=sampling_params,
                    pooling_params=None,
                    block_ids=self._block_ids,
                    num_computed_tokens=0,
                    lora_request=None,
                )
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={self._SLOT_ID: num_prompt_tokens},
            total_num_scheduled_tokens=num_prompt_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs=({self._SLOT_ID: scheduled_encoder_inputs} if scheduled_encoder_inputs else {}),
            num_common_prefix_blocks=[0] * len(self._block_ids),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=self._new_block_ids,
        )
        runner = self._require_runner()
        runner.execute_model(scheduler_output)
        return runner.sample_tokens(None)

    def _decode(
        self,
        num_prompt_tokens: int,
        num_output_tokens: int,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Run one fixed Batch-1 decode step."""
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[self._SLOT_ID],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[num_prompt_tokens + num_output_tokens - 1],
                num_output_tokens=[num_output_tokens],
            ),
            num_scheduled_tokens={self._SLOT_ID: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self._block_ids),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        runner = self._require_runner()
        runner.execute_model(scheduler_output)
        return runner.sample_tokens(None)

    @staticmethod
    def _extract_sampled_token(output: ModelRunnerOutput | AsyncModelRunnerOutput) -> int:
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()
        return int(output.sampled_token_ids[0][0])

    def _reset(self, *, has_visual_input: bool) -> None:
        runner = self._require_runner()
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids = {self._SLOT_ID}
        with set_current_vllm_config(self._vllm_config):
            runner.execute_model(cleanup)
        if has_visual_input:
            # GPUModelRunner uses encoder_cache as the transient hand-off from
            # the vision encoder to the backbone. Do not retain it across calls.
            runner.reset_encoder_cache()


__all__ = [
    "ARModelRunner",
    "GenerationResult",
]
