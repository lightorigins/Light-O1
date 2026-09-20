"""Transformers batch-1 AR backend for Qwen3.5 Motion.

A torch-native alternative to :class:`light_deploy.runner.ARModelRunner` that runs
the same split-head decode FSM without vLLM. It loads the checkpoint through
``transformers`` using light_deploy's own model class
(:mod:`light_deploy.transformers_model`, no ``trust_remote_code``), concatenates the model's separate text and motion heads into the single
union vector the FSM masks, and returns the same
:class:`~light_deploy.processors.GenerationResult` the vLLM runner returns.

The union index equals the global token id: text ids occupy ``[0, motion_id_bias)``
and motion ids occupy ``[motion_id_bias, motion_end_id]``, so a sampled index is
already a global token id and ``light_deploy.generate.generate_action`` maps it to a
local Motion id with the same ``- motion_id_bias`` rule as the vLLM path.

This backend runs on any CUDA build ``transformers`` supports; it does not require
vLLM or a cu130 torch. It is single-request (batch 1) and text-only. Visual input
remains on the vLLM backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from light_deploy.fsm import apply_motion_fsm_mask, in_open_motion_span
from light_deploy.processors import (
    GenerationResult,
    MMMotionInputProcessor,
    MMMotionOutputProcessor,
)

_ARCHITECTURE = "Qwen3_5ActionForConditionalGeneration"


def build_union_logits(text_logits: torch.Tensor, motion_logits: torch.Tensor, *, motion_id_bias: int) -> torch.Tensor:
    """Concatenate the split heads into one vector indexed by global token id.

    ``text_logits`` covers the text vocabulary and ``motion_logits`` covers the
    local Motion ids ``[0, num_motion_tokens)``; the result places text logits at
    ``[0, motion_id_bias)`` and Motion logits at ``[motion_id_bias, ...]`` so a row
    index is the global token id the FSM and output parser expect. Any pad slots
    above ``motion_end_id`` are left for :func:`apply_motion_fsm_mask` to remove.
    """
    text_part = text_logits[:motion_id_bias]
    if text_part.shape[0] < motion_id_bias:
        pad = text_part.new_full((motion_id_bias - text_part.shape[0],), -math.inf)
        text_part = torch.cat([text_part, pad])
    return torch.cat([text_part, motion_logits])


def sample_token(
    row: torch.Tensor, *, temperature: float, top_p: float, top_k: int, generator: torch.Generator
) -> int:
    """Temperature + top-k + nucleus sampling over one masked logits row.

    ``temperature <= 0`` is greedy. The row is assumed already FSM-masked, so
    illegal ids sit at ``-inf`` and never survive sampling.
    """
    if temperature <= 0:
        return int(torch.argmax(row).item())
    logits = row / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        kth_value = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth_value, logits.new_full((), -math.inf), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cumulative = probs.cumsum(dim=-1)
        sorted_logits[(cumulative - probs) >= top_p] = -math.inf
        logits = torch.full_like(logits, -math.inf).scatter(0, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


class TransformersRunner:
    """Run one Qwen3.5 Motion AR model through transformers, no vLLM.

    Exposes the small contract :func:`light_deploy.generate.generate_action` and
    :class:`light_deploy.api` depend on: ``generate`` returning a
    :class:`GenerationResult`, ``input_processor`` exposing ``motion_end_id`` and
    ``motion_id_bias``, and ``close``.
    """

    _ARCHITECTURE = _ARCHITECTURE

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: Any,
        input_processor: MMMotionInputProcessor,
        output_processor: MMMotionOutputProcessor,
        device: torch.device,
        max_model_len: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.device = device
        self.max_model_len = max_model_len
        self.motion_start_id = int(config.motion_start_id)
        self.motion_end_id = int(config.motion_end_id)
        self.motion_id_bias = int(config.motion_id_bias)
        self.split_embeddings = bool(getattr(config, "split_embeddings", False))
        self.motion_id_threshold = self.motion_id_bias if self.split_embeddings else int(config.motion_pad_id)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_model_len: int = 320,
        attn_implementation: str = "sdpa",
    ) -> TransformersRunner:
        """Load the checkpoint and prepare the transformers AR runtime."""
        from transformers import AutoTokenizer

        from light_deploy.transformers_model import Qwen3_5ActionConfig, Qwen3_5ActionForConditionalGeneration

        checkpoint = Path(model_path).expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Checkpoint is missing config.json: {checkpoint}")
        config = Qwen3_5ActionConfig.from_pretrained(checkpoint)
        if cls._ARCHITECTURE not in (getattr(config, "architectures", None) or ()) or not getattr(
            config, "split_embeddings", False
        ):
            raise ValueError("Checkpoint is not a split-embedding Qwen3.5 Action AR model")

        target_device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="right")
        model = Qwen3_5ActionForConditionalGeneration.from_pretrained(
            checkpoint,
            config=config,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        model.to(target_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            input_processor=MMMotionInputProcessor(tokenizer, config),
            output_processor=MMMotionOutputProcessor(tokenizer, config),
            device=target_device,
            max_model_len=max_model_len,
        )

    def run(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = 0,
        on_token: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Generate global Motion token ids for one text prompt (direct Motion)."""
        return self.generate(
            prompt,
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
        images: Sequence[Any] = (),
        enable_thinking: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = 0,
        on_token: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        """Generate one direct-Motion or Thinking-to-Motion result (text-only)."""
        if images:
            raise NotImplementedError(
                "the transformers backend is text-only; use the vllm backend for visual input"
            )
        prompt_token_ids, stop_token_ids = self.input_processor.prepare_text_generation(
            prompt, enable_thinking=enable_thinking
        )
        raw_token_ids = self._decode_tokens(
            prompt_token_ids,
            stop_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            on_token=on_token,
        )
        return self.output_processor.parse(raw_token_ids, enable_thinking=enable_thinking)

    @torch.inference_mode()
    def _decode_tokens(
        self,
        prompt_token_ids: list[int],
        stop_token_ids: set[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        on_token: Callable[[int], None] | None,
    ) -> list[int]:
        """Batch-1 prefill/decode applying the split-head FSM at every step."""
        self._validate_context_length(len(prompt_token_ids) + max_new_tokens)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))

        next_input = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.device)
        past_key_values = None
        raw_token_ids: list[int] = []
        last = prompt_token_ids[-1]
        # Once motion_start is emitted (or pre-fed in direct mode) the rest of the
        # episode is Motion tokens, so the text lm_head is dead weight: run the
        # model motion_only past that point and stand in a cached -inf text row so
        # the same union FSM still applies.
        motion_started = last == self.motion_start_id or last >= self.motion_id_bias
        neg_text = None

        for _ in range(max_new_tokens):
            outputs = self.model(
                next_input,
                past_key_values=past_key_values,
                use_cache=True,
                motion_only=motion_started,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            motion_logits = outputs.motion_logits[:, -1, :].float().squeeze(0)
            if motion_started:
                if neg_text is None:
                    neg_text = motion_logits.new_full((self.motion_id_bias,), -math.inf)
                union = torch.cat([neg_text, motion_logits])
                open_span = False
            else:
                text_logits = outputs.logits[:, -1, :].float().squeeze(0)
                union = build_union_logits(text_logits, motion_logits, motion_id_bias=self.motion_id_bias)
                open_span = last < self.motion_start_id and in_open_motion_span(
                    prompt_token_ids,
                    raw_token_ids,
                    motion_start_id=self.motion_start_id,
                    motion_end_id=self.motion_end_id,
                )
            apply_motion_fsm_mask(
                union,
                last,
                motion_start_id=self.motion_start_id,
                motion_id_bias=self.motion_id_bias,
                motion_end_id=self.motion_end_id,
                motion_id_threshold=self.motion_id_threshold,
                split_embeddings=self.split_embeddings,
                open_motion_span=open_span,
            )
            token_id = sample_token(union, temperature=temperature, top_p=top_p, top_k=top_k, generator=generator)
            raw_token_ids.append(token_id)
            if on_token is not None:
                on_token(token_id)
            last = token_id
            if token_id in stop_token_ids:
                break
            if token_id == self.motion_start_id:
                motion_started = True
            next_input = torch.tensor([[token_id]], dtype=torch.long, device=self.device)

        return raw_token_ids

    def _validate_context_length(self, total_tokens: int) -> None:
        if total_tokens > self.max_model_len:
            raise ValueError(f"Prompt and generation length {total_tokens} exceeds max_model_len {self.max_model_len}")

    def close(self) -> None:
        """Release the model and its GPU memory."""
        model = getattr(self, "model", None)
        self.model = None
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> TransformersRunner:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["TransformersRunner", "build_union_logits", "sample_token"]
