"""Split-head decode FSM for Qwen3.5 Motion checkpoints.

The text lm_head and the motion head are trained under separate softmaxes, and
``compute_logits`` concatenates them into one vector the model never learned as a
union. Every decode step therefore has to be masked down to exactly one head's
range: a decode contract, not an optimization.

    text -> motion_start -> motion content -> motion_end -> text

The trajectory layout adds a fifth case. One ``<|motion_start|>`` covers the whole
episode and vision spans sit *between* motion chunks, so after an interior keyframe
the last id is ``<|vision_end|>`` -- a text-range id inside a span that is still
open. A last-token-only rule reads that as "back to text" and masks every motion id
to -inf; since the layout never emits a second ``<|motion_start|>``, the rollout can
never re-enter motion and the next chunk comes back empty. ``open_motion_span`` is
that fifth case.

The pure masking rule (:func:`apply_motion_fsm_mask`, :func:`in_open_motion_span`)
imports no vLLM, so the transformers backend can apply it directly. The vLLM
backend's ``MotionFsmLogitsProcessor`` wraps that rule in vLLM's ``LogitsProcessor``
interface; its vLLM import is guarded so importing this module never requires vLLM
(when vLLM is absent the processor is inert -- only the vLLM runner instantiates it).

This module is self-contained so the transformers backend needs no serving
framework: the pure masking rule lives here, and the guarded vLLM
``MotionFsmLogitsProcessor`` is the only vLLM-specific wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain

import torch

try:
    from vllm.config import VllmConfig
    from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor, MoveDirectionality
except ImportError:  # vLLM is an optional backend; the pure rule below needs none of it.
    VllmConfig = object  # type: ignore[assignment,misc]
    LogitsProcessor = object  # type: ignore[assignment,misc]
    BatchUpdate = MoveDirectionality = None  # type: ignore[assignment]


def apply_motion_fsm_mask(
    row: torch.Tensor,
    last: int,
    *,
    motion_start_id: int,
    motion_id_bias: int,
    motion_end_id: int,
    motion_id_threshold: int,
    split_embeddings: bool,
    open_motion_span: bool,
) -> None:
    """Mask one row of logits in place so only the legal head's range survives."""
    if split_embeddings:
        # in split mode motion_end_id is the largest valid id; anything past it is a
        # GEMM-alignment pad slot (motion_head_align), invalid in every state
        row[motion_end_id + 1 :] = -math.inf
    if last < motion_start_id:
        if open_motion_span:
            # trajectory interior keyframe: continue the next chunk, or end the task
            row[:motion_id_threshold] = -math.inf
        else:
            row[motion_start_id + 1 :] = -math.inf
    elif last == motion_start_id:
        row[:motion_id_bias] = -math.inf
        if split_embeddings:
            row[motion_end_id] = -math.inf
    elif last == motion_end_id:
        row[motion_start_id:] = -math.inf
    elif last >= motion_id_threshold:
        row[:motion_id_threshold] = -math.inf
    else:
        raise ValueError(f"motion FSM: {last} is a pad slot, which no state can produce as a last token")


def in_open_motion_span(prompt_ids, output_ids, *, motion_start_id: int, motion_end_id: int) -> bool:
    """Is the sequence inside an unclosed motion span?

    Walks back to the nearest motion marker. In the trajectory layout the only
    ``motion_start`` sits at the head of the episode, so this walks the whole
    committed stream -- but the caller only asks on the one decode step per robot
    step whose last id is text-range, and a few thousand ids is microseconds.
    Both id spaces are the motion vocabulary: the trainer remaps the text-space
    ``<|motion_end|>`` into it, so the scan never needs the text-space id.
    """
    for token in chain(reversed(output_ids), reversed(prompt_ids)):
        if token == motion_start_id:
            return True
        if token == motion_end_id:
            return False
    return False


def count_motion_tokens(
    prompt_ids: list[int],
    output_ids: list[int],
    *,
    motion_start_id: int,
    motion_end_id: int,
    motion_id_threshold: int,
) -> int:
    count = 0
    for token in chain(reversed(output_ids), reversed(prompt_ids)):
        if token == motion_start_id:
            return count
        if token == motion_end_id:
            return 0
        if token >= motion_id_threshold:
            count += 1
    return 0


@dataclass(slots=True)
class _RequestState:
    prompt_ids: list[int]
    output_ids: list[int]
    reasoning_token_budget: int | None
    min_motion_tokens: int


class MotionFsmLogitsProcessor(LogitsProcessor):
    """vLLM v1 logits processor enforcing the FSM above.

    Only the vLLM backend instantiates this; when vLLM is not installed the base
    class falls back to ``object`` and the processor is inert.

    Holds each request's ``(prompt_ids, output_ids)`` from ``BatchUpdate``;
    ``output_ids`` is a live reference the scheduler appends to, so the last token
    is available as a Python int without a per-step device sync.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        config = vllm_config.model_config.hf_config
        self.motion_start_id = int(config.motion_start_id)
        self.motion_id_bias = int(config.motion_id_bias)
        self.motion_end_id = int(getattr(config, "motion_end_id", self.motion_id_bias - 1))
        self.split_embeddings = bool(getattr(config, "split_embeddings", False))
        # split puts motion content and motion_end above motion_id_bias; non-split
        # puts the pads, motion_end and the content all above motion_pad_id
        self.motion_id_threshold = self.motion_id_bias if self.split_embeddings else int(config.motion_pad_id)
        self.token_refs: dict[int, _RequestState] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not batch_update:
            return
        for index, params, prompt_ids, output_ids in batch_update.added:
            extra_args = params.extra_args or {}
            reasoning_token_budget = extra_args.get("reasoning_token_budget")
            min_motion_tokens = extra_args.get("min_motion_tokens", 0)
            if reasoning_token_budget is not None and reasoning_token_budget < 0:
                raise ValueError("reasoning_token_budget must be non-negative")
            if min_motion_tokens < 0:
                raise ValueError("min_motion_tokens must be non-negative")
            self.token_refs[index] = _RequestState(
                prompt_ids=list(prompt_ids or []),
                output_ids=output_ids,
                reasoning_token_budget=reasoning_token_budget,
                min_motion_tokens=min_motion_tokens,
            )
        for index in batch_update.removed:
            self.token_refs.pop(index, None)
        for source, target, direction in batch_update.moved:
            moved = self.token_refs.pop(source, None)
            displaced = self.token_refs.pop(target, None)
            if moved is not None:
                self.token_refs[target] = moved
            if direction != MoveDirectionality.UNIDIRECTIONAL and displaced is not None:
                self.token_refs[source] = displaced

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index in range(logits.shape[0]):
            state = self.token_refs.get(index)
            # an untracked row means the batch bookkeeping desynced; the text state is
            # the conservative read, since it is the one that never admits motion ids
            prompt_ids = state.prompt_ids if state is not None else []
            output_ids = state.output_ids if state is not None else []
            last = output_ids[-1] if output_ids else (prompt_ids[-1] if prompt_ids else 0)
            if (
                state is not None
                and state.reasoning_token_budget is not None
                and len(output_ids) >= state.reasoning_token_budget
                and not any(
                    token in (self.motion_start_id, self.motion_end_id) for token in chain(prompt_ids, output_ids)
                )
            ):
                motion_start_logit = logits[index, self.motion_start_id].clone()
                logits[index].fill_(-math.inf)
                logits[index, self.motion_start_id] = motion_start_logit
                continue
            apply_motion_fsm_mask(
                logits[index],
                last,
                motion_start_id=self.motion_start_id,
                motion_id_bias=self.motion_id_bias,
                motion_end_id=self.motion_end_id,
                motion_id_threshold=self.motion_id_threshold,
                split_embeddings=self.split_embeddings,
                # only a text-range last can be an interior keyframe; every other
                # state reads off the last token alone
                open_motion_span=last < self.motion_start_id
                and in_open_motion_span(
                    prompt_ids,
                    output_ids,
                    motion_start_id=self.motion_start_id,
                    motion_end_id=self.motion_end_id,
                ),
            )
            if (
                state is not None
                and state.min_motion_tokens
                and in_open_motion_span(
                    prompt_ids,
                    output_ids,
                    motion_start_id=self.motion_start_id,
                    motion_end_id=self.motion_end_id,
                )
                and count_motion_tokens(
                    prompt_ids,
                    output_ids,
                    motion_start_id=self.motion_start_id,
                    motion_end_id=self.motion_end_id,
                    motion_id_threshold=self.motion_id_threshold,
                )
                < state.min_motion_tokens
            ):
                logits[index, self.motion_end_id] = -math.inf
        return logits


__all__ = ["MotionFsmLogitsProcessor", "apply_motion_fsm_mask", "in_open_motion_span"]
