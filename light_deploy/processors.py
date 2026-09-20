"""Qwen3.5 Motion prompt and special-token processing.

The processors hold only the Motion protocol (special-token ids, chat templating,
and output parsing), so they take the model's ``hf_config`` rather than a
``VllmConfig``. The vLLM image path additionally needs a ``VllmConfig`` to build a
``VllmInputProcessor``; text-only callers (the transformers backend) omit it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase


class MMMotionInputProcessor:
    """Build text or visual prompts for autonomous or direct Motion generation."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Any,
        *,
        vllm_config: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.vllm_config = vllm_config
        self.motion_start_id = int(config.motion_start_id)
        self.motion_end_id = int(config.motion_end_id)
        self.motion_id_bias = int(config.motion_id_bias)
        self.thinking_start_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
        self.im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.generation_stop_token_ids = {self.motion_end_id, self.im_end_id}
        self.motion_stop_token_ids = {self.motion_end_id}
        self._vllm_input_processor: Any | None = None

    def prepare_text_generation(self, prompt: str, *, enable_thinking: bool = False) -> tuple[list[int], set[int]]:
        """Build text-only prompt token ids and stop ids without vLLM.

        Applies the same thinking/``motion_start`` hand-off rule as
        :meth:`prepare_generation`, so the transformers and vLLM backends open the
        assistant turn identically for text prompts.
        """
        prompt_token_ids = self._build_text_prompt(prompt, enable_thinking=enable_thinking)
        return self._apply_handoff(prompt_token_ids, enable_thinking=enable_thinking)

    def prepare_generation(
        self,
        request_id: str,
        prompt: str,
        images: Sequence[Any],
        sampling_params: Any,
        *,
        enable_thinking: bool = False,
    ):
        """Build prompt token ids, sampling params, image features, and stop ids (vLLM)."""
        prompt_token_ids, sampling_params, mm_features = self._prepare_prompt(
            request_id,
            prompt,
            images,
            sampling_params,
            enable_thinking=enable_thinking,
        )
        prompt_token_ids, stop_token_ids = self._apply_handoff(prompt_token_ids, enable_thinking=enable_thinking)
        return prompt_token_ids, sampling_params, mm_features, stop_token_ids

    def _apply_handoff(self, prompt_token_ids: list[int], *, enable_thinking: bool) -> tuple[list[int], set[int]]:
        if enable_thinking:
            if prompt_token_ids[-len(self.thinking_start_ids) :] != self.thinking_start_ids:
                prompt_token_ids.extend(self.thinking_start_ids)
            stop_token_ids = self.generation_stop_token_ids
        else:
            if prompt_token_ids[-len(self.thinking_start_ids) :] == self.thinking_start_ids:
                del prompt_token_ids[-len(self.thinking_start_ids) :]
            if not prompt_token_ids or prompt_token_ids[-1] != self.motion_start_id:
                prompt_token_ids.append(self.motion_start_id)
            stop_token_ids = self.motion_stop_token_ids
        return prompt_token_ids, stop_token_ids

    def close(self) -> None:
        if self._vllm_input_processor is not None:
            self._vllm_input_processor.renderer.shutdown()
            self._vllm_input_processor = None

    def _build_text_prompt(self, prompt: str, *, enable_thinking: bool) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=enable_thinking,
        )

    def _prepare_prompt(
        self,
        request_id: str,
        prompt: str,
        images: Sequence[Any],
        sampling_params: Any,
        *,
        enable_thinking: bool,
    ):
        if not images:
            return self._build_text_prompt(prompt, enable_thinking=enable_thinking), sampling_params, []

        image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        # Match the training sequence exactly: user text first, then images.
        # Reversing them makes native-thinking checkpoints behave like VQA and
        # may emit im_end instead of handing off to motion_start.
        rendered_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + image_placeholder * len(images)}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        prompt_input = {
            "prompt": rendered_prompt,
            "multi_modal_data": {"image": list(images)},
        }
        input_processor = self._get_vllm_input_processor()
        inputs = input_processor.process_inputs(
            request_id,
            input_processor.renderer.render_cmpl([prompt_input])[0],
            sampling_params,
            supported_tasks=("generate",),
        )
        return inputs.prompt_token_ids, inputs.sampling_params, inputs.mm_features

    def _get_vllm_input_processor(self):
        if self.vllm_config is None:
            raise RuntimeError("visual input requires a VllmConfig; construct MMMotionInputProcessor with vllm_config")
        from vllm.config import set_current_vllm_config
        from vllm.v1.engine.input_processor import InputProcessor as VllmInputProcessor

        if self._vllm_input_processor is None:
            with set_current_vllm_config(self.vllm_config):
                self._vllm_input_processor = VllmInputProcessor(self.vllm_config)
        return self._vllm_input_processor


@dataclass(slots=True)
class GenerationResult:
    thinking_text: str | None
    motion_token_ids: list[int]
    raw_token_ids: list[int]
    finish_reason: str


class MMMotionOutputProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: Any) -> None:
        self.tokenizer = tokenizer
        self.motion_start_id = int(config.motion_start_id)
        self.motion_end_id = int(config.motion_end_id)
        self.im_end_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    def parse(self, raw_token_ids: list[int], *, enable_thinking: bool) -> GenerationResult:
        motion_started = not enable_thinking
        thinking_text = None
        motion_token_ids = list(raw_token_ids)

        if enable_thinking:
            try:
                motion_start = raw_token_ids.index(self.motion_start_id)
            except ValueError:
                thinking_token_ids = raw_token_ids
                motion_token_ids = []
            else:
                motion_started = True
                thinking_token_ids = raw_token_ids[:motion_start]
                motion_token_ids = raw_token_ids[motion_start + 1 :]
            thinking_text = self._decode_thinking(thinking_token_ids)

        if motion_token_ids and motion_token_ids[-1] == self.motion_end_id:
            finish_reason = "motion_end"
        elif motion_started:
            finish_reason = "max_tokens_during_motion"
        elif raw_token_ids and raw_token_ids[-1] == self.im_end_id:
            finish_reason = "im_end_before_motion"
        else:
            finish_reason = "max_tokens_before_motion"

        return GenerationResult(
            thinking_text=thinking_text,
            motion_token_ids=motion_token_ids,
            raw_token_ids=raw_token_ids,
            finish_reason=finish_reason,
        )

    def _decode_thinking(self, token_ids: list[int]) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.split("</think>", 1)[0].strip()


__all__ = ["GenerationResult", "MMMotionInputProcessor", "MMMotionOutputProcessor"]
