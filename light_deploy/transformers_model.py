"""Qwen3.5 Action model on vanilla transformers, inference only.

The transformers backend's model class. The training-framework base class is
swapped for upstream ``transformers.Qwen3_5ForConditionalGeneration`` and the
training-only surface is dropped (losses, fused kernels, parallel plans, log-prob
path). Weight names match the checkpoint exactly: ``motion_embeddings.weight``
and ``motion_head.weight`` on top of the stock Qwen3.5 parameters (the trained
action vocabulary retains its native ``motion_*`` tensor names and token ids).

:class:`~light_deploy.transformers_runner.TransformersRunner` loads a checkpoint
through this class directly, so the checkpoint needs no bundled modeling code
(no ``auto_map`` / ``trust_remote_code``).

Requires transformers>=5.8 (first release with ``qwen3_5``).
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
)


class Qwen3_5ActionConfig(Qwen3_5Config):
    """Qwen3.5 config + discrete-action-token fields (split-embeddings layout)."""

    model_type = "qwen3_5"

    def __init__(
        self,
        split_embeddings: bool = False,
        num_motion_tokens: int = 0,
        motion_head_align: int = 1,
        motion_embedding_dim: Optional[int] = None,
        motion_projection_layers: int = 0,
        motion_start_id: Optional[int] = None,
        motion_pad_id: Optional[int] = None,
        motion_end_id: Optional[int] = None,
        motion_id_bias: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_embeddings = split_embeddings
        self.num_motion_tokens = num_motion_tokens
        self.motion_head_align = motion_head_align
        # Hidden dim for the action tower lives on the text tower in VL-style models.
        text_hidden_size = self.text_config.hidden_size
        self.motion_embedding_dim = text_hidden_size if motion_embedding_dim is None else motion_embedding_dim
        self.motion_projection_layers = motion_projection_layers
        self.motion_start_id = motion_start_id
        self.motion_pad_id = motion_pad_id
        self.motion_end_id = motion_end_id
        self.motion_id_bias = motion_id_bias


@dataclass
class Qwen3_5CausalLMOutputWithAction(Qwen3_5CausalLMOutputWithPast):
    motion_logits: Optional[torch.FloatTensor] = None


class Qwen3_5ActionForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 with a separate action vocabulary: dedicated embedding table and
    logit head for action token ids ``>= config.motion_id_bias``."""

    config_class = Qwen3_5ActionConfig

    def __init__(self, config):
        super().__init__(config)

        if not getattr(config, "split_embeddings", False):
            raise ValueError("Qwen3_5ActionForConditionalGeneration requires split_embeddings=True.")
        if getattr(config, "num_motion_tokens", 0) <= 0:
            raise ValueError("Qwen3_5ActionForConditionalGeneration requires num_motion_tokens > 0.")

        text_hidden_size = config.text_config.hidden_size
        motion_embedding_dim = text_hidden_size if config.motion_embedding_dim is None else config.motion_embedding_dim
        motion_projection_layers = min(getattr(config, "motion_projection_layers", 0), 2)
        if motion_projection_layers == 0 and motion_embedding_dim != text_hidden_size:
            raise ValueError("motion_embedding_dim must equal text hidden size when motion_projection_layers is 0")

        self.motion_embeddings = nn.Embedding(config.num_motion_tokens, motion_embedding_dim)
        if motion_projection_layers >= 2:
            self.motion_embeddings_projection = nn.Sequential(
                nn.Linear(motion_embedding_dim, text_hidden_size),
                nn.GELU(),
                nn.Linear(text_hidden_size, text_hidden_size),
            )
        elif motion_projection_layers == 1:
            self.motion_embeddings_projection = nn.Linear(motion_embedding_dim, text_hidden_size)
        else:
            self.motion_embeddings_projection = nn.Identity()
        # Checkpoint stores the head as a (num_motion_tokens, hidden) matrix — a
        # bias-free Linear named ``motion_head`` matches ``motion_head.weight``.
        self.motion_head = nn.Linear(text_hidden_size, config.num_motion_tokens, bias=False)

    def _prepare_motion_inputs(self, input_ids: Optional[torch.LongTensor]):
        if input_ids is None:
            raise ValueError("input_ids is required when split_embeddings=True.")
        motion_id_mask = input_ids >= self.config.motion_id_bias
        motion_input_ids = (input_ids - self.config.motion_id_bias).clamp(min=0)
        text_input_ids = input_ids.masked_fill(motion_id_mask, 0)
        motion_token_indexes = motion_id_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
        return text_input_ids, motion_input_ids, motion_token_indexes

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        motion_only: bool = False,
        **kwargs,
    ) -> Qwen3_5CausalLMOutputWithAction:
        """``motion_only=True`` skips the text lm_head projection — during
        action decoding only ``motion_logits`` is consumed, and the 248k-row
        lm_head matmul is a significant share of per-step memory traffic."""
        if inputs_embeds is None:
            text_input_ids, motion_input_ids, motion_token_indexes = self._prepare_motion_inputs(input_ids)
            inputs_embeds = self.get_input_embeddings()(text_input_ids)
            motion_embeds = self.motion_embeddings(motion_input_ids.view(-1)[motion_token_indexes])
            motion_embeds = self.motion_embeddings_projection(motion_embeds)
            inputs_embeds.view(-1, inputs_embeds.shape[-1])[motion_token_indexes] = motion_embeds.to(
                inputs_embeds.dtype
            )

        # position_ids stays None for text+action sequences: with no image/video
        # grids the base model's compute_3d_position_ids falls back to standard
        # sequential positions, which is exactly what a token stream needs.

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        return Qwen3_5CausalLMOutputWithAction(
            logits=None if motion_only else self.lm_head(hidden_states),
            motion_logits=self.motion_head(hidden_states),
            past_key_values=outputs.past_key_values,
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )


__all__ = ["Qwen3_5ActionConfig", "Qwen3_5ActionForConditionalGeneration", "Qwen3_5CausalLMOutputWithAction"]
