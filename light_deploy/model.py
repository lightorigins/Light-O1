from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ActionForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 with split action embeddings, projection and output head.

    Motion modules are attached as top-level attributes so they survive Qwen3-VL's
    ``hf_to_vllm_mapper`` (which only rewrites ``model.visual.`` / ``lm_head.`` /
    ``model.language_model.`` prefixes) unchanged, matching the HF checkpoint keys
    ``motion_embeddings.*`` / ``motion_embeddings_projection.*`` / ``motion_head.*``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        self.motion_id_bias = config.motion_id_bias
        self.text_vocab_size = text_config.vocab_size
        motion_projection_layers = config.motion_projection_layers

        if get_pp_group().is_last_rank:
            self.initialize_motion_modules(
                config=config,
                hidden_size=text_config.hidden_size,
                motion_projection_layers=motion_projection_layers,
                quant_config=vllm_config.quant_config,
                prefix=prefix,
            )
        else:
            self.motion_embeddings = None
            self.motion_embeddings_projection = None
            self.motion_head = None
            self.motion_projection_layers = 0

    def initialize_motion_modules(
        self,
        *,
        config,
        hidden_size: int,
        motion_projection_layers: int,
        quant_config,
        prefix: str,
    ) -> None:
        num_motion_tokens = config.num_motion_tokens
        motion_embed_dim = (
            hidden_size
            if motion_projection_layers == 0 or config.motion_embedding_dim is None
            else config.motion_embedding_dim
        )
        self.motion_embeddings = VocabParallelEmbedding(
            num_motion_tokens,
            motion_embed_dim,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "motion_embeddings"),
        )
        self.motion_projection_layers = motion_projection_layers
        if motion_projection_layers >= 2:
            self.motion_embeddings_projection = nn.ModuleDict(
                {
                    "0": ColumnParallelLinear(
                        motion_embed_dim,
                        hidden_size,
                        bias=True,
                        gather_output=False,
                        quant_config=quant_config,
                        prefix=maybe_prefix(prefix, "motion_embeddings_projection.0"),
                    ),
                    "2": RowParallelLinear(
                        hidden_size,
                        hidden_size,
                        bias=True,
                        input_is_parallel=True,
                        reduce_results=True,
                        quant_config=quant_config,
                        prefix=maybe_prefix(prefix, "motion_embeddings_projection.2"),
                    ),
                }
            )
        elif motion_projection_layers == 1:
            self.motion_embeddings_projection = ColumnParallelLinear(
                motion_embed_dim,
                hidden_size,
                bias=True,
                gather_output=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "motion_embeddings_projection"),
            )
        else:
            self.motion_embeddings_projection = None
        self.motion_head = ColumnParallelLinear(
            hidden_size,
            num_motion_tokens,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "motion_head"),
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.motion_embeddings is None:
            return super().embed_input_ids(
                input_ids,
                multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        # Clamp each lookup independently to prevent OOB reads from CUDA graph
        # padding tokens (garbage residual values in the buffer).
        motion_id_mask = input_ids >= self.motion_id_bias
        safe_ids = torch.where(motion_id_mask, torch.zeros_like(input_ids), input_ids)
        safe_ids = safe_ids.clamp(0, self.text_vocab_size - 1)
        # Route through the parent so vision embeddings are spliced at `is_multimodal`
        # positions; vision placeholders are text-vocab ids (< motion_id_bias), so they
        # never collide with the motion positions overwritten below.
        base = super().embed_input_ids(
            safe_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        motion_indices = (input_ids - self.motion_id_bias).clamp(
            min=0,
            max=self.motion_embeddings.num_embeddings - 1,
        )
        motion_emb = self.motion_embeddings(motion_indices)
        if self.motion_projection_layers >= 2:
            motion_emb, _ = self.motion_embeddings_projection["0"](motion_emb)
            motion_emb = torch.nn.functional.gelu(motion_emb)
            motion_emb, _ = self.motion_embeddings_projection["2"](motion_emb)
        elif self.motion_projection_layers == 1:
            motion_emb, _ = self.motion_embeddings_projection(motion_emb)
        return torch.where(motion_id_mask.unsqueeze(-1), motion_emb, base)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # slow/fast deployment checkpoints carry fast-path DiT weights
        # (action_expert.*); the Block-AR slow path must skip, not reject, them
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp.", "action_expert."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        language_model = self.language_model
        logits = language_model.logits_processor(language_model.lm_head, hidden_states)
        if logits is None or self.motion_head is None:
            return logits

        motion_logits, _ = self.motion_head(hidden_states)
        expanded = logits.new_full(
            (logits.shape[0], self.motion_id_bias + motion_logits.shape[-1]),
            float("-inf"),
        )
        expanded[:, : logits.shape[-1]] = logits
        expanded[:, self.motion_id_bias :] = motion_logits
        return expanded


def register_model() -> None:
    from vllm import ModelRegistry
    from vllm.config.model import ModelConfig

    ModelRegistry.register_model(
        "Qwen3_5ActionForConditionalGeneration",
        "light_deploy.model:Qwen3_5ActionForConditionalGeneration",
    )
    if getattr(ModelConfig, "motion_vocab_patched", False):
        return
    original_get_vocab_size = ModelConfig.get_vocab_size

    def get_vocab_size(config):
        vocab_size = original_get_vocab_size(config)
        motion_end_id = getattr(config.hf_config, "motion_end_id", vocab_size - 1)
        return max(vocab_size, motion_end_id + 1)

    ModelConfig.get_vocab_size = get_vocab_size
    ModelConfig.motion_vocab_patched = True
