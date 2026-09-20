from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from light_deploy.action_contract import ACTION_REPRESENTATION_NAME, validate_action_output


@dataclass(slots=True)
class ActionOutput:
    action: np.ndarray
    representation: str
    reasoning: str | None
    finish_reason: str
    raw_token_count: int
    action_token_count: int
    action_token_ids: list[int] = field(default_factory=list)
    action_codebook_ids: list[int] = field(default_factory=list)


def generate_action(
    runner,
    decoder,
    prompt: str,
    *,
    images: Sequence[Image.Image] = (),
    enable_thinking: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 0,
    reasoning_token_budget: int | None = None,
    reasoning_temperature: float | None = None,
    reasoning_top_p: float | None = None,
    min_action_tokens: int = 0,
    max_action_tokens: int | None = None,
    action_temperature: float | None = None,
    action_top_p: float | None = None,
) -> ActionOutput:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    # The trained AR protocol retains its native motion_* names and token IDs.
    result = runner.generate(
        prompt,
        images=images,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        seed=seed,
        reasoning_token_budget=reasoning_token_budget,
        reasoning_temperature=reasoning_temperature,
        reasoning_top_p=reasoning_top_p,
        min_motion_tokens=min_action_tokens,
        max_motion_tokens=max_action_tokens,
        motion_temperature=action_temperature,
        motion_top_p=action_top_p,
    )
    action_end_id = runner.input_processor.motion_end_id
    action_id_bias = runner.input_processor.motion_id_bias
    global_ids = [token_id for token_id in result.motion_token_ids if token_id != action_end_id]
    local_ids = [token_id - action_id_bias for token_id in global_ids]
    if not local_ids:
        raise RuntimeError(f"generation produced no Action tokens: {result.finish_reason}")
    decoded = decoder.decode(np.asarray(local_ids, dtype=np.int64))
    action = decoded.detach().cpu().numpy() if hasattr(decoded, "detach") else decoded
    action = validate_action_output(action, representation=ACTION_REPRESENTATION_NAME)
    return ActionOutput(
        action=action,
        representation=ACTION_REPRESENTATION_NAME,
        reasoning=result.thinking_text,
        finish_reason={
            "motion_end": "action_end",
            "max_tokens_during_motion": "max_tokens_during_action",
        }.get(result.finish_reason, result.finish_reason),
        raw_token_count=len(result.raw_token_ids),
        action_token_count=len(local_ids),
        action_token_ids=global_ids,
        action_codebook_ids=local_ids,
    )


class Inference:
    representation = ACTION_REPRESENTATION_NAME

    def __init__(
        self,
        model_path: str | Path,
        decoder_path: str | Path | None = None,
        *,
        device: str = "cuda:0",
        max_model_len: int = 320,
        kv_cache_memory_bytes: int = 1 << 30,
        deterministic: bool = False,
        backend: Literal["vllm", "transformers"] = "vllm",
    ) -> None:
        """Load one AR backend and the action decoder.

        ``backend="vllm"`` is the production resident-GPU runtime. ``backend=
        "transformers"`` is a torch-native, text-only alternative that needs
        neither vLLM nor a cu130 torch (``kv_cache_memory_bytes`` and
        ``deterministic`` are ignored).
        """
        from light_deploy.action_tokenizer.decoder import load_action_decoder

        if backend == "vllm":
            from light_deploy.runner import ARModelRunner

            self.runner = ARModelRunner.from_pretrained(
                model_path,
                device=device,
                max_model_len=max_model_len,
                kv_cache_memory_bytes=kv_cache_memory_bytes,
                deterministic=deterministic,
            )
        elif backend == "transformers":
            from light_deploy.transformers_runner import TransformersRunner

            self.runner = TransformersRunner.from_pretrained(
                model_path,
                device=device,
                max_model_len=max_model_len,
            )
        else:
            raise ValueError(f"unknown backend: {backend!r}")
        try:
            bundle_path = Path(model_path) / "action_tokenizer" if decoder_path is None else decoder_path
            self.decoder = load_action_decoder(bundle_path, device=device)
        except BaseException:
            self.runner.close()
            raise

    def generate(self, prompt: str, **kwargs) -> ActionOutput:
        return generate_action(self.runner, self.decoder, prompt, **kwargs)

    def close(self) -> None:
        self.runner.close()

    def __enter__(self) -> Inference:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _load_images(paths: Sequence[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact humanoid action with Qwen3.5 Action AR")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, help="Optional override for the decoder packaged with the model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("human_action.npy"))
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Inference(args.model, args.decoder, device=args.device) as inference:
        output = inference.generate(
            args.prompt,
            images=_load_images(args.image),
            enable_thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
    np.save(args.output, output.action, allow_pickle=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "representation": output.representation,
                "shape": list(output.action.shape),
                "reasoning": output.reasoning,
                "finish_reason": output.finish_reason,
                "raw_token_count": output.raw_token_count,
                "action_token_count": output.action_token_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
