"""Resident GPU HTTP API; independent of the browser Control Server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import logging
import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from light_deploy import __version__
from light_deploy.action_contract import ACTION_REPRESENTATION_NAME, validate_action_output
from light_deploy.cuda_compat import build_worker_environment
from light_deploy.errors import ContextLengthError
from light_deploy.generate import Inference
from light_deploy.protocol import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, GenerateRequest
from light_deploy.sampling import (
    ACTION_FPS,
    action_token_limits,
    generation_token_limit,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model: Path
    device: str = "cuda:0"
    host: str = "127.0.0.1"
    port: int = 8030
    max_model_len: int = 1024
    kv_cache_memory_bytes: int = 2 << 30
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not self.model.is_absolute():
            raise ValueError("model must be an absolute path")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.max_model_len <= 0 or self.kv_cache_memory_bytes <= 0:
            raise ValueError("model length and KV cache size must be positive")


def _decode_images(values: list[str]) -> list[Image.Image]:
    images = []
    for value in values:
        try:
            raw = base64.b64decode(value, validate=True)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds size limit")
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG"} or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("unsupported image format or dimensions")
                images.append(image.convert("RGB"))
        except (binascii.Error, OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError("images must be base64 PNG/JPEG, at most 6 MiB and 4 megapixels each") from error
    return images


def create_app(config: InferenceConfig, *, inference_factory: Callable[..., Any] = Inference) -> FastAPI:
    generation_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference") as executor:
            application.state.executor = executor
            runtime = await asyncio.wrap_future(
                executor.submit(
                    inference_factory,
                    config.model,
                    device=config.device,
                    max_model_len=config.max_model_len,
                    kv_cache_memory_bytes=config.kv_cache_memory_bytes,
                    deterministic=config.deterministic,
                )
            )
            application.state.runtime = runtime
            try:
                yield
            finally:
                application.state.runtime = None

                def close_runtime() -> None:
                    with generation_lock:
                        executor.submit(runtime.close).result()

                await asyncio.to_thread(close_runtime)

    application = FastAPI(title="Light Deploy Inference API", version=__version__, lifespan=lifespan)
    application.state.runtime = None

    @application.get("/health")
    def health() -> JSONResponse:
        ready = application.state.runtime is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "model": config.model.name,
                "representation": ACTION_REPRESENTATION_NAME,
                "action_dim": 138,
                "fps": ACTION_FPS,
                "capabilities": {"thinking": True, "vision": True},
            },
        )

    @application.post("/api/generate")
    def generate(request: GenerateRequest) -> JSONResponse:
        if not generation_lock.acquire(blocking=False):
            return JSONResponse(status_code=409, content={"error": {"code": "busy", "message": "Model is busy"}})
        try:
            runtime = application.state.runtime
            if runtime is None:
                return JSONResponse(
                    status_code=503, content={"error": {"code": "not_ready", "message": "Model is not loaded"}}
                )
            try:
                images = _decode_images(request.images)
            except ValueError as error:
                return JSONResponse(
                    status_code=400, content={"error": {"code": "invalid_image", "message": str(error)}}
                )
            minimum, maximum = action_token_limits(request.action_min_seconds, request.action_max_seconds)
            max_new_tokens = generation_token_limit(request.enable_thinking, request.reasoning_token_budget, maximum)
            if max_new_tokens >= config.max_model_len:
                raise ContextLengthError(max_new_tokens + 1, config.max_model_len)
            kwargs = request.model_dump(exclude={"prompt", "images", "action_min_seconds", "action_max_seconds"})
            output = application.state.executor.submit(
                runtime.generate,
                request.prompt,
                images=images,
                min_action_tokens=minimum,
                max_action_tokens=maximum,
                max_new_tokens=max_new_tokens,
                **kwargs,
            ).result()
            action = validate_action_output(output.action, representation=output.representation)
            return JSONResponse(
                content={
                    "schema_version": 1,
                    "representation": ACTION_REPRESENTATION_NAME,
                    "action": action.tolist(),
                    "num_frames": len(action),
                    "fps": ACTION_FPS,
                    "reasoning": output.reasoning,
                    "finish_reason": output.finish_reason,
                    "raw_token_count": output.raw_token_count,
                    "action_token_count": output.action_token_count,
                    "action_token_ids": output.action_token_ids,
                    "action_codebook_ids": output.action_codebook_ids,
                }
            )
        except ContextLengthError:
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "context_length_exceeded",
                        "message": (
                            f"Request exceeds context length {config.max_model_len}; reduce the prompt, images or "
                            "generation budget, or restart inference with a larger context."
                        ),
                    }
                },
            )
        except Exception:
            LOGGER.exception("Action generation failed")
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "generation_failed", "message": "Generation failed; check inference logs"}},
            )
        finally:
            generation_lock.release()

    return application


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 << 30)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    config = InferenceConfig(**{**vars(args), "model": args.model.expanduser().resolve()})
    if config.device.startswith("cuda") and os.environ.get("_LIGHT_DEPLOY_API_BOOTSTRAPPED") != "1":
        environment = build_worker_environment()
        environment["_LIGHT_DEPLOY_API_BOOTSTRAPPED"] = "1"
        os.execve(sys.executable, [sys.executable, "-m", "light_deploy.api", *sys.argv[1:]], environment)
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
