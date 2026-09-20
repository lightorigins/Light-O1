"""Bounded session results; completed action owns immutable float32 bytes."""

import hashlib
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from light_deploy.action_contract import ACTION_REPRESENTATION_NAME, validate_action_output


@dataclass
class Generation:
    generation_id: str
    request: dict[str, Any]
    state: str = "running"
    action: np.ndarray | None = None
    action_sha256: str = ""
    reasoning: str = ""
    error: str | None = None

    def public(self):
        frames = len(self.action) if self.action is not None else 0
        return {
            "generation_id": self.generation_id,
            "state": self.state,
            "representation": ACTION_REPRESENTATION_NAME,
            "num_frames": frames,
            "fps": 20,
            "duration_s": frames / 20,
            "action_sha256": self.action_sha256,
            "prompt": self.request["prompt"],
            "seed": self.request.get("seed", 0),
            "reasoning": self.reasoning,
            "error": {"message": self.error} if self.error else None,
        }


class GenerationStore:
    def __init__(self, max_items=8):
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self.items: OrderedDict[str, Generation] = OrderedDict()

    def create(self, request):
        if len(self.items) >= self.max_items:
            oldest = next((key for key, value in self.items.items() if value.state != "running"), None)
            if oldest is None:
                raise RuntimeError("Generation store busy")
            del self.items[oldest]
        result = Generation(uuid.uuid4().hex, {key: value for key, value in request.items() if key != "images"})
        self.items[result.generation_id] = result
        return result

    def get(self, generation_id):
        return self.items[generation_id]

    def clear(self):
        self.items.clear()

    def complete(self, generation_id, result):
        job = self.get(generation_id)
        if job.state != "running":
            raise RuntimeError("Generation already completed")
        if (result.get("schema_version"), result.get("representation"), result.get("fps")) != (
            1,
            ACTION_REPRESENTATION_NAME,
            20,
        ):
            raise ValueError("Invalid inference response contract")
        raw = result.get("action")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 400 or result.get("num_frames") != len(raw):
            raise ValueError("Invalid inference frame count")
        try:
            action = validate_action_output(
                np.asarray(raw, dtype=np.float32), representation=ACTION_REPRESENTATION_NAME
            )
        except (TypeError, RuntimeError, OverflowError) as error:
            raise ValueError("Invalid inference action") from error
        reasoning = "" if result["reasoning"] is None else result["reasoning"]
        if not isinstance(reasoning, str) or len(reasoning) > 65536:
            raise ValueError("Invalid inference reasoning")
        content = action.tobytes()
        job.action = np.frombuffer(content, dtype=np.float32).reshape(action.shape)
        job.action_sha256 = hashlib.sha256(content).hexdigest()
        job.reasoning = reasoning
        job.state = "done"
