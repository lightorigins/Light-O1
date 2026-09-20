from pydantic import BaseModel, ConfigDict, Field


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str | None = None
    device: str | None = None
    max_model_len: int | None = Field(default=None, gt=0)
    kv_cache_memory_bytes: int | None = Field(default=None, gt=0)
    deterministic: bool | None = None


class SonicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replan_frames: int = Field(default=8, ge=1, le=40)
    lookahead_frames: int = Field(default=12, ge=1, le=40)
