"""Versioned compact-action HTTP request contract shared by API and Control Server."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from light_deploy.sampling import (
    DEFAULT_ACTION_TEMPERATURE,
    DEFAULT_ACTION_TOP_P,
    DEFAULT_REASONING_TEMPERATURE,
    DEFAULT_REASONING_TOKEN_BUDGET,
    DEFAULT_REASONING_TOP_P,
    MAX_ACTION_SECONDS,
    MIN_ACTION_SECONDS,
    action_token_limits,
)

MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_194_304
MAX_IMAGE_B64_CHARS = 4 * ((MAX_IMAGE_BYTES + 2) // 3)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    prompt: str = Field(min_length=1, max_length=4000)
    images: list[str] = Field(default_factory=list, max_length=4)
    enable_thinking: bool = True
    reasoning_token_budget: int = Field(default=DEFAULT_REASONING_TOKEN_BUDGET, ge=0, le=4096)
    reasoning_temperature: float = Field(default=DEFAULT_REASONING_TEMPERATURE, ge=0, le=2)
    reasoning_top_p: float = Field(default=DEFAULT_REASONING_TOP_P, gt=0, le=1)
    action_min_seconds: float = Field(default=MIN_ACTION_SECONDS, ge=MIN_ACTION_SECONDS, le=MAX_ACTION_SECONDS)
    action_max_seconds: float = Field(default=MAX_ACTION_SECONDS, ge=MIN_ACTION_SECONDS, le=MAX_ACTION_SECONDS)
    action_temperature: float = Field(default=DEFAULT_ACTION_TEMPERATURE, ge=0, le=2)
    action_top_p: float = Field(default=DEFAULT_ACTION_TOP_P, gt=0, le=1)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value.strip()

    @field_validator("images")
    @classmethod
    def validate_image_sizes(cls, values: list[str]) -> list[str]:
        if any(len(value) > MAX_IMAGE_B64_CHARS for value in values):
            raise ValueError("each image must be at most 6 MiB")
        return values

    @model_validator(mode="after")
    def validate_duration(self) -> "GenerateRequest":
        action_token_limits(self.action_min_seconds, self.action_max_seconds)
        return self
