import json
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from light_deploy.errors import DecoderBundleError

from .bundle import validate_compact_decoder_bundle
from .network import ActionDecoderNetwork
from .representation import FEATURE_DIM

_SUPPORTED_DECODER_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_WEIGHT_DIRECTION_SUFFIX = ".parametrizations.weight.original1"


class ActionDecoder:
    def __init__(
        self,
        *,
        config: dict,
        network: ActionDecoderNetwork,
        mean: torch.Tensor,
        scale: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.config = config
        self.network = network
        self.mean = mean.to(device=device, dtype=torch.float32)
        self.scale = scale.to(device=device, dtype=torch.float32)
        self.device = device
        self.levels = torch.tensor(config["levels"], device=device, dtype=torch.long)
        self.basis = torch.cumprod(
            torch.tensor([1, *config["levels"][:-1]], device=device, dtype=torch.long),
            dim=0,
        )

    def decode(self, local_token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        tokens = torch.as_tensor(local_token_ids, device=self.device)
        if tokens.dtype.is_floating_point or tokens.dtype.is_complex:
            raise TypeError("Action token IDs must use an integer dtype")
        if tokens.ndim not in (1, 2):
            raise ValueError(f"Action token IDs must have shape (time,) or (batch,time), got {tuple(tokens.shape)}")
        if tokens.numel() == 0:
            raise ValueError("Action token IDs must not be empty")
        if torch.any(tokens < 0) or torch.any(tokens >= self.config["codebook_size"]):
            raise ValueError(f"Action token IDs must be in [0, {self.config['codebook_size']})")

        remove_batch = tokens.ndim == 1
        if remove_batch:
            tokens = tokens.unsqueeze(0)
        codes = (tokens.unsqueeze(-1) // self.basis) % self.levels
        half_width = self.levels // 2
        latents = ((codes - half_width) / half_width).to(next(self.network.parameters()).dtype)
        latents = latents.permute(0, 2, 1).unsqueeze(-1)
        with torch.inference_mode():
            action = self.network(latents).squeeze(1).float()
            action = action * self.scale + self.mean
            action = torch.cat(
                (
                    action[..., :3],
                    torch.atan2(action[..., 4:5], action[..., 3:4]),
                    action[..., 5:137],
                    action[..., 137:].clamp(0, 1),
                ),
                dim=-1,
            )
        return action.squeeze(0) if remove_batch else action


def _require_supported_float(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.dtype not in _SUPPORTED_DECODER_DTYPES:
        raise DecoderBundleError(f"compact decoder tensor {name} must use a supported floating dtype")


def load_action_decoder(
    bundle_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ActionDecoder:
    requested_path = Path(bundle_path).expanduser()
    if requested_path.suffix == ".pth":
        raise DecoderBundleError("the compact decoder runtime does not load Pickle assets")
    bundle = validate_compact_decoder_bundle(requested_path)
    if dtype not in _SUPPORTED_DECODER_DTYPES:
        raise DecoderBundleError("compact decoder runtime dtype must be a supported floating dtype")
    config = json.loads((bundle / "decoder_config.json").read_text(encoding="utf-8"))
    try:
        tensors = load_file(bundle / "action_decoder.safetensors", device="cpu")
    except SafetensorError as error:
        raise DecoderBundleError(f"invalid compact decoder tensor file: {error}") from error
    expected_normalization_keys = {"normalization.mean", "normalization.scale"}
    missing_normalization = expected_normalization_keys - tensors.keys()
    if missing_normalization:
        raise DecoderBundleError(f"compact decoder bundle is missing tensors: {sorted(missing_normalization)}")
    mean = tensors.pop("normalization.mean")
    scale = tensors.pop("normalization.scale")
    projected_feature_size = FEATURE_DIM + 1
    if mean.dtype != torch.float32:
        raise DecoderBundleError("compact decoder normalization.mean must use float32")
    if scale.dtype != torch.float32:
        raise DecoderBundleError("compact decoder normalization.scale must use float32")
    if tuple(mean.shape) != (projected_feature_size,):
        raise DecoderBundleError(
            f"compact decoder normalization.mean must have shape ({projected_feature_size},), got {tuple(mean.shape)}"
        )
    if tuple(scale.shape) != (projected_feature_size,):
        raise DecoderBundleError(
            "compact decoder normalization.scale must have shape "
            f"({projected_feature_size},), got {tuple(scale.shape)}"
        )
    if not torch.isfinite(mean).all():
        raise DecoderBundleError("compact decoder contains non-finite tensor normalization.mean")
    if not torch.isfinite(scale).all():
        raise DecoderBundleError("compact decoder contains non-finite tensor normalization.scale")
    if torch.any(scale <= 0):
        raise DecoderBundleError("compact decoder normalization.scale must be strictly positive")
    unexpected_namespaces = sorted(key for key in tensors if not key.startswith("decoder."))
    if unexpected_namespaces:
        raise DecoderBundleError(
            f"compact decoder bundle contains unexpected tensor namespaces: {unexpected_namespaces}"
        )
    state = {key.removeprefix("decoder."): value for key, value in tensors.items()}
    for key, value in state.items():
        bundle_key = f"decoder.{key}"
        _require_supported_float(value, name=bundle_key)
        if not torch.isfinite(value).all():
            raise DecoderBundleError(f"compact decoder contains non-finite tensor {bundle_key}")

    target_device = torch.device(device)
    try:
        with torch.device("meta"):
            network = ActionDecoderNetwork(config["decoder"])
    except (RuntimeError, TypeError, ValueError) as error:
        raise DecoderBundleError(f"invalid compact decoder network config: {error}") from error
    expected_state = network.state_dict()
    if set(state) != set(expected_state):
        difference = sorted(set(state) ^ set(expected_state))
        raise DecoderBundleError(f"invalid compact decoder tensor set: {difference}")
    for key, expected in expected_state.items():
        if tuple(state[key].shape) != tuple(expected.shape):
            raise DecoderBundleError(
                f"compact decoder tensor decoder.{key} has shape {tuple(state[key].shape)}, "
                f"expected {tuple(expected.shape)}"
            )
    for key, value in state.items():
        if key.endswith(_WEIGHT_DIRECTION_SUFFIX):
            target_direction = value.to(dtype=dtype)
            has_zero_channel = torch.any(
                torch.count_nonzero(target_direction.reshape(target_direction.shape[0], -1), dim=1) == 0
            )
            del target_direction
            if has_zero_channel:
                bundle_key = f"decoder.{key}"
                raise DecoderBundleError(
                    f"compact decoder contains zero-norm weight direction tensor in target dtype {dtype}: {bundle_key}"
                )
    try:
        network.load_state_dict(state, strict=True, assign=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise DecoderBundleError(f"invalid compact decoder state: {error}") from error
    network.to(device=target_device, dtype=dtype).eval()
    return ActionDecoder(
        config=config,
        network=network,
        mean=mean,
        scale=scale,
        device=target_device,
    )
