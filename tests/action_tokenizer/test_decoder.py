import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from light_deploy.action_tokenizer.bundle import validate_compact_decoder_bundle
from light_deploy.action_tokenizer.decoder import ActionDecoder
from light_deploy.action_tokenizer.network import ActionDecoderNetwork
from light_deploy.errors import DecoderBundleError


def _compact_config() -> dict:
    return {
        "schema_version": 1,
        "architecture": "causal_conv1d_fsq_action_decoder",
        "representation": "human_action_138_v1",
        "fps": 20,
        "feature_size": 138,
        "projected_feature_size": 139,
        "codebook_size": 16,
        "levels": [4, 4],
        "decoder": {
            "feature_size": 139,
            "hidden_dim": 8,
            "temporal_kernel_size": 3,
            "num_stages": 2,
            "channel_blocks_per_stage": 1,
            "dilations": [1, 2],
            "expansion_ratio": 2,
            "num_norm_groups": 1,
            "norm_type": "weight_norm",
            "causal": True,
            "use_sandwich_block": False,
            "proj_out_kernel_size": 3,
            "z_channels": 2,
        },
    }


def _action_decoder() -> tuple[ActionDecoderNetwork, ActionDecoder, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    config = _compact_config()
    network = ActionDecoderNetwork(config["decoder"])
    mean = torch.linspace(-0.8, 0.8, config["projected_feature_size"])
    scale = torch.linspace(0.5, 1.5, config["projected_feature_size"])
    projected_decoder = ActionDecoder(
        config=config,
        network=network,
        mean=mean,
        scale=scale,
        device=torch.device("cpu"),
    )
    return network, projected_decoder, mean, scale


def _compact_from_projected(projected: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            projected[..., :3],
            torch.atan2(projected[..., 4:5], projected[..., 3:4]),
            projected[..., 5:137],
            projected[..., 137:139].clamp(0, 1),
        ),
        dim=-1,
    )


def _write_bundle(
    path: Path,
    config: dict,
    network: ActionDecoderNetwork,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    path.mkdir()
    config_path = path / "decoder_config.json"
    weights_path = path / "action_decoder.safetensors"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tensors = {f"decoder.{key}": value.detach().contiguous() for key, value in network.state_dict().items()}
    tensors["normalization.mean"] = mean
    tensors["normalization.scale"] = scale
    save_file(tensors, weights_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "action_tokenizer",
        "representation": "human_action_138_v1",
        "files": {
            "decoder_config.json": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "action_decoder.safetensors": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _refresh_weights_hash(path: Path) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = (path / "action_decoder.safetensors").read_bytes()
    manifest["files"]["action_decoder.safetensors"] = hashlib.sha256(weights).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _replace_bundle_tensor(bundle: Path, tensor_name: str, value: torch.Tensor) -> None:
    weights_path = bundle / "action_decoder.safetensors"
    tensors = load_file(weights_path)
    tensors[tensor_name] = value
    save_file(tensors, weights_path)
    _refresh_weights_hash(bundle)


@pytest.mark.parametrize("tokens", [torch.tensor([0, 5, 15]), torch.tensor([[0, 5], [7, 15]])])
def test_compact_decoder_converts_denormalized_projection(tokens: torch.Tensor) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    compact_decoder = decoder_module.ActionDecoder(
        config=_compact_config(),
        network=network,
        mean=mean,
        scale=scale,
        device=torch.device("cpu"),
    )

    batched = tokens.unsqueeze(0) if tokens.ndim == 1 else tokens
    digits = torch.stack((batched % 4, batched // 4), dim=-1)
    latents = ((digits - 2) / 2).permute(0, 2, 1).unsqueeze(-1)
    with torch.inference_mode():
        projected = network(latents).squeeze(1).float() * scale + mean
    if tokens.ndim == 1:
        projected = projected.squeeze(0)
    compact = compact_decoder.decode(tokens)

    torch.testing.assert_close(compact, _compact_from_projected(projected), rtol=0, atol=0)
    assert compact.dtype == torch.float32
    assert compact.shape == (*tokens.shape, 138)


def test_action_decoder_has_no_redundant_subclass() -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    assert not hasattr(decoder_module, "HumanActionDecoder")


@pytest.mark.parametrize(
    ("tokens", "error", "message"),
    [
        (torch.tensor([1.0]), TypeError, "integer dtype"),
        (torch.zeros((1, 1, 1), dtype=torch.long), ValueError, "shape"),
        (torch.tensor([], dtype=torch.long), ValueError, "must not be empty"),
        (torch.tensor([16]), ValueError, "must be in"),
    ],
)
def test_compact_decoder_preserves_token_validation(
    tokens: torch.Tensor, error: type[Exception], message: str
) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    decoder = decoder_module.ActionDecoder(
        config=_compact_config(), network=network, mean=mean, scale=scale, device=torch.device("cpu")
    )

    with pytest.raises(error, match=message):
        decoder.decode(tokens)


def test_load_compact_decoder_rejects_legacy_architecture(tmp_path: Path) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    config = _compact_config()
    config["architecture"] = "causal_conv1d_fsq_motion_decoder"
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, config, network, mean, scale)

    with pytest.raises(DecoderBundleError, match="compact decoder architecture"):
        decoder_module.load_action_decoder(bundle)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 2, "schema_version"),
        (("representation",), "other_action", "representation"),
        (("fps",), 30, "fps"),
        (("feature_size",), 137, "feature_size"),
        (("projected_feature_size",), 138, "projected_feature_size"),
        (("decoder", "feature_size"), 138, "decoder feature_size"),
    ],
)
def test_load_compact_decoder_rejects_invalid_contract(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    config = _compact_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, config, network, mean, scale)

    with pytest.raises(DecoderBundleError, match=message):
        decoder_module.load_action_decoder(bundle)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_load_compact_decoder_requires_exact_decoder_field_set(tmp_path: Path, mutation: str) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    config = _compact_config()
    if mutation == "missing":
        del config["decoder"]["hidden_dim"]
    else:
        config["decoder"]["source_feature_size"] = 274
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, config, network, mean, scale)

    with pytest.raises(DecoderBundleError, match="decoder field set"):
        decoder_module.load_action_decoder(bundle)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("decoder", "hidden_dim"), True, "hidden_dim"),
        (("decoder", "temporal_kernel_size"), 0, "temporal_kernel_size"),
        (("decoder", "num_stages"), 3, "dilations"),
        (("decoder", "channel_blocks_per_stage"), -1, "channel_blocks_per_stage"),
        (("decoder", "dilations"), [1, 0], "dilations"),
        (("decoder", "expansion_ratio"), 0, "expansion_ratio"),
        (("decoder", "num_norm_groups"), 0, "num_norm_groups"),
        (("decoder", "proj_out_kernel_size"), 0, "proj_out_kernel_size"),
        (("levels",), [1, 4], "levels"),
    ],
)
def test_load_compact_decoder_rejects_invalid_decoder_ranges(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    config = _compact_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, config, network, mean, scale)

    with pytest.raises(DecoderBundleError, match=message):
        decoder_module.load_action_decoder(bundle)


def test_compact_bundle_rejects_nested_private_asset_directory(tmp_path: Path) -> None:
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    private = bundle / "private"
    private.mkdir()
    (private / "weights.pkl").write_bytes(b"private")

    with pytest.raises(DecoderBundleError, match="entry set mismatch.*private"):
        validate_compact_decoder_bundle(bundle)


@pytest.mark.parametrize(
    ("location", "key", "value"),
    [
        ("top", "internal_mapping", [0, 1, 9]),
        ("top", "source", {"checkpoint_path": "/private/checkpoint"}),
        ("files", "private.bin", "4" * 64),
    ],
)
def test_compact_bundle_rejects_extra_manifest_fields(tmp_path: Path, location: str, key: str, value: object) -> None:
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if location == "top":
        manifest[key] = value
    else:
        manifest[location][key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DecoderBundleError, match=f"manifest {location} field set"):
        validate_compact_decoder_bundle(bundle)


def test_compact_bundle_rejects_symlink(tmp_path: Path) -> None:
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    (bundle / "alias").symlink_to(bundle / "decoder_config.json")

    with pytest.raises(DecoderBundleError, match="symlink"):
        validate_compact_decoder_bundle(bundle)


@pytest.mark.parametrize(
    ("tensor_name", "value"),
    [
        ("normalization.mean", float("nan")),
        ("normalization.scale", float("inf")),
        ("decoder.proj_out.conv.bias", float("nan")),
        ("decoder.proj_in.parametrizations.weight.original1", float("inf")),
    ],
)
def test_load_compact_decoder_rejects_non_finite_tensor(tmp_path: Path, tensor_name: str, value: float) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    weights_path = bundle / "action_decoder.safetensors"
    tensors = load_file(weights_path)
    tensors[tensor_name] = tensors[tensor_name].clone()
    tensors[tensor_name].view(-1)[0] = value
    save_file(tensors, weights_path)
    _refresh_weights_hash(bundle)

    with pytest.raises(DecoderBundleError, match=f"non-finite.*{tensor_name}"):
        decoder_module.load_action_decoder(bundle)


def test_compact_decoder_yaw_uses_denormalized_heading_and_preserves_branch_cut() -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    heading_xz = torch.tensor(
        [
            [2.0, 3.0],
            [-1.0, 1e-7],
            [-1.0, -1e-7],
            [0.0, 0.0],
        ]
    )
    mean = torch.zeros(139)
    scale = torch.ones(139)
    mean[3:5] = torch.tensor([0.5, -1.0])
    scale[3:5] = torch.tensor([2.0, 4.0])
    physical = torch.zeros((4, 139))
    physical[:, 3:5] = heading_xz
    normalized = (physical - mean) / scale

    class FixedProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            values = normalized[: latents.shape[2]].unsqueeze(0).unsqueeze(1)
            return values.expand(latents.shape[0], -1, -1, -1)

    decoder = decoder_module.ActionDecoder(
        config=_compact_config(),
        network=FixedProjection(),
        mean=mean,
        scale=scale,
        device=torch.device("cpu"),
    )

    action = decoder.decode(torch.tensor([0, 1, 2, 3]))

    torch.testing.assert_close(action[:, 3], torch.atan2(heading_xz[:, 1], heading_xz[:, 0]))
    assert action[1, 3] > 0
    assert action[2, 3] < 0
    assert action[3, 3].item() == 0.0


def test_load_assigns_state_into_meta_network_before_single_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    observed: dict[str, object] = {}
    network_class = decoder_module.ActionDecoderNetwork

    class InspectingNetwork(network_class):
        def __init__(self, config: dict) -> None:
            super().__init__(config)
            observed["constructor_devices"] = {parameter.device.type for parameter in self.parameters()}

        def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
            observed["load_devices"] = {parameter.device.type for parameter in self.parameters()}
            observed["assign"] = assign
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        def to(self, *args, **kwargs):
            observed["to_calls"] = int(observed.get("to_calls", 0)) + 1
            observed["pre_to_devices"] = {parameter.device.type for parameter in self.parameters()}
            return super().to(*args, **kwargs)

    monkeypatch.setattr(decoder_module, "ActionDecoderNetwork", InspectingNetwork)

    decoder = decoder_module.load_action_decoder(bundle, dtype=torch.float64)

    assert observed == {
        "constructor_devices": {"meta"},
        "load_devices": {"meta"},
        "assign": True,
        "to_calls": 1,
        "pre_to_devices": {"cpu"},
    }
    assert {parameter.dtype for parameter in decoder.network.parameters()} == {torch.float64}
    assert decoder.decode([0, 1]).dtype == torch.float32


def test_invalid_state_shape_fails_before_meta_network_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    key = "decoder.proj_out.conv.parametrizations.weight.original1"
    _replace_bundle_tensor(bundle, key, torch.tensor(0.0))
    observed: dict[str, object] = {}
    network_class = decoder_module.ActionDecoderNetwork

    class InspectingNetwork(network_class):
        def __init__(self, config: dict) -> None:
            super().__init__(config)
            observed["constructor_devices"] = {parameter.device.type for parameter in self.parameters()}

        def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
            observed["load_called"] = True
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        def to(self, *args, **kwargs):
            observed["to_called"] = True
            return super().to(*args, **kwargs)

    monkeypatch.setattr(decoder_module, "ActionDecoderNetwork", InspectingNetwork)

    with pytest.raises(DecoderBundleError, match=f"shape.*{key}|{key}.*shape"):
        decoder_module.load_action_decoder(bundle)

    assert observed == {"constructor_devices": {"meta"}}


@pytest.mark.parametrize(
    ("tensor_name", "dtype", "message"),
    [
        ("normalization.mean", torch.float64, "normalization.mean.*float32"),
        ("decoder.proj_out.conv.bias", torch.int32, "supported floating dtype"),
    ],
)
def test_load_compact_decoder_rejects_invalid_tensor_dtype(
    tmp_path: Path, tensor_name: str, dtype: torch.dtype, message: str
) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    tensors = load_file(bundle / "action_decoder.safetensors")
    _replace_bundle_tensor(bundle, tensor_name, tensors[tensor_name].to(dtype))

    with pytest.raises(DecoderBundleError, match=message):
        decoder_module.load_action_decoder(bundle)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_load_compact_decoder_rejects_non_positive_normalization_scale(tmp_path: Path, value: float) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    scale = scale.clone()
    scale[0] = value
    _replace_bundle_tensor(bundle, "normalization.scale", scale)

    with pytest.raises(DecoderBundleError, match="normalization.scale must be strictly positive"):
        decoder_module.load_action_decoder(bundle)


def test_load_compact_decoder_rejects_zero_norm_weight_direction(tmp_path: Path) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    key = "decoder.proj_out.conv.parametrizations.weight.original1"
    tensors = load_file(bundle / "action_decoder.safetensors")
    direction = tensors[key].clone()
    direction[0].zero_()
    _replace_bundle_tensor(bundle, key, direction)

    with pytest.raises(DecoderBundleError, match=f"zero-norm.*{key}"):
        decoder_module.load_action_decoder(bundle)


def test_load_rejects_weight_direction_that_underflows_in_runtime_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)
    key = "decoder.proj_out.conv.parametrizations.weight.original1"
    tensors = load_file(bundle / "action_decoder.safetensors")
    direction = tensors[key].clone()
    direction[0].fill_(1e-8)
    _replace_bundle_tensor(bundle, key, direction)

    decoder = decoder_module.load_action_decoder(bundle, dtype=torch.float32)

    assert {parameter.dtype for parameter in decoder.network.parameters()} == {torch.float32}
    observed: dict[str, object] = {}
    network_class = decoder_module.ActionDecoderNetwork

    class InspectingNetwork(network_class):
        def __init__(self, config: dict) -> None:
            super().__init__(config)
            observed["constructor_devices"] = {parameter.device.type for parameter in self.parameters()}

        def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
            observed["load_called"] = True
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        def to(self, *args, **kwargs):
            observed["to_called"] = True
            return super().to(*args, **kwargs)

    monkeypatch.setattr(decoder_module, "ActionDecoderNetwork", InspectingNetwork)

    with pytest.raises(DecoderBundleError, match=f"zero-norm.*target dtype.*{key}"):
        decoder_module.load_action_decoder(bundle, dtype=torch.float16)

    assert observed == {"constructor_devices": {"meta"}}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_load_supports_normal_low_precision_runtime_dtypes(tmp_path: Path, dtype: torch.dtype) -> None:
    decoder_module = importlib.import_module("light_deploy.action_tokenizer.decoder")
    network, _, mean, scale = _action_decoder()
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, _compact_config(), network, mean, scale)

    decoder = decoder_module.load_action_decoder(bundle, dtype=dtype)

    assert {parameter.dtype for parameter in decoder.network.parameters()} == {dtype}


def test_runtime_decoder_does_not_contain_legacy_source_indices() -> None:
    decoder_path = Path(__file__).resolve().parents[2] / "light_deploy/action_tokenizer/decoder.py"

    assert "274" not in decoder_path.read_text(encoding="utf-8")
