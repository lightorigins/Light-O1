import hashlib
import json
import math
import string
from pathlib import Path

from safetensors import safe_open

from light_deploy.errors import DecoderBundleError

from .representation import FEATURE_DIM, FPS, REPRESENTATION_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in string.hexdigits for character in value):
        raise DecoderBundleError(f"invalid {field}")
    return value.lower()


def _require_equal(*, field: str, actual: object, expected: object) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise DecoderBundleError(f"unsupported compact decoder {field}: expected {expected!r}, got {actual!r}")


def _require_field_set(*, value: object, expected: set[str], field: str) -> dict:
    if not isinstance(value, dict):
        raise DecoderBundleError(f"{field} must be an object")
    if set(value) != expected:
        raise DecoderBundleError(f"{field} field set mismatch: {sorted(set(value) ^ expected)}")
    return value


def _require_int(*, field: str, value: object, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise DecoderBundleError(f"compact decoder {field} must be an integer >= {minimum}, got {value!r}")
    return value


def validate_compact_decoder_config(config: object) -> dict:
    config = _require_field_set(
        value=config,
        expected={
            "schema_version",
            "architecture",
            "representation",
            "fps",
            "feature_size",
            "projected_feature_size",
            "codebook_size",
            "levels",
            "decoder",
        },
        field="compact decoder config",
    )
    _require_equal(field="schema_version", actual=config["schema_version"], expected=1)
    _require_equal(
        field="architecture",
        actual=config["architecture"],
        expected="causal_conv1d_fsq_action_decoder",
    )
    _require_equal(field="representation", actual=config["representation"], expected=REPRESENTATION_NAME)
    _require_equal(field="fps", actual=config["fps"], expected=FPS)
    _require_equal(field="feature_size", actual=config["feature_size"], expected=FEATURE_DIM)
    projected_feature_size = FEATURE_DIM + 1
    _require_equal(
        field="projected_feature_size", actual=config["projected_feature_size"], expected=projected_feature_size
    )

    decoder = _require_field_set(
        value=config["decoder"],
        expected={
            "feature_size",
            "hidden_dim",
            "temporal_kernel_size",
            "num_stages",
            "channel_blocks_per_stage",
            "dilations",
            "expansion_ratio",
            "num_norm_groups",
            "norm_type",
            "causal",
            "use_sandwich_block",
            "proj_out_kernel_size",
            "z_channels",
        },
        field="compact decoder",
    )
    _require_equal(field="decoder feature_size", actual=decoder["feature_size"], expected=projected_feature_size)
    _require_int(field="hidden_dim", value=decoder["hidden_dim"], minimum=1)
    _require_int(field="temporal_kernel_size", value=decoder["temporal_kernel_size"], minimum=1)
    num_stages = _require_int(field="num_stages", value=decoder["num_stages"], minimum=1)
    _require_int(field="channel_blocks_per_stage", value=decoder["channel_blocks_per_stage"], minimum=0)
    dilations = decoder["dilations"]
    if (
        not isinstance(dilations, list)
        or len(dilations) != num_stages
        or any(type(dilation) is not int or dilation < 1 for dilation in dilations)
    ):
        raise DecoderBundleError(f"compact decoder dilations must contain {num_stages} positive integers")
    _require_int(field="expansion_ratio", value=decoder["expansion_ratio"], minimum=1)
    _require_int(field="num_norm_groups", value=decoder["num_norm_groups"], minimum=1)
    _require_equal(field="decoder norm_type", actual=decoder["norm_type"], expected="weight_norm")
    _require_equal(field="decoder causal", actual=decoder["causal"], expected=True)
    _require_equal(field="decoder use_sandwich_block", actual=decoder["use_sandwich_block"], expected=False)
    _require_int(field="proj_out_kernel_size", value=decoder["proj_out_kernel_size"], minimum=1)

    levels = config["levels"]
    if not isinstance(levels, list) or not levels or any(type(level) is not int or level < 2 for level in levels):
        raise DecoderBundleError("compact decoder levels must be a non-empty list of integers >= 2")
    _require_equal(field="codebook_size", actual=config["codebook_size"], expected=math.prod(levels))
    z_channels = _require_int(field="decoder z_channels", value=decoder["z_channels"], minimum=1)
    _require_equal(field="decoder z_channels", actual=z_channels, expected=len(levels))
    return config


def validate_compact_decoder_bundle(bundle_path: str | Path) -> Path:
    bundle = Path(bundle_path).expanduser().resolve()
    if not bundle.is_dir():
        raise DecoderBundleError(f"compact decoder bundle is not a directory: {bundle}")
    entries = list(bundle.iterdir())
    if any(path.is_symlink() for path in entries):
        raise DecoderBundleError("compact decoder bundle must not contain symlinks")
    required_names = {"decoder_config.json", "action_decoder.safetensors", "manifest.json"}
    actual_names = {path.name for path in entries}
    if actual_names != required_names:
        raise DecoderBundleError(f"compact decoder bundle entry set mismatch: {sorted(actual_names ^ required_names)}")
    if any(not path.is_file() for path in entries):
        raise DecoderBundleError("compact decoder bundle entries must be regular files")

    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DecoderBundleError("invalid compact decoder manifest JSON") from error
    manifest = _require_field_set(
        value=manifest,
        expected={"schema_version", "artifact_type", "representation", "files"},
        field="manifest top",
    )
    _require_equal(field="manifest schema_version", actual=manifest["schema_version"], expected=1)
    _require_equal(field="manifest artifact_type", actual=manifest["artifact_type"], expected="action_tokenizer")
    _require_equal(field="manifest representation", actual=manifest["representation"], expected=REPRESENTATION_NAME)
    expected_files = {"decoder_config.json", "action_decoder.safetensors"}
    files = _require_field_set(
        value=manifest["files"],
        expected=expected_files,
        field="manifest files",
    )
    for name in sorted(expected_files):
        expected = _validate_sha256(files[name], field=f"manifest files {name}")
        actual = _sha256(bundle / name)
        if actual != expected:
            raise DecoderBundleError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")

    try:
        with safe_open(bundle / "action_decoder.safetensors", framework="pt", device="cpu") as tensors:
            metadata = tensors.metadata()
    except Exception as error:
        raise DecoderBundleError("invalid compact decoder safetensors metadata") from error
    if metadata not in (None, {}):
        raise DecoderBundleError("compact decoder safetensors metadata must be empty")

    try:
        config = json.loads((bundle / "decoder_config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DecoderBundleError("invalid compact decoder config JSON") from error
    validate_compact_decoder_config(config)
    return bundle
