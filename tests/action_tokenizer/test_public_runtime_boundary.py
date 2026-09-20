import ast
import subprocess
import sys
from pathlib import Path

RUNTIME_MODULES = (
    "__init__.py",
    "bundle.py",
    "decoder.py",
    "fk.py",
    "network.py",
    "representation.py",
)
ALLOWED_LIGHT_DEPLOY_IMPORTS = {
    "light_deploy.action_tokenizer",
    "light_deploy.errors",
    "light_deploy.action_contract",
}


def _compact_package() -> Path:
    return Path(__file__).resolve().parents[2] / "light_deploy/action_tokenizer"


def test_compact_runtime_has_only_standalone_runtime_dependencies() -> None:
    package = _compact_package()
    for name in RUNTIME_MODULES:
        source = (package / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        external_light_deploy = {
            module
            for module in imported_modules
            if module.startswith("light_deploy.")
            and not any(module.startswith(root) for root in ALLOWED_LIGHT_DEPLOY_IMPORTS)
        }
        assert not external_light_deploy, f"{name}: {sorted(external_light_deploy)}"


def test_compact_runtime_imports_in_isolated_process() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path.insert(0, {str(repository)!r})\n"
        + """
from light_deploy.action_tokenizer.bundle import validate_compact_decoder_bundle
from light_deploy.action_tokenizer.decoder import ActionDecoder, load_action_decoder
from light_deploy.errors import DecoderBundleError
from light_deploy.action_tokenizer.network import ActionDecoderNetwork
assert validate_compact_decoder_bundle
assert ActionDecoder and load_action_decoder and DecoderBundleError and ActionDecoderNetwork
"""
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
