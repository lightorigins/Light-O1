from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOTS = (
    REPOSITORY / "light_deploy",
    REPOSITORY / "server",
    REPOSITORY / "webui/src",
    REPOSITORY / "examples/sonic",
)


def _public_source_files() -> list[Path]:
    files = []
    for root in PUBLIC_SOURCE_ROOTS:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".js", ".html", ".css", ".toml"}
        )
    return files


def test_public_tree_contains_only_runtime_product_roots() -> None:
    assert not (REPOSITORY / "tools").exists()
    assert not (REPOSITORY / "robot_gateway").exists()
    assert not (REPOSITORY / "assets").exists()
    assert {path.name for path in (REPOSITORY / "light_deploy").iterdir() if path.is_dir()} <= {
        "__pycache__",
        "action_tokenizer",
    }


def test_public_runtime_has_one_action_representation() -> None:
    import light_deploy.action_contract as contract

    assert not hasattr(contract, "ACTION_REPRESENTATIONS")


def test_public_compact_package_has_only_runtime_modules() -> None:
    package = REPOSITORY / "light_deploy/action_tokenizer"
    assert {path.name for path in package.iterdir() if path.name != "__pycache__"} == {
        "__init__.py",
        "bundle.py",
        "decoder.py",
        "fk.py",
        "network.py",
        "representation.py",
    }


def test_public_source_contains_no_machine_specific_paths() -> None:
    violations = []
    for path in _public_source_files():
        source = path.read_text(encoding="utf-8")
        for marker in ("/mnt/data/", "/home/", "/root/"):
            if marker in source:
                violations.append(f"{path.relative_to(REPOSITORY)}: {marker}")
    assert not violations, "\n".join(violations)
