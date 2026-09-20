import os
import sys
from pathlib import Path

import light_deploy.cuda_compat as cuda_compat
from light_deploy.cuda_compat import build_worker_environment


def test_build_worker_environment_exposes_current_python_tools(tmp_path: Path, monkeypatch) -> None:
    python_path = tmp_path / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    monkeypatch.setattr(sys, "executable", str(python_path))
    monkeypatch.setattr(cuda_compat, "loaded_driver_api", lambda: 13000)
    monkeypatch.setenv("PATH", "/usr/bin")

    environment = build_worker_environment(
        compat_dir=tmp_path / "unused",
        required_driver_api=13000,
    )

    assert environment["PATH"].split(os.pathsep) == [str(python_path.parent), "/usr/bin"]
