from __future__ import annotations

import ctypes
import os
import sys
import sysconfig
from pathlib import Path

CUDA_13_DRIVER_API = 13000
DEFAULT_CUDA_COMPAT_DIR = Path("/usr/local/cuda-13.0/compat")


def resolve_cuda_library_path(
    *,
    purelib: str | Path,
    compat_dir: str | Path,
    driver_api: int,
    required_driver_api: int,
    existing: str,
) -> str:
    purelib = Path(purelib)
    parts = sorted(str(path) for path in (purelib / "nvidia").glob("*/lib") if path.is_dir())
    if driver_api < required_driver_api:
        compat_dir = Path(compat_dir)
        if not (compat_dir / "libcuda.so.1").exists():
            raise RuntimeError(
                f"CUDA compatibility library is required for driver API {driver_api}, "
                f"but {compat_dir / 'libcuda.so.1'} does not exist"
            )
        parts.insert(0, str(compat_dir))
    if existing:
        parts.append(existing)
    return ":".join(parts)


def loaded_driver_api() -> int:
    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return 0
    version = ctypes.c_int()
    if library.cuDriverGetVersion(ctypes.byref(version)) != 0:
        return 0
    return int(version.value)


def build_worker_environment(
    *,
    compat_dir: str | Path | None = None,
    required_driver_api: int = CUDA_13_DRIVER_API,
) -> dict[str, str]:
    environment = dict(os.environ)
    python_bin = str(Path(sys.executable).parent)
    path_parts = [part for part in environment.get("PATH", "").split(os.pathsep) if part and part != python_bin]
    environment["PATH"] = os.pathsep.join((python_bin, *path_parts))
    selected_compat = compat_dir or environment.get("LIGHT_DEPLOY_CUDA_COMPAT_DIR", DEFAULT_CUDA_COMPAT_DIR)
    environment["LD_LIBRARY_PATH"] = resolve_cuda_library_path(
        purelib=sysconfig.get_paths()["purelib"],
        compat_dir=selected_compat,
        driver_api=loaded_driver_api(),
        required_driver_api=required_driver_api,
        existing=environment.get("LD_LIBRARY_PATH", ""),
    )
    return environment
