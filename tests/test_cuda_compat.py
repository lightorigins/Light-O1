from pathlib import Path

import pytest

from light_deploy.cuda_compat import resolve_cuda_library_path


def test_resolve_cuda_library_path_adds_compat_for_old_driver(tmp_path: Path) -> None:
    purelib = tmp_path / "site-packages"
    cuda_lib = purelib / "nvidia" / "cu13" / "lib"
    cuda_lib.mkdir(parents=True)
    compat = tmp_path / "compat"
    compat.mkdir()
    (compat / "libcuda.so.1").write_bytes(b"")

    result = resolve_cuda_library_path(
        purelib=purelib,
        compat_dir=compat,
        driver_api=12020,
        required_driver_api=13000,
        existing="/custom/lib",
    )

    assert result.split(":") == [str(compat), str(cuda_lib), "/custom/lib"]


def test_resolve_cuda_library_path_skips_compat_for_current_driver(tmp_path: Path) -> None:
    purelib = tmp_path / "site-packages"
    cuda_lib = purelib / "nvidia" / "runtime" / "lib"
    cuda_lib.mkdir(parents=True)

    result = resolve_cuda_library_path(
        purelib=purelib,
        compat_dir=tmp_path / "missing",
        driver_api=13000,
        required_driver_api=13000,
        existing="",
    )

    assert result == str(cuda_lib)


def test_resolve_cuda_library_path_sorts_venv_libraries(tmp_path: Path) -> None:
    purelib = tmp_path / "site-packages"
    second = purelib / "nvidia" / "zeta" / "lib"
    first = purelib / "nvidia" / "alpha" / "lib"
    second.mkdir(parents=True)
    first.mkdir(parents=True)

    result = resolve_cuda_library_path(
        purelib=purelib,
        compat_dir=tmp_path / "missing",
        driver_api=13000,
        required_driver_api=13000,
        existing="",
    )

    assert result.split(":") == [str(first), str(second)]


def test_resolve_cuda_library_path_requires_compat_for_old_driver(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CUDA compatibility library"):
        resolve_cuda_library_path(
            purelib=tmp_path / "site-packages",
            compat_dir=tmp_path / "missing",
            driver_api=12020,
            required_driver_api=13000,
            existing="",
        )
