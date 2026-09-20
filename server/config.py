"""Control-host configuration; remote mode never resolves a GPU checkpoint path."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class ControlConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    inference_url: str | None = None
    inference_port: int = 8030
    model_path: Path | None = None
    device: str = "cuda:0"
    max_model_len: int = 1024
    kv_cache_memory_bytes: int = 2 << 30
    deterministic: bool = False
    webui_dist: Path = Path(__file__).resolve().with_name("webui")
    temp_root: Path = Path("/tmp")
    sonic_python: Path = Path(sys.executable)
    sonic_checkpoint: Path | None = None

    def __post_init__(self):
        if not 1 <= self.port <= 65535 or not 1 <= self.inference_port <= 65535:
            raise ValueError("ports must be between 1 and 65535")
        if self.port == self.inference_port:
            raise ValueError("control and inference must use different ports")
        if self.max_model_len <= 0 or self.kv_cache_memory_bytes <= 0:
            raise ValueError("model length and KV cache memory must be positive")
        if self.inference_url:
            parsed = urlparse(self.inference_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("inference_url must be an HTTP(S) service address without credentials")
            if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
                raise ValueError("inference_url must not contain a path, query or fragment")
            if self.model_path is not None:
                raise ValueError("choose inference_url or model_path, not both")
        elif self.model_path is not None and not self.model_path.is_absolute():
            raise ValueError("model_path must be absolute")

    @property
    def api_url(self):
        return self.inference_url or f"http://127.0.0.1:{self.inference_port}"

    @classmethod
    def from_env(cls):
        model = os.environ.get("LIGHT_DEPLOY_MODEL_PATH")
        sonic = os.environ.get("LIGHT_DEPLOY_SONIC_CKPT")
        return cls(
            host=os.environ.get("LIGHT_DEPLOY_HOST", "127.0.0.1"),
            port=int(os.environ.get("LIGHT_DEPLOY_PORT", "8090")),
            inference_url=os.environ.get("LIGHT_DEPLOY_INFERENCE_URL"),
            inference_port=int(os.environ.get("LIGHT_DEPLOY_INFERENCE_PORT", "8030")),
            model_path=Path(model).expanduser().absolute() if model else None,
            device=os.environ.get("LIGHT_DEPLOY_DEVICE", "cuda:0"),
            max_model_len=int(os.environ.get("LIGHT_DEPLOY_MAX_MODEL_LEN", "1024")),
            kv_cache_memory_bytes=int(os.environ.get("LIGHT_DEPLOY_KV_CACHE_MEMORY_BYTES", str(2 << 30))),
            webui_dist=Path(os.environ.get("LIGHT_DEPLOY_WEBUI_DIST", str(cls().webui_dist))),
            temp_root=Path(os.environ.get("LIGHT_DEPLOY_TEMP_ROOT", "/tmp")),
            sonic_python=Path(os.environ.get("LIGHT_DEPLOY_SONIC_PYTHON", sys.executable)),
            sonic_checkpoint=Path(sonic).expanduser() if sonic else None,
        )
