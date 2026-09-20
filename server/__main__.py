"""Launch the Control Server, optionally owning a local GPU API subprocess."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import uvicorn

from server.app import create_app
from server.config import ControlConfig


def main():
    defaults = ControlConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--model-path", type=Path, default=defaults.model_path)
    mode.add_argument("--inference-url", default=defaults.inference_url)
    parser.add_argument("--inference-port", type=int, default=defaults.inference_port)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--max-model-len", type=int, default=defaults.max_model_len)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=defaults.kv_cache_memory_bytes)
    parser.add_argument("--deterministic", action="store_true", default=defaults.deterministic)
    parser.add_argument("--temp-root", type=Path, default=defaults.temp_root)
    parser.add_argument("--webui-dist", type=Path, default=defaults.webui_dist)
    parser.add_argument("--sonic-python", type=Path, default=defaults.sonic_python)
    parser.add_argument("--sonic-checkpoint", type=Path, default=defaults.sonic_checkpoint)
    config = replace(defaults, **vars(parser.parse_args()))
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
