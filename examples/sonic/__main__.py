import argparse
import importlib
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Track compact humanoid action using G1 Sonic in MuJoCo")
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Directory containing the low-latency ONNX encoder/decoder pair"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--mp4", required=True)
    parser.add_argument("--replan-frames", type=int, default=8)
    parser.add_argument("--lookahead-frames", type=int, default=12)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    simulation = importlib.import_module("examples.sonic.simulation")
    print(json.dumps(simulation.run(args)))


if __name__ == "__main__":
    main()
