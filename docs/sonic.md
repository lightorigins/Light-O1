# Sonic simulation (optional)

Sonic renders a generated action as closed-loop MuJoCo physics on a Unitree G1, producing a 25 FPS video.
It uses the same public skeleton as the rest of the runtime, with a G1-specific adapter.

```bash
uv sync --extra inference --extra sonic
uv run --extra inference --extra sonic light-deploy-server \
  --model-path /absolute/path/to/Light-O1-Preview \
  --sonic-checkpoint /absolute/path/to/sonic/low_latency
```

## Requirements

- **FFmpeg** on the host.
- A matching GEAR-SONIC **low-latency** ONNX encoder/decoder pair. **Policy weights are not bundled.**
- `--sonic-python /path/to/python` if the simulation needs a separate environment.

Startup checks the dependencies and the policy input shapes.

> [!NOTE]
> There is no real-hardware command path. Sonic generates video from simulation only.

A completed rollout may still fall over — inspect the reported metrics rather than assuming success.

See [the Sonic example](../examples/sonic/README.md) for its contract and limits.
