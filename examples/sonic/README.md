# G1 Sonic simulation

This optional example tracks the public humanoid skeleton with a real closed-loop
Sonic policy in MuJoCo. It never sends commands to physical hardware. The human
preview remains the original compact-action FK trajectory; the G1 result is a
retargeted policy rollout and is not guaranteed to reproduce it exactly or remain upright.

Install the `sonic` extra, and install `ffmpeg` on the simulation host. Supply a
matching **low-latency** GEAR-SONIC ONNX pair named `model_encoder.onnx` and
`model_decoder.onnx`; policy weights are not distributed here. The required input
shapes are `(1, 1247)` and `(1, 994)`, with a 64-value latent and 29 joint actions.
Use the original policy provider's terms for those weights.

The WebUI creates a plan from the selected generation and launches this example.
It can also be called directly with a prepared plan:

```bash
python -m examples.sonic --plan /tmp/plan.npz --checkpoint /path/to/sonic/low_latency \
  --out /tmp/result.json --mp4 /tmp/result.mp4
```

`adapter.py` maps public FK into pelvis-relative, X-forward/Z-up reference joints,
adding two fixed wrist endpoints for the policy's 24-joint input. It uses neither
a human mesh nor a fitted body model. `reference.py` resamples to a global 50 Hz
timeline and holds the last available reference at the tail. Four future samples
are encoded against the simulated robot's **current** root orientation, with the
initial headings aligned once at startup.

The wrist adapter is geometric: each forearm's outward axis becomes wrist +X and
human +Y up becomes wrist +Z. The transformed local rotations are decomposed into
G1 intrinsic XYZ roll/pitch/yaw and clipped to the model's wrist limits. This is a
simple embodiment-specific approximation, not a calibrated retargeter. The two
hand-opening channels do not affect this example.

The controller uses 50 Hz policy updates and four 5 ms torque-controlled physics
steps per update. A virtual support releases after initialization; it is absent
during the recorded action. Falling below 0.35 m pelvis height ends the rollout
and is reported as `fell_at`, not hidden as a successful tracking result. Videos
are always rendered at 640×480, 25 FPS; durations are quantized to control/video ticks.
Existing outputs are never overwritten.

Validated simulation environment: MuJoCo 3.11.0, ONNX Runtime 1.28.0, NumPy 2.5.1,
Pillow 12.3.0. Simulation dependencies are imported only by the simulation process;
the GPU model runner does not import them. The G1 assets' upstream BSD-3-Clause
notice and pinned source are in `assets/g1/`.

```bash
SONIC_TEST_PYTHON=/path/to/simulation/python \
SONIC_TEST_CHECKPOINT=/path/to/sonic/low_latency \
python -m pytest -n 1 tests/test_sonic_reference.py tests/test_sonic_simulation.py
```
