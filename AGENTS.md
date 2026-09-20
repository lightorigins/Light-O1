# AGENTS.md

Guidance for AI coding agents helping someone **use** Light-O1 — installing it, running inference, and
working with what it returns.

Light-O1 turns a natural-language instruction into a sequence of whole-body humanoid actions, together with
the reasoning trace it produced on the way. This repository is the inference runtime for it.

## 1. First, work out what the user's machine can actually run

Do this **before** suggesting an install. The wrong branch here wastes a multi-gigabyte download.

```bash
uname -s -m          # want: Linux x86_64
python3 --version    # want: 3.11.x  (the project pins >=3.11,<3.12)
nvidia-smi           # driver version and CUDA version, if any
```

| What you find | What to do |
| :--- | :--- |
| Linux x86-64, NVIDIA GPU, CUDA 13 | Default path. Use the **vLLM** backend. |
| Linux x86-64, NVIDIA GPU, CUDA < 13 | Use the **transformers** backend: `backend="transformers"`. It needs neither vLLM nor a cu130 torch. One request at a time. |
| No NVIDIA GPU, or macOS, or Windows | **Do not install.** Point the user at the [hosted playground](https://huggingface.co/spaces/LightOriginsHQ/Light-O1-Preview-playground). |

Do not try to force the vLLM backend onto an older CUDA. Torch is pinned to a cu130 wheel in
`pyproject.toml`; it will not resolve against an older driver, and no amount of flag-tweaking changes that.

> [!NOTE]
> The `inference` extra installs vLLM **and** cu130 torch even if the user only intends to use the
> transformers backend — there is currently no lighter extra. The transformers backend does not *import*
> vLLM at runtime (the import is lazy, inside the backend branch), but it is still installed. On a machine
> where the cu130 wheel will not resolve, `uv sync --extra inference` fails outright; install
> `transformers[torch]` and `safetensors` against the user's own working torch instead, and run this package
> from a source checkout.

## 2. Install

```bash
git clone https://github.com/lightorigins/Light-O1.git
cd Light-O1
uv sync --extra inference
```

Requires `uv >= 0.9.8`. Use plain `uv sync` if the machine will only run the control server and talk to a GPU
elsewhere.

## 3. Get the weights

Download [LightOriginsHQ/Light-O1-Preview](https://huggingface.co/LightOriginsHQ/Light-O1-Preview). The
action decoder ships **inside** the checkpoint under `action_tokenizer/` — do not go looking for a separate
decoder download, and do not pass `--decoder` unless deliberately overriding it.

The checkpoint's `architectures` field must be `Qwen3_5ActionForConditionalGeneration`. If a user has
retagged or converted a checkpoint and it no longer says that, loading fails — that is the first thing to
check.

## 4. Generate

Command line, for a quick check that everything works:

```bash
uv run --extra inference light-deploy \
  --model /absolute/path/to/Light-O1-Preview \
  --prompt "a person waves with the right hand" \
  --thinking --output human_action.npy
```

From Python, which is what you want inside an application:

```python
from light_deploy.generate import Inference

inference = Inference("/absolute/path/to/Light-O1-Preview", device="cuda:0")
try:
    out = inference.generate(
        "a person raises the right arm and waves",
        enable_thinking=True,
        temperature=0.6,   # defaults shown
        top_p=0.95,
        seed=0,
    )
finally:
    inference.close()
```

`generate()` also accepts `max_new_tokens`, `min_action_tokens`, and separate `reasoning_temperature` /
`reasoning_top_p` / `reasoning_token_budget` if the reasoning and the action should be sampled differently.

Loading the model is expensive. Construct `Inference` **once** and reuse it; do not build one per request.

## 5. Understand what comes back

`generate()` returns an `ActionOutput`:

| Field | Meaning |
| :--- | :--- |
| `action` | `(frames, 138)` float32, 20 FPS |
| `reasoning` | the Thinking text, or `None` if thinking was off |
| `finish_reason` | why generation stopped — check this before trusting a short result |
| `raw_token_count`, `action_token_count` | token accounting |
| `action_token_ids`, `action_codebook_ids` | the underlying token ids |

The 138 columns are **not** world-space joint positions. Decode them rather than slicing by hand:

```python
from light_deploy.action_tokenizer import unpack_human_action, validate_human_action

fields = unpack_human_action(validate_human_action(out.action))
fields.root_delta_xz_local       # (T, 2)   per-frame root translation, in the *local* frame
fields.pelvis_height_y           # (T, 1)
fields.yaw_delta_rad_per_frame   # (T, 1)   per-frame yaw *delta*
fields.joint_rot6d_local         # (T, 22, 6) parent-local 6D rotations
fields.left_hand_open            # (T, 1)
fields.right_hand_open           # (T, 1)
```

Root translation and yaw are **per-frame deltas in the local frame**, so summing columns naively gives a
wrong trajectory. To get world-space positions, run forward kinematics:

```python
from light_deploy.action_tokenizer.fk import forward_kinematics

result = forward_kinematics(out.action)   # integrates root motion, applies the skeleton
```

Useful constants live in `light_deploy.action_tokenizer`: `FEATURE_DIM` (138), `FPS` (20), `NUM_JOINTS` (22),
`JOINT_NAMES`, `JOINT_PARENTS`, `AXIS_UP` (`"Y"`), `COORDINATE_SYSTEM` (`"right_handed"`), `FIELD_SLICES`.
Read those rather than hardcoding numbers — **`AXIS_UP` is Y, not Z**, which is a common source of models
lying on their side in a downstream viewer.

## 6. Serving it over HTTP

`light-deploy-api` is the GPU service, `light-deploy-server` is the control server and web console. See
[docs/deployment.md](docs/deployment.md) and [docs/api.md](docs/api.md).

Two behaviours matter when writing a client:

- **It is single-flight.** Initialization, generation and shutdown share one CUDA-owning thread, and a
  concurrent request returns **HTTP 409**. Serialize requests, or use the control server's asynchronous
  `POST /api/generations` and poll. Do not build a client that fans out parallel requests.
- **Overflowing the context returns HTTP 422**, not a truncated result.

Neither service is authenticated. Do not put either on a public address, and do not suggest doing so as a
quick way to share a demo.

## 7. Limits and failure modes

| Symptom | Cause |
| :--- | :--- |
| `HTTP 409` | A generation is already running. Serialize. |
| `HTTP 422` | Prompt plus generation exceeds the context. Shorten the prompt or lower the duration. |
| Load fails on `architectures` | Checkpoint was converted or retagged. It must say `Qwen3_5ActionForConditionalGeneration`. |
| `uv sync --extra inference` cannot resolve torch | CUDA older than 13. Take the transformers route in §1. |
| Action looks sideways downstream | The representation is **Y-up**, right-handed. |
| Action shorter than expected | Check `finish_reason`. |

Durations are bounded to **2–20 seconds**.

## 8. What this repository does not do

Do not attempt these here, and do not tell a user they are possible with this package:

- **No training or fine-tuning.** Inference only.
- **No image input.** Light-O1-Preview takes a text prompt only. Do not offer visual conditioning, and do
  not reach for the image arguments still present in the runtime — they are not supported by this model.
- **No checkpoint conversion.**
- **No physical-robot command path.** `examples/sonic/` is MuJoCo simulation, producing video and metrics.
  It needs a GEAR-SONIC low-latency policy checkpoint that is not bundled.
- **No retargeting to other embodiments** ships in this repository. The action representation is designed to
  be embodiment-agnostic, but mapping it onto a specific robot is the caller's job — the Sonic example shows
  one such adapter.

If a user needs any of these, say so plainly rather than improvising something that appears to work.
