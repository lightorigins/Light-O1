<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/LO_lockup_h_white.svg">
    <img src="docs/assets/LO_lockup_h_black.svg" width="55%" alt="Light Origins" />
  </picture>
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://lightorigins.com/"><img alt="Homepage"
    src="https://img.shields.io/badge/Homepage-Light%20Origins-9c403d?style=flat"/></a>
  <a href="https://huggingface.co/LightOriginsHQ/Light-O1-Preview"><img alt="Model"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Light--O1--Preview-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://huggingface.co/spaces/LightOriginsHQ/Light-O1-Preview-playground"><img alt="Playground"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Playground-Try%20it%20now-ff9d00?color=ff9d00&logoColor=white"/></a>
  <br>
  <a href="https://discord.gg/zwZuD9JG"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-Light%20Origins-5865F2?style=flat&logo=discord&logoColor=white"/></a>
  <a href="#7-community"><img alt="WeChat"
    src="https://img.shields.io/badge/WeChat-Light%20Origins-07C160?style=flat&logo=wechat&logoColor=white"/></a>
  <br>
  <a href="LICENSE"><img alt="Code License"
    src="https://img.shields.io/badge/Code_License-Apache_2.0-f5de53?&color=f5de53"/></a>
  <a href="https://huggingface.co/LightOriginsHQ/Light-O1-Preview"><img alt="Model License"
    src="https://img.shields.io/badge/Model_License-Apache_2.0-f5de53?&color=f5de53"/></a>
  <a href="pyproject.toml"><img alt="Python"
    src="https://img.shields.io/badge/Python-3.11-3776AB.svg"/></a>
  <br>
  <a href="https://www.lightorigins.com/en/blog/light-o1"><b>Tech Blog</b>👁️</a>
</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Downloads](#2-model-downloads)
3. [Playground](#3-playground)
4. [How to Run Locally](#4-how-to-run-locally)
5. [License](#5-license)
6. [Citation](#6-citation)
7. [Community](#7-community)
8. [Contact](#8-contact)


## 1. Introduction

Light-O1-Preview turns an instruction into whole-body humanoid action. It is pretrained on
structured human action recovered from internet video, which gives it a transferable action prior,
and then post-trained on purpose-collected data to adapt that prior to a target embodiment and to
human intent.

Given a prompt, the model answers in two parts: it states in language what the instruction requires
of the body, then generates the action. The plan is readable, so you can see what the model
understood before it moved. Actions come out as a `(frames, 138)` unified human action
representation at 20 FPS, which renders a character directly or drives a behavior foundation model
on a robot.

The tech blog compares Light-O1-Preview with two public text-to-action models, HY-Motion-1.0 and
Kimodo. On HY-Motion-Bench, a VLM judge decomposes each prompt into a checklist and scores the
generated action question by question.

| HY-Motion-Bench, SSAE | Overall |
| :--- | ---: |
| Light-O1-Preview | **78.0** |
| HY-Motion-1.0 | 74.7 |
| Kimodo | 61.4 |

Protocols, baselines and caveats are in the
[tech blog](https://www.lightorigins.com/en/blog/light-o1).

## 2. Model Downloads

| Model | Representation | Context | Download |
| :---: | :---: | :---: | :---: |
| Light-O1-Preview | `human_action_138_v1` | text | [🤗 Hugging Face](https://huggingface.co/LightOriginsHQ/Light-O1-Preview) |

The checkpoint ships the action decoder alongside the model weights:

```text
checkpoint/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── action_checkpoint_manifest.json
└── action_tokenizer/
    ├── decoder_config.json
    ├── manifest.json
    └── action_decoder.safetensors
```

The checkpoint's `architectures` field must be `Qwen3_5ActionForConditionalGeneration`. Special tokens and
weight keys retain their trained spelling.


## 3. Playground

Try Light-O1-Preview without any local setup at
**[Light-O1-Preview Playground](https://huggingface.co/spaces/LightOriginsHQ/Light-O1-Preview-playground)** —
enter a prompt, watch the reasoning stream, and inspect the generated action in a 3D viewer.


## 4. How to Run Locally

There are three ways to run Light-O1-Preview locally, in increasing order of setup: a one-shot
**command line** generation, the **Python API**, and a local **web console** with a 3D viewer.

Beyond that: running the Control Server and GPU API on separate hosts is covered in
[docs/deployment.md](docs/deployment.md), the HTTP endpoints in [docs/api.md](docs/api.md), the optional
MuJoCo simulation in [docs/sonic.md](docs/sonic.md), and tests and linting in
[docs/development.md](docs/development.md).

### 4.1 Requirements

| | |
| :--- | :--- |
| OS | Linux x86-64 |
| Python | 3.11 (the project pins `>=3.11,<3.12`) |
| GPU | NVIDIA, CUDA 13 compatible |

> [!NOTE]
> macOS and Windows are not supported for inference. To try the model without a GPU, use the
> [playground](https://huggingface.co/spaces/LightOriginsHQ/Light-O1-Preview-playground).

Download the weights from
[LightOriginsHQ/Light-O1-Preview](https://huggingface.co/LightOriginsHQ/Light-O1-Preview). The action decoder
ships inside the checkpoint — see [2. Model Downloads](#2-model-downloads).

### 4.2 Install

```bash
git clone https://github.com/lightorigins/Light-O1.git
cd Light-O1
uv sync --extra inference
```

### 4.3 Generate from the command line

```bash
uv run --extra inference light-deploy \
  --model /absolute/path/to/Light-O1-Preview \
  --prompt "a person waves with the right hand" \
  --thinking \
  --output human_action.npy
```

This writes a float32 `(frames, 138)` array at 20 FPS. Pass `--temperature`, `--top-p` or `--seed` to
control sampling.

### 4.4 Generate from Python

```python
from light_deploy.generate import Inference

inference = Inference("/absolute/path/to/Light-O1-Preview", device="cuda:0")
output = inference.generate("a person raises the right arm and waves", enable_thinking=True)

output.action     # (frames, 138) action array
output.reasoning  # the Thinking text
```

Two interchangeable backends sit behind this API, selected per instance with `backend=`:

- **`vllm`** (default) — the resident-GPU production runtime.
- **`transformers`** — torch-native, single-request, no vLLM. Works in lazy-init hosts such as
  ZeroGPU.

Both share the same masking FSM, prompt and output processors, and action decoder, so a token stream is
masked and parsed identically whichever backend produced it.

### 4.5 Web console

```bash
uv run --extra inference light-deploy-server \
  --model-path /absolute/path/to/Light-O1-Preview --port 8090
```

Open `http://127.0.0.1:8090` to enter prompts, watch the reasoning stream and inspect the generated action in
a 3D viewer. This also starts a managed GPU API on port 8030; change it with `--inference-port`. Model
loading is asynchronous, so generation becomes available only once that API reports ready. Without
`--model-path` the page starts without a model and lets you enter a checkpoint path.

> [!WARNING]
> This is a trusted-workstation service, not authenticated multi-user hosting — it permits local checkpoint
> selection and GPU work. Keep the localhost defaults, or put it behind an authenticated reverse proxy. Do
> not expose it directly to the Internet.


### 4.6 Sonic example: Unitree G1 simulation

Our [tech blog](https://www.lightorigins.com/en/blog/light-o1) demonstrates Light-O1-Preview on both
our LightBot robot and the Unitree G1. Light-O1-Preview outputs a unified action representation,
which is connected to each robot's BFM for execution: our own BFM for LightBot, and GEAR-SONIC as
the low-level motion controller for the Unitree G1. For the community, we provide a
[Sonic example](examples/sonic/README.md) that adapts this unified representation to GEAR-SONIC for
closed-loop G1 simulation in MuJoCo.

The example produces a simulation video and rollout metrics. It requires a separate GEAR-SONIC
low-latency policy checkpoint and runs in simulation only. See the [setup guide](docs/sonic.md) to run it
from the web console, or the [example README](examples/sonic/README.md) for standalone usage and details
of the action adapter.

## 5. License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The Light-O1-Preview weights are distributed under their own terms on the
[model page](https://huggingface.co/LightOriginsHQ/Light-O1-Preview). Required model and asset notices are
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).


## 6. Citation

```bibtex
@misc{lightorigins2026lighto1,
  title        = {Light-O1: Scaling Whole-Body Intelligence with Human Action Pretraining},
  author       = {Light Origins Team},
  year         = {2026},
  howpublished = {\url{https://www.lightorigins.com/en/blog/light-o1}}
}
```


## 7. Community

Join us on [Discord](https://discord.gg/zwZuD9JG), or scan the QR code to join the WeChat group:

<div align="center">
  <img src="docs/assets/wechat_group.png" alt="Light Origins WeChat group QR code" width="280"/>
</div>


## 8. Contact

Questions and bug reports are welcome as
[GitHub issues](https://github.com/lightorigins/Light-O1/issues). For anything else, reach us through
[Discord](https://discord.gg/zwZuD9JG) or the WeChat group above.
