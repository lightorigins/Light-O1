# Deployment

Running the Control Server and the GPU API on separate hosts, and the operational knobs for both.

The browser talks only to the Control Server. Control communicates with the GPU API over HTTP, including in
local mode, so no shared checkpoint mount or shared result files are needed between the two hosts.

## Separate GPU and Control hosts

On the GPU host:

```bash
uv sync --extra inference
uv run --extra inference light-deploy-api \
  --model /absolute/path/to/Light-O1-Preview --host 0.0.0.0 --port 8030
```

On the Control host:

```bash
uv sync
uv run light-deploy-server --inference-url http://GPU_HOST:8030 --port 8090
```

Remote mode disables the local model controls. Disconnecting cancels Control's pending requests and clears
its session; it does not stop the remote GPU service. A generation already running there can finish and will
temporarily return busy to new requests. Manage remote model loading and shutdown on the GPU host itself.

> [!WARNING]
> These are trusted-workstation services, not authenticated multi-user hosting. They permit local checkpoint
> selection and GPU work. Keep the localhost defaults, or use a trusted network or SSH tunnel behind an
> authenticated reverse proxy. Do not expose either API directly to the public Internet.

## Operational notes

- **CUDA forward compatibility.** If the host needs NVIDIA forward-compatibility libraries, set
  `LIGHT_DEPLOY_CUDA_COMPAT_DIR` to the directory containing `libcuda.so.1` before launch.
- **Scratch files.** Private session directories under `/tmp`, relocatable with `--temp-root`.
- **Logs.** The Control terminal prints the local inference log location; detailed model errors are in that
  log rather than on the console.
- **GPU API tuning.** `light-deploy-api` accepts `--device`, `--max-model-len`, `--kv-cache-memory-bytes` and
  `--deterministic`. `light-deploy-server` accepts the same when it manages a local GPU API.

## Rebuilding the web console

The wheel ships a production build of the web console. Rebuild it after changing browser sources:

```bash
cd webui
npm ci
npm run build
```

## Repository layout

```text
light_deploy/          Model runner, ActionDecoder and resident GPU HTTP API
  action_tokenizer/    Action representation, decoder bundle and forward kinematics
server/                Control API, local process lifecycle, generation store and previews
webui/                 Browser source; its production build ships in server/webui/
examples/sonic/        Standalone policy-driven MuJoCo example and required G1 assets
tests/                 Decoder, API, lifecycle, browser and simulation tests
```
