# HTTP API

Two services expose HTTP: the resident **GPU API** (`light-deploy-api`) and the **Control Server**
(`light-deploy-server`). See [deployment.md](deployment.md) for how they are run.

## GPU API

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Readiness probe |
| `POST` | `/api/generate` | Generate an action |

`POST /api/generate` accepts a JSON `prompt` and independent reasoning and action sampling parameters.

It returns schema version 1: a float32-compatible `(frames, 138)` action at 20 FPS, the reasoning text and
the token IDs.

Defaults and limits:

- Thinking is on by default.
- Both temperatures default to `0.6`, both top-p values to `0.95`.
- Durations are bounded to 2–20 seconds.
- Concurrent requests return **HTTP 409**; context overflow returns **HTTP 422**.
- Initialization, generation and shutdown share one CUDA-owning thread.

## Control API

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/defaults` | Server defaults |
| `GET`/`POST` | `/api/service/{status,start,stop}` | Local GPU service lifecycle |
| `POST` | `/api/generations` | Start a generation (same request body as the GPU API) |
| `GET` | `/api/generations/{id}` | Poll for completion |
| `GET` | `/api/generations/{id}/action` | Fetch the skeleton |
| `GET` | `/api/sonic/jobs/{id}` | Sonic job status and range-enabled video |

Generation is asynchronous: `POST /api/generations` returns an id, which you poll. Control retains at most
**eight** generations, each with an immutable action digest. Expired or stopped-session IDs return **404**.

Sonic runs explicitly for a selected generation rather than automatically; see [sonic.md](sonic.md).
