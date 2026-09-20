const DEFAULT_TIMEOUT_MS = 10_000;
const GENERATION_TIMEOUT_MS = 3_700_000;
const POLL_INTERVAL_MS = 250;

export class ApiError extends Error {
  constructor(message, { status = 0, payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function request(path, { method = "GET", body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const form = body instanceof FormData;

  try {
    const response = await fetch(path, {
      method,
      headers: body === undefined || form ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : form ? body : JSON.stringify(body),
      signal: controller.signal,
    });
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok || (payload && typeof payload === "object" && payload.ok === false)) {
      const detail = payload && typeof payload === "object" ? payload.detail || payload.error : payload;
      const message = Array.isArray(detail)
        ? detail.map((item) => `${(item.loc || []).join(".")}: ${item.msg || JSON.stringify(item)}`).join("; ")
        : typeof detail === "object" && detail ? detail.message || JSON.stringify(detail) : detail;
      throw new ApiError(`HTTP ${response.status}: ${message || response.statusText || "Request failed"}`, {
        status: response.status,
        payload,
      });
    }
    return payload;
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new ApiError(`Request timed out after ${timeoutMs} ms`);
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError(error?.message || "Control Server is unavailable");
  } finally {
    window.clearTimeout(timeout);
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function poll(path, timeoutMs = GENERATION_TIMEOUT_MS) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const payload = await request(path);
    if (payload.state === "done") return payload;
    if (["error", "cancelled"].includes(payload.state)) {
      throw new ApiError(payload.error?.message || "Runtime job failed", { payload });
    }
    await delay(POLL_INTERVAL_MS);
  }
  throw new ApiError(`Runtime job timed out after ${Math.round(timeoutMs / 1000)} seconds`);
}

async function generateOne(payload, seed) {
  const { text, num_generations: _count, ...parameters } = payload;
  const created = await request("/api/generations", {
    method: "POST",
    body: { ...parameters, prompt: text, seed },
    timeoutMs: 30_000,
  });
  return poll(`/api/generations/${encodeURIComponent(created.generation_id)}`);
}

export const controlApi = {
  defaults() {
    return request("/api/defaults", { timeoutMs: 3_000 });
  },

  health() {
    return request("/api/health", { timeoutMs: 3_000 });
  },

  serviceStatus() {
    return request("/api/service/status", { timeoutMs: 3_000 });
  },

  startService(payload) {
    return request("/api/service/start", { method: "POST", body: payload, timeoutMs: 30_000 });
  },

  stopService() {
    return request("/api/service/stop", { method: "POST", timeoutMs: 30_000 });
  },

  async generate(payload) {
    const count = Math.max(1, Math.min(2, Number(payload.num_generations) || 1));
    const baseSeed = Number(payload.seed) || 0;
    const generations = [];
    for (let index = 0; index < count; index += 1) {
      generations.push(await generateOne(payload, baseSeed + index));
    }
    return { ...generations[0], generations };
  },

  humanoid(generationId) {
    return request(`/api/generations/${encodeURIComponent(generationId)}/action`);
  },

  async startSonic(generationId, payload) {
    const created = await request(
      `/api/generations/${encodeURIComponent(generationId)}/sonic/sim`,
      { method: "POST", body: payload, timeoutMs: 30_000 },
    );
    return poll(`/api/sonic/jobs/${encodeURIComponent(created.job_id)}`);
  },

  sonicVideoUrl(jobId) {
    return `/api/sonic/jobs/${encodeURIComponent(jobId)}/video`;
  },
};
