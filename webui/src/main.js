import "./styles.css";

import { controlApi } from "./api.js";
import { PreviewSurface } from "./preview.js";
import { initUsability, t } from "./usability.js";

const $ = (selector) => document.querySelector(selector);
const elements = {
  systemState: $("[data-system-state]"),
  systemStateLabel: $("[data-system-state-label]"),
  controlIndicator: $("[data-control-indicator]"),
  controlStatus: $("[data-control-status]"),
  inferenceIndicator: $("[data-inference-indicator]"),
  inferenceStatus: $("[data-inference-status]"),
  modelPath: $("[data-model-path]"),
  loadedContext: $("[data-loaded-context]"),
  loadedModel: $("[data-loaded-model]"),
  actionRepresentation: $("[data-action-representation]"),
  device: $("[data-device]"),
  maxModelLen: $("[data-max-model-len]"),
  kvCacheGb: $("[data-kv-cache-gb]"),
  deterministic: $("[data-deterministic]"),
  serviceStart: $("[data-service-start]"),
  serviceStop: $("[data-service-stop]"),
  prompt: $("[data-prompt]"),
  thinkingToggle: $("[data-thinking-toggle]"),
  thinkingState: $("[data-thinking-state]"),
  generationCount: $("[data-generation-count]"),
  generate: $("[data-generate]"),
  generateLabel: $("[data-generate-label]"),
  sonic: $("[data-sonic]"),
  sonicIndicator: $("[data-sonic-indicator]"),
  sonicStatus: $("[data-sonic-status]"),
  sonicReplan: $("[data-sonic-replan]"),
  sonicLookahead: $("[data-sonic-lookahead]"),
  sonicJobState: $("[data-sonic-job-state]"),
  sonicVideo: $("[data-sonic-video]"),
  sonicPlaceholder: $("[data-sonic-placeholder]"),
  runtimeFrames: $("[data-runtime-frames]"),
  runtimeDuration: $("[data-runtime-duration]"),
  actionIdentity: $("[data-action-identity]"),
  previewCaption: $("[data-preview-caption]"),
  previewSubtitle: $("[data-preview-subtitle]"),
  candidatePicker: $("[data-candidate-picker]"),
  reasoningState: $("[data-reasoning-state]"),
  reasoning: $("[data-reasoning]"),
  play: $("[data-play]"),
  resetView: $("[data-reset-view]"),
  timeline: $("[data-timeline]"),
  frameLabel: $("[data-frame-label]"),
  log: $("[data-log]"),
  clearLog: $("[data-clear-log]"),
};

const state = {
  service: "offline",
  lastServiceError: null,
  busy: false,
  generations: [],
  previews: [],
  selected: 0,
  renderedGenerationId: null,
  sonicAvailable: false,
  remote: false,
  sonicReason: "",
};

const secondPreview = new PreviewSurface($("[data-second-canvas]"));
const preview = new PreviewSurface($("[data-preview-canvas]"), {
  onFrame(frame, count) {
    secondPreview.setPlaying(false);
    secondPreview.seek(frame);
    elements.frameLabel.textContent = count ? `${frame + 1} / ${count}` : "— / —";
    elements.timeline.max = String(Math.max(0, count - 1));
    elements.timeline.value = String(frame);
  },
});

function value(name) {
  return document.querySelector(`[name="${name}"]`)?.value;
}

function numeric(name) {
  const parsed = Number(value(name));
  return Number.isFinite(parsed) ? parsed : undefined;
}

function log(message, kind = "info") {
  const item = document.createElement("li");
  item.dataset.kind = kind;
  const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const time = document.createElement("time");
  const copy = document.createElement("span");
  time.textContent = timestamp;
  copy.textContent = message;
  item.append(time, copy);
  elements.log.prepend(item);
  while (elements.log.children.length > 16) elements.log.lastElementChild.remove();
}

function setReady(element, ready) {
  element.dataset.ready = String(Boolean(ready));
}

function updateActions() {
  const ready = state.service === "ready";
  elements.serviceStart.disabled = !["stopped", "error"].includes(state.service) || state.busy;
  elements.serviceStop.disabled = ["stopped", "offline"].includes(state.service) || state.busy;
  elements.generate.disabled = !ready || state.busy;
  $("[data-download]").disabled = state.busy || !state.generations[state.selected];
  $("[data-service-hint]").textContent = state.busy
    ? (document.documentElement.lang === "zh-CN" ? "任务运行中，请等待结果。" : "A task is running. Waiting for its result.")
    : !ready ? (document.documentElement.lang === "zh-CN" ? "先启动或连接推理服务，才能生成动作。" : "Start or connect the inference service to generate actions.")
    : !state.sonicAvailable ? (document.documentElement.lang === "zh-CN" ? "推理已就绪；Sonic 未配置。请按 README 安装 Sonic 环境并配置策略权重。" : "Inference is ready; Sonic is not configured. Follow the README to install Sonic and configure its policy checkpoint.")
    : (document.documentElement.lang === "zh-CN" ? "选择一个候选，用于下载和 Sonic 仿真。" : "Select a candidate for downloads and Sonic simulation.");
  if (!state.busy && state.lastServiceError) $("[data-service-hint]").textContent += ` ${JSON.parse(state.lastServiceError)[3]}`;
  if (!state.busy && !state.sonicAvailable && state.sonicReason) $("[data-service-hint]").textContent += ` Sonic: ${state.sonicReason}`;
  elements.sonic.disabled = !ready || state.busy || !state.sonicAvailable || !state.generations[state.selected];
  elements.thinkingToggle.disabled = state.busy;
  elements.candidatePicker.querySelectorAll("button").forEach((button) => {
    button.disabled = state.busy;
  });
}

function renderService(status) {
  const becameReady = state.service !== "ready" && status.state === "ready";
  state.service = status.state;
  const connected = status.state !== "offline";
  const ready = status.state === "ready";
  elements.systemState.dataset.systemState = connected ? status.state : "offline";
  elements.systemStateLabel.textContent = t(connected ? `Service ${status.state}` : "Server offline");
  setReady(elements.controlIndicator, connected);
  setReady(elements.inferenceIndicator, ready);
  elements.controlStatus.textContent = t(connected ? "Connected" : "Offline");
  elements.inferenceStatus.textContent = t(status.state);
  elements.loadedContext.hidden = !ready;
  elements.loadedModel.textContent = ready && status.model_path ? status.model_path.split("/").pop() : "—";
  elements.actionRepresentation.textContent = ready && status.ready?.representation ? status.ready.representation : "—";
  updateActions();
  if (becameReady) log("Inference service is ready.", "success");
  const errorKey = status.error?.message
    ? JSON.stringify([status.service_id, status.error.stage, status.error.type, status.error.message])
    : null;
  if (errorKey && errorKey !== state.lastServiceError) {
    log(status.error.message, "error");
  }
  state.lastServiceError = errorKey;
  updateActions();
}

async function refreshService() {
  try {
    renderService(await controlApi.serviceStatus());
  } catch {
    renderService({ state: "offline" });
    elements.inferenceStatus.textContent = "Unavailable";
  }
}

function applyDefaults(payload) {
  const defaults = payload.defaults;
  state.remote = defaults.mode === "remote";
  for (const input of [elements.modelPath, elements.device, elements.maxModelLen, elements.kvCacheGb, elements.deterministic]) {
    input.disabled = state.remote;
  }
  elements.serviceStart.textContent = t(state.remote ? "Connect service" : "Start service");
  elements.serviceStop.textContent = t(state.remote ? "Disconnect" : "Stop");
  elements.modelPath.value = state.remote ? defaults.inference_url : defaults.model_path;
  elements.device.value = defaults.device;
  elements.maxModelLen.value = defaults.max_model_len;
  elements.kvCacheGb.value = defaults.kv_cache_memory_bytes / 2 ** 30;
  elements.deterministic.checked = defaults.deterministic;
  state.sonicAvailable = Boolean(defaults.capabilities.sonic);
  state.sonicReason = defaults.sonic_unavailable_reason || "";
  setReady(elements.sonicIndicator, state.sonicAvailable);
  elements.sonicStatus.textContent = t(state.sonicAvailable ? "Available" : "Unavailable");
  for (const [name, setting] of Object.entries(defaults.generation)) {
    const input = document.querySelector(`[name="${name}"]`);
    if (input) input.value = setting;
  }
  updateActions();
}

let videoFrameCallbackId = null;
let videoFrameCallbackKind = null;

function sonicVideoActive() {
  return !elements.sonicVideo.hidden && Boolean(elements.sonicVideo.getAttribute("src"));
}

function cancelVideoFrameSync() {
  if (videoFrameCallbackId === null) return;
  if (videoFrameCallbackKind === "video") {
    elements.sonicVideo.cancelVideoFrameCallback(videoFrameCallbackId);
  } else {
    window.cancelAnimationFrame(videoFrameCallbackId);
  }
  videoFrameCallbackId = null;
  videoFrameCallbackKind = null;
}

function syncPreviewFromSonic(seconds = elements.sonicVideo.currentTime) {
  if (!sonicVideoActive() || !Number.isFinite(seconds)) return;
  preview.setPlaying(false);
  preview.seek(Math.round(seconds * preview.fps));
  elements.play.textContent = elements.sonicVideo.paused || elements.sonicVideo.ended ? "▶" : "Ⅱ";
}

function scheduleVideoFrameSync() {
  cancelVideoFrameSync();
  if (!sonicVideoActive() || elements.sonicVideo.paused || elements.sonicVideo.ended) return;
  if (typeof elements.sonicVideo.requestVideoFrameCallback === "function") {
    videoFrameCallbackKind = "video";
    videoFrameCallbackId = elements.sonicVideo.requestVideoFrameCallback((_time, metadata) => {
      videoFrameCallbackId = null;
      videoFrameCallbackKind = null;
      syncPreviewFromSonic(metadata.mediaTime);
      scheduleVideoFrameSync();
    });
    return;
  }
  videoFrameCallbackKind = "animation";
  videoFrameCallbackId = window.requestAnimationFrame(() => {
    videoFrameCallbackId = null;
    videoFrameCallbackKind = null;
    syncPreviewFromSonic();
    scheduleVideoFrameSync();
  });
}

function stopSonicPlayback() {
  $("[data-video-download]").hidden = true;
  $("[data-video-download]").removeAttribute("href");
  cancelVideoFrameSync();
  elements.sonicVideo.hidden = true;
  elements.sonicVideo.pause();
  elements.sonicVideo.removeAttribute("src");
  elements.sonicVideo.load();
  elements.play.textContent = preview.playing ? "Ⅱ" : "▶";
}

function clearSonic(message = "Generate an action, then run Sonic for the selected candidate.") {
  elements.sonicJobState.textContent = "Idle";
  stopSonicPlayback();
  elements.sonicPlaceholder.hidden = false;
  elements.sonicPlaceholder.querySelector("small").textContent = message;
}

function clearInference() {
  state.generations = [];
  state.previews = [];
  state.selected = 0;
  state.renderedGenerationId = null;
  preview.setActions([]);
  secondPreview.setActions([]);
  $("[data-second-stage]").hidden = true;
  $("[data-selected-label]").textContent = "";
  elements.candidatePicker.hidden = true;
  elements.previewCaption.textContent = "Waiting for generation";
  elements.previewSubtitle.textContent = "Start the service and describe an action.";
  elements.runtimeFrames.textContent = "—";
  elements.runtimeDuration.textContent = "—";
  elements.actionIdentity.textContent = "No action";
  elements.reasoningState.textContent = "Standby";
  elements.reasoning.textContent = "Reasoning text will appear after generation.";
  elements.play.disabled = true;
  elements.resetView.disabled = true;
  elements.frameLabel.textContent = "— / —";
  clearSonic();
  updateActions();
}

async function startService() {
  state.busy = true;
  updateActions();
  log("Loading compact checkpoint…");
  try {
    const status = await controlApi.startService(state.remote ? {} : {
      model_path: elements.modelPath.value.trim(),
      device: elements.device.value.trim(),
      max_model_len: Number(elements.maxModelLen.value),
      kv_cache_memory_bytes: Math.round(Number(elements.kvCacheGb.value) * 2 ** 30),
      deterministic: elements.deterministic.checked,
    });
    if (status.state !== "ready") log("Inference process started; waiting for model readiness…");
  } catch (error) {
    log(`Service start failed: ${error.message}`, "error");
  } finally {
    state.busy = false;
    await refreshService();
  }
}

async function stopService() {
  state.busy = true;
  updateActions();
  try {
    await controlApi.stopService();
    clearInference();
    log("Inference service stopped.");
  } catch (error) {
    log(`Service stop failed: ${error.message}`, "error");
  } finally {
    state.busy = false;
    await refreshService();
  }
}

function generationRequest() {
  return {
    text: elements.prompt.value.trim(),
    enable_thinking: elements.thinkingToggle.getAttribute("aria-checked") === "true",
    reasoning_token_budget: numeric("reasoning_token_budget"),
    reasoning_temperature: numeric("reasoning_temperature"),
    reasoning_top_p: numeric("reasoning_top_p"),
    action_min_seconds: numeric("action_min_seconds"),
    action_max_seconds: numeric("action_max_seconds"),
    action_temperature: numeric("action_temperature"),
    action_top_p: numeric("action_top_p"),
    seed: numeric("seed"),
    num_generations: Number(elements.generationCount.value),
  };
}

function renderCandidate(index, announce = false) {
  const generation = state.generations[index];
  if (!generation) return;
  const selectionChanged = state.renderedGenerationId !== generation.generation_id;
  if (!selectionChanged && state.selected === index) return;
  state.selected = index;
  $("[data-selected-label]").textContent = `${t(index ? "Candidate 2" : "Candidate 1")} · Sonic / ZIP`;
  document.querySelectorAll("[data-candidate-view]").forEach((view) => {
    view.dataset.selected = String(Number(view.dataset.candidateView) === index);
  });
  elements.play.textContent = preview.playing ? "Ⅱ" : "▶";
  elements.play.disabled = preview.frameCount < 2;
  elements.resetView.disabled = preview.frameCount < 1;
  elements.previewSubtitle.textContent = `${generation.num_frames} frames · compact output`;
  elements.candidatePicker.querySelectorAll("button").forEach((button, buttonIndex) => {
    button.setAttribute("aria-pressed", String(buttonIndex === index));
  });
  elements.actionIdentity.textContent = generation.action_sha256
    ? generation.action_sha256.slice(0, 12)
    : generation.generation_id.slice(0, 12);
  elements.runtimeFrames.textContent = String(generation.num_frames);
  elements.runtimeDuration.textContent = `${generation.duration_s.toFixed(2)} s`;
  elements.reasoningState.textContent = "Complete";
  elements.reasoning.textContent = generation.reasoning || "No reasoning text was returned.";
  state.renderedGenerationId = generation.generation_id;
  if (selectionChanged) clearSonic("Run Sonic to simulate this candidate in MuJoCo.");
  updateActions();
  if (announce && selectionChanged) log(`Selected candidate ${index + 1}; previous Sonic output cleared.`);
}

async function generate() {
  if (state.busy || state.service !== "ready") return;
  const payload = generationRequest();
  if (!payload.text) {
    elements.prompt.focus();
    log("A action prompt is required.", "error");
    return;
  }
  state.busy = true;
  elements.generateLabel.textContent = t("Generating…");
  elements.previewCaption.textContent = "Autoregressive generation in progress";
  elements.previewSubtitle.textContent = "Reasoning and action use independent sampling controls.";
  elements.reasoningState.textContent = "Running";
  elements.reasoning.textContent = "Generating reasoning and action tokens…";
  clearSonic("Sonic becomes available after action generation.");
  updateActions();
  log("Generation submitted.");

  try {
    const response = await controlApi.generate(payload);
    state.renderedGenerationId = null;
    state.generations = response.generations;
    state.previews = await Promise.all(state.generations.map((item) => controlApi.humanoid(item.generation_id)));
    $("[data-second-stage]").hidden = state.previews.length < 2;
    secondPreview.setActions(state.previews.slice(1), 20);
    secondPreview.setPlaying(false);
    preview.setActions(state.previews.slice(0, 1), 20);
    // One shared timeline; a shorter candidate holds its last frame.
    preview.frameCount = Math.max(preview.frameCount, secondPreview.frameCount);
    preview.seek(0);
    library.remember(payload.text);
    elements.candidatePicker.hidden = state.generations.length < 2;
    state.generations.forEach((item, index) => {
      const seed = elements.candidatePicker.querySelector(`[data-candidate-seed="${index}"]`);
      if (seed) seed.textContent = `seed ${item.seed}`;
    });
    renderCandidate(0);
    elements.previewCaption.textContent = "Decoded human action";
    log(`Generation complete: ${state.generations.length} candidate(s).`, "success");
  } catch (error) {
    clearInference();
    elements.previewCaption.textContent = "Generation failed";
    elements.previewSubtitle.textContent = error.message;
    elements.reasoningState.textContent = "Error";
    elements.reasoning.textContent = error.message;
    log(`Generation failed: ${error.message}`, "error");
  } finally {
    state.busy = false;
    elements.generateLabel.textContent = t("Generate action");
    await refreshService();
  }
}

function publicSonicResult(result) {
  return Object.fromEntries(Object.entries(result || {}).filter(([key]) => !key.toLowerCase().includes("path")));
}

function logSonicResult(result, candidateIndex, generationId) {
  const metrics = Object.entries(publicSonicResult(result)).map(([key, metric]) => `${key} ${String(metric)}`);
  const detail = metrics.length ? ` · ${metrics.join(" · ")}` : "";
  log(`Sonic completed · candidate ${candidateIndex + 1} · ${generationId}${detail}`, "success");
}

async function runSonic() {
  const candidateIndex = state.selected;
  const generation = state.generations[candidateIndex];
  if (!generation || !state.sonicAvailable) return;
  const generationId = generation.generation_id;
  stopSonicPlayback();
  state.busy = true;
  elements.sonicJobState.textContent = "Running";
  elements.sonicPlaceholder.hidden = false;
  elements.sonicPlaceholder.querySelector("small").textContent = "Sonic is tracking the action in MuJoCo…";
  updateActions();
  log(`Sonic simulation started · candidate ${candidateIndex + 1} · ${generationId}.`);

  try {
    const job = await controlApi.startSonic(generationId, {
      replan_frames: Number(elements.sonicReplan.value),
      lookahead_frames: Number(elements.sonicLookahead.value),
    });
    logSonicResult(job.result, candidateIndex, generationId);
    if (state.generations[state.selected]?.generation_id !== generationId) return;
    elements.sonicJobState.textContent = "Done";
    elements.sonicVideo.src = `${controlApi.sonicVideoUrl(job.job_id)}?v=${Date.now()}`;
    elements.sonicVideo.hidden = false;
    $("[data-video-download]").href = controlApi.sonicVideoUrl(job.job_id);
    $("[data-video-download]").hidden = false;
    elements.sonicPlaceholder.hidden = true;
    preview.setPlaying(false);
    preview.seek(0);
    elements.play.textContent = "▶";
  } catch (error) {
    if (state.generations[state.selected]?.generation_id === generationId) {
      elements.sonicJobState.textContent = "Error";
      elements.sonicPlaceholder.querySelector("small").textContent = error.message;
    }
    log(`Sonic failed · candidate ${candidateIndex + 1} · ${generationId}: ${error.message}`, "error");
  } finally {
    state.busy = false;
    updateActions();
  }
}

elements.serviceStart.addEventListener("click", startService);
elements.serviceStop.addEventListener("click", stopService);
elements.generate.addEventListener("click", generate);
elements.sonic.addEventListener("click", runSonic);
elements.thinkingToggle.addEventListener("click", () => {
  const enabled = elements.thinkingToggle.getAttribute("aria-checked") !== "true";
  elements.thinkingToggle.setAttribute("aria-checked", String(enabled));
  elements.thinkingState.textContent = enabled ? "ON" : "OFF";
});
elements.candidatePicker.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => renderCandidate(Number(button.dataset.candidateIndex), true));
});
elements.play.addEventListener("click", async () => {
  if (sonicVideoActive()) {
    if (elements.sonicVideo.paused || elements.sonicVideo.ended) {
      if (elements.sonicVideo.ended) elements.sonicVideo.currentTime = 0;
      try {
        await elements.sonicVideo.play();
      } catch (error) {
        log(`Video playback failed: ${error.message}`, "error");
      }
    } else {
      elements.sonicVideo.pause();
    }
    return;
  }
  elements.play.textContent = preview.toggle() ? "Ⅱ" : "▶";
});
elements.resetView.addEventListener("click", () => {
  preview.resetView();
  secondPreview.resetView();
  if (sonicVideoActive()) elements.sonicVideo.currentTime = 0;
});
elements.timeline.addEventListener("input", () => {
  const frame = Number(elements.timeline.value);
  preview.seek(frame);
  if (sonicVideoActive()) elements.sonicVideo.currentTime = frame / preview.fps;
});
elements.sonicVideo.addEventListener("loadedmetadata", () => syncPreviewFromSonic());
elements.sonicVideo.addEventListener("timeupdate", () => syncPreviewFromSonic());
elements.sonicVideo.addEventListener("seeking", () => syncPreviewFromSonic());
elements.sonicVideo.addEventListener("play", () => {
  syncPreviewFromSonic();
  scheduleVideoFrameSync();
});
elements.sonicVideo.addEventListener("pause", () => {
  cancelVideoFrameSync();
  syncPreviewFromSonic();
});
elements.sonicVideo.addEventListener("ended", () => {
  cancelVideoFrameSync();
  syncPreviewFromSonic();
});
elements.clearLog.addEventListener("click", () => elements.log.replaceChildren());
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") generate();
});

async function initialize() {
  clearInference();
  try {
    applyDefaults(await controlApi.defaults());
    setReady(elements.controlIndicator, true);
    elements.controlStatus.textContent = "Connected";
    log("Control Server connected.", "success");
  } catch (error) {
    log(`Control Server unavailable: ${error.message}`, "error");
  }
  await refreshService();
  window.setInterval(refreshService, 1500);
}

const library = initUsability({ prompt: elements.prompt, onLanguage() {
  elements.serviceStart.textContent = t(state.remote ? "Connect service" : "Start service");
  elements.serviceStop.textContent = t(state.remote ? "Disconnect" : "Stop");
  elements.sonicStatus.textContent = t(state.sonicAvailable ? "Available" : "Unavailable");
  elements.generateLabel.textContent = t(state.busy ? "Generating…" : "Generate action");
  const selected = state.generations[state.selected];
  if (selected) $("[data-selected-label]").textContent = `${t(state.selected ? "Candidate 2" : "Candidate 1")} · Sonic / ZIP`;
  updateActions();
} });
$("[data-download]").addEventListener("click", async () => {
  const generation = state.generations[state.selected];
  if (!generation || state.busy) return;
  try {
    const response = await fetch(`/api/generations/${encodeURIComponent(generation.generation_id)}/download`);
    if (!response.ok) throw new Error(`Download HTTP ${response.status}: ${await response.text()}`);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = `action-${generation.generation_id}.zip`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { log(error.message, "error"); }
});
initialize();
