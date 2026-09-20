import assert from "node:assert/strict";
import { chromium } from "playwright-core";

const [baseUrl, screenshotPath, executablePath] = process.argv.slice(2);
assert(baseUrl && screenshotPath && executablePath, "expected base URL, screenshot path, and browser executable");

const parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19];

function frame(offset, raisedArm) {
  return [
    [offset, 0.9, 0],
    [offset + 0.09, 0.82, 0],
    [offset - 0.09, 0.82, 0],
    [offset, 1.01, 0],
    [offset + 0.09, 0.4, 0],
    [offset - 0.09, 0.4, 0],
    [offset, 1.13, 0],
    [offset + 0.09, -0.01, 0],
    [offset - 0.09, -0.01, 0],
    [offset, 1.27, 0],
    [offset + 0.09, -0.07, 0.13],
    [offset - 0.09, -0.07, 0.13],
    [offset, 1.43, 0],
    [offset + 0.08, 1.33, 0],
    [offset - 0.08, 1.33, 0],
    [offset, 1.61, 0],
    [offset + 0.22, raisedArm ? 1.52 : 1.33, 0],
    [offset - 0.22, 1.33, 0],
    [offset + 0.47, raisedArm ? 1.7 : 1.33, 0],
    [offset - 0.47, 1.33, 0],
    [offset + 0.71, raisedArm ? 1.82 : 1.33, 0],
    [offset - 0.71, 1.33, 0],
  ];
}

function action(index) {
  return {
    representation: "human_action_138_v1",
    profile: "humanoid22_v1",
    fps: 20,
    parents,
    positions: index === 0 ? [frame(0, false), frame(0, true)] : [frame(0, true), frame(0, true)],
    rotations: [0, 1].map((step) => parents.map((_, joint) => {
      const angle = joint >= 16 ? (index + step) * 0.35 : 0;
      return [Math.cos(angle / 2), 0, 0, Math.sin(angle / 2)];
    })),
  };
}

function json(route, payload, status = 200) {
  return route.fulfill({ status, contentType: "application/json", body: JSON.stringify(payload) });
}

const requests = [];
const pageErrors = [];
const sonicGenerations = [];
const sonicRequestBodies = [];
const generationRequestBodies = [];
const sonicPollCounts = new Map();
const generationPollCounts = new Map();
let serviceState = "stopped";
let generationCount = 0;
const browser = await chromium.launch({ executablePath, headless: true, args: ["--no-sandbox"] });

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, reducedMotion: "reduce" });
  const page = await context.newPage();
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("request", (request) => requests.push(new URL(request.url()).pathname));
  await page.route("**/*", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/api/defaults") {
      return json(route, {
        ok: true,
        defaults: {
          model_path: "/public/reasoning_text2action_step3908_human_action_138_v1",
          device: "cuda:0",
          max_model_len: 1024,
          kv_cache_memory_bytes: 2147483648,
          deterministic: false,
          generation: {
            enable_thinking: false,
            reasoning_token_budget: 384,
            reasoning_temperature: 0.6,
            reasoning_top_p: 0.95,
            action_min_seconds: 2,
            action_max_seconds: 20,
            action_temperature: 0.6,
            action_top_p: 0.95,
            seed: 7,
          },
          capabilities: { inference: true, thinking: true, vision: true, sonic: true },
        },
      });
    }
    if (path === "/api/service/status") {
      return json(route, {
        state: serviceState,
        model_path: serviceState === "ready" ? "reasoning_text2action_step3908_human_action_138_v1" : null,
        ready: serviceState === "ready" ? { representation: "human_action_138_v1" } : null,
        error: null,
      });
    }
    if (path === "/api/service/start" && request.method() === "POST") {
      serviceState = "ready";
      return json(route, { state: "ready", ready: { representation: "human_action_138_v1" } });
    }
    if (path === "/api/service/stop" && request.method() === "POST") {
      serviceState = "stopped";
      return json(route, { state: "stopped" });
    }
    if (path === "/api/generations" && request.method() === "POST") {
      generationRequestBodies.push(request.postData());
      generationCount += 1;
      return json(route, { generation_id: `g${generationCount}`, state: "queued" });
    }
    const generationStatus = path.match(/^\/api\/generations\/(g[12])$/);
    if (generationStatus) {
      const count = generationPollCounts.get(generationStatus[1]) || 0;
      generationPollCounts.set(generationStatus[1], count + 1);
      if (!count) return json(route, { state: "running" });
      const index = Number(generationStatus[1].slice(1)) - 1;
      return json(route, {
        generation_id: generationStatus[1],
        state: "done",
        representation: "human_action_138_v1",
          num_frames: 2,
          duration_s: 0.1,
          seed: 7 + index,
          action_sha256: (index ? "b" : "a").repeat(64),
          reasoning: `candidate ${index + 1} action plan`,
      });
    }
    const actionRequest = path.match(/^\/api\/generations\/(g[12])\/action$/);
    if (/^\/api\/generations\/g[12]\/download$/.test(path)) {
      return route.fulfill({ status: 200, contentType: "application/zip", body: "test-download" });
    }
    if (actionRequest) return json(route, action(Number(actionRequest[1].slice(1)) - 1));
    const sonicStart = path.match(/^\/api\/generations\/(g[12])\/sonic\/sim$/);
    if (sonicStart && request.method() === "POST") {
      sonicRequestBodies.push(request.postDataJSON());
      sonicGenerations.push(sonicStart[1]);
      return json(route, { job_id: `sonic-${sonicStart[1]}`, state: "running" });
    }
    const sonicStatus = path.match(/^\/api\/sonic\/jobs\/(sonic-g[12])$/);
    if (sonicStatus) {
      const pollCount = sonicPollCounts.get(sonicStatus[1]) || 0;
      sonicPollCounts.set(sonicStatus[1], pollCount + 1);
      return json(route, {
        job_id: sonicStatus[1],
        state: pollCount ? "done" : "running",
        result: pollCount ? { tracking_score: 0.95, frames: 2 } : null,
      });
    }
    if (/^\/api\/sonic\/jobs\/sonic-g[12]\/video$/.test(path)) {
      return route.fulfill({ status: 200, contentType: "video/mp4", body: "" });
    }
    return route.continue();
  });

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator("[data-control-status]").waitFor({ state: "visible" });
  assert.equal(await page.locator("[data-control-status]").textContent(), "Connected");
  assert.equal(await page.locator("[data-sonic-status]").textContent(), "Available");
  assert.equal(await page.locator("[data-skeleton-panel] h2").textContent(), "Human Action View");
  assert.equal(await page.locator("[data-sonic-panel] h2").textContent(), "G1 · Sonic MuJoCo");
  assert(await page.locator("[data-control-workspace]").isVisible());
  assert(await page.locator("[data-activity-panel]").isVisible());
  assert(await page.locator("[data-prompt-composer]").isVisible());
  assert.equal(await page.locator('input[type="file"]').count(), 0);
  assert(!/add images|vision/i.test(await page.locator("body").innerText()));
  const thinkingToggle = page.locator("[data-thinking-toggle]");
  assert.equal(await thinkingToggle.getAttribute("aria-checked"), "true");
  assert.equal(await page.locator("[data-thinking-state]").textContent(), "ON");
  await thinkingToggle.click();
  assert.equal(await thinkingToggle.getAttribute("aria-checked"), "false");
  assert.equal(await page.locator("[data-thinking-state]").textContent(), "OFF");
  await thinkingToggle.click();
  assert.equal(await thinkingToggle.getAttribute("aria-checked"), "true");
  assert.equal(await page.locator("[data-preview-mode]").count(), 0);
  assert.equal(await page.locator("[data-generate]").locator("xpath=parent::*").getAttribute("class"), "composer-actions");
  assert.equal(await page.locator("[data-sonic]").locator("xpath=parent::*").getAttribute("class"), "composer-actions");
  const visualDesign = await page.evaluate(() => {
    const box = (selector) => {
      const rect = document.querySelector(selector).getBoundingClientRect();
      return { top: rect.top, bottom: rect.bottom, height: rect.height };
    };
    const style = (selector) => getComputedStyle(document.querySelector(selector));
    return {
      stage: box("[data-preview-stage]"),
      timeline: box(".timeline"),
      generateBackground: style("[data-generate]").backgroundColor,
      generateColor: style("[data-generate]").color,
      sonicBackground: style("[data-sonic]").backgroundColor,
      sonicColor: style("[data-sonic]").color,
      fieldLabelSize: Number.parseFloat(style(".field > span").fontSize),
      inputSize: Number.parseFloat(style("input").fontSize),
      reasoningSize: Number.parseFloat(style(".reasoning-block p").fontSize),
      visualTitleSize: Number.parseFloat(style(".visual-heading h2").fontSize),
    };
  });
  assert(visualDesign.stage.height >= 300, `skeleton stage collapsed to ${visualDesign.stage.height}px`);
  assert(visualDesign.timeline.height <= 60, `timeline expanded to ${visualDesign.timeline.height}px`);
  assert(Math.abs(visualDesign.stage.bottom - visualDesign.timeline.top) < 1);
  assert.equal(visualDesign.sonicBackground, visualDesign.generateBackground);
  assert.equal(visualDesign.sonicColor, visualDesign.generateColor);
  assert(visualDesign.fieldLabelSize >= 11);
  assert(visualDesign.inputSize >= 12);
  assert(visualDesign.reasoningSize >= 12);
  assert(visualDesign.visualTitleSize >= 16);

  await page.locator("[data-service-start]").click();
  await page.locator("[data-inference-status]").filter({ hasText: "ready" }).waitFor();
  assert.equal(
    await page.locator("[data-loaded-model]").textContent(),
    "reasoning_text2action_step3908_human_action_138_v1",
  );
  assert.equal(await page.locator("[data-action-representation]").textContent(), "human_action_138_v1");
  await page.locator("[data-prompt]").fill("a person raises the right arm and waves");
  await page.locator("[data-generation-count]").selectOption("2");
  await page.locator("[data-generate]").click();
  await page.keyboard.press("Control+Enter");
  await page.keyboard.press("Control+Enter");
  assert.equal(generationCount, 1, "keyboard shortcuts must not bypass busy state");
  await page.locator("[data-reasoning-state]").filter({ hasText: "Complete" }).waitFor();
  assert.equal(JSON.parse(generationRequestBodies[0]).enable_thinking, true);
  assert.deepEqual(JSON.parse(generationRequestBodies[0]).images ?? [], []);
  assert.equal(await page.locator("[data-reasoning]").textContent(), "candidate 1 action plan");
  assert.equal(await page.locator("[data-runtime-frames]").textContent(), "2");
  assert.equal(await page.locator("[data-runtime-duration]").textContent(), "0.10 s");

  const canvas = page.locator("[data-preview-canvas]");
  await page.waitForTimeout(100);
  assert.equal(await canvas.getAttribute("data-renderer"), "three");
  await page.waitForFunction(() => document.querySelector("[data-preview-canvas]").dataset.avatar === "mannequin");
  await canvas.screenshot({ path: `${screenshotPath}.mesh.png` });
  assert.equal(await canvas.getAttribute("data-camera-view"), "sonic-mujoco");
  const firstCanvas = await canvas.evaluate((element) => element.toDataURL());
  assert.equal(await page.locator("[data-action-identity]").textContent(), "a".repeat(12));
  assert.equal(await page.locator('[data-candidate-index="0"]').getAttribute("aria-pressed"), "true");

  await page.locator("[data-sonic]").click();
  await page.locator("[data-sonic-job-state]").filter({ hasText: "Running" }).waitFor();
  assert(await page.locator('[data-candidate-index="0"]').isDisabled());
  assert(await page.locator('[data-candidate-index="1"]').isDisabled());
  await page.locator('[data-candidate-index="1"]').evaluate((button) => {
    button.disabled = false;
    button.click();
  });
  await page.locator("[data-log] li").filter({ hasText: "Sonic completed · candidate 1 · g1" }).waitFor();
  await page.waitForTimeout(100);
  assert(await page.locator("[data-second-canvas]").isVisible());
  const secondCanvas = await page.locator("[data-second-canvas]").evaluate((element) => element.toDataURL());
  assert.equal(await page.locator("[data-action-identity]").textContent(), "b".repeat(12));
  assert.equal(await page.locator('[data-candidate-index="1"]').getAttribute("aria-pressed"), "true");
  assert.notEqual(firstCanvas, secondCanvas, "two candidates must render independently");
  assert.equal(await page.locator("[data-reasoning]").textContent(), "candidate 2 action plan");
  assert.equal(await page.locator("[data-sonic-job-state]").textContent(), "Idle");
  assert(await page.locator("[data-sonic-video]").isHidden());
  assert.match(await page.locator("[data-log]").textContent(), /candidate 1 · g1 · tracking_score 0\.95/);

  await page.waitForFunction(() => !document.querySelector("[data-sonic]").disabled);
  await page.locator("[data-sonic]").click();
  await page.locator("[data-sonic-job-state]").filter({ hasText: "Running" }).waitFor();
  assert(await page.locator('[data-candidate-index="0"]').isDisabled());
  assert(await page.locator('[data-candidate-index="1"]').isDisabled());
  await page.locator("[data-sonic-job-state]").filter({ hasText: "Done" }).waitFor();
  assert(await page.locator("[data-sonic-video]").isVisible());
  assert.match(await page.locator("[data-sonic-video]").getAttribute("src"), /sonic-g2\/video/);
  await page.locator("[data-sonic-video]").evaluate((video) => {
    Object.defineProperty(video, "currentTime", { configurable: true, value: 0.05, writable: true });
    video.dispatchEvent(new Event("timeupdate"));
  });
  assert.equal(await page.locator("[data-frame-label]").textContent(), "2 / 2");
  await page.locator("[data-timeline]").fill("0");
  await page.locator("[data-timeline]").dispatchEvent("input");
  assert.equal(await page.locator("[data-sonic-video]").evaluate((video) => video.currentTime), 0);
  await page.locator("[data-sonic-video]").evaluate((video) => {
    let paused = true;
    Object.defineProperty(video, "requestVideoFrameCallback", { configurable: true, value: undefined });
    Object.defineProperty(video, "cancelVideoFrameCallback", { configurable: true, value: undefined });
    Object.defineProperty(video, "paused", { configurable: true, get: () => paused });
    video.play = () => {
      paused = false;
      video.dataset.playCalled = "true";
      video.dispatchEvent(new Event("play"));
      return Promise.resolve();
    };
    video.pause = () => {
      paused = true;
      video.dataset.pauseCount = String(Number(video.dataset.pauseCount || 0) + 1);
      video.dispatchEvent(new Event("pause"));
    };
  });
  await page.locator("[data-play]").click();
  assert.equal(await page.locator("[data-sonic-video]").getAttribute("data-play-called"), "true");
  await page.locator("[data-sonic-video]").evaluate((video) => {
    video.currentTime = 0.05;
  });
  await page.locator("[data-frame-label]").filter({ hasText: "2 / 2" }).waitFor();
  assert.match(await page.locator("[data-log]").textContent(), /candidate 2 · g2 · tracking_score 0\.95 · frames 2/);

  await page.locator("[data-sonic]").click();
  await page.locator("[data-sonic-job-state]").filter({ hasText: "Done" }).waitFor();
  assert.equal(await page.locator("[data-sonic-video]").getAttribute("data-pause-count"), "1");
  await page.locator('[data-candidate-index="1"]').click();
  assert.equal(await page.locator("[data-sonic-job-state]").textContent(), "Done");
  assert(await page.locator("[data-sonic-video]").isVisible());
  assert.deepEqual(sonicGenerations, ["g1", "g2", "g2"]);
  assert(sonicRequestBodies.every((body) => !Object.hasOwn(body, "render")));

  assert.equal(await page.locator("[data-history] option").count(), 2);
  const resultDownload = page.waitForEvent("download");
  await page.locator("[data-download]").click();
  assert.equal((await resultDownload).suggestedFilename(), "action-g2.zip");
  assert(requests.includes("/api/generations/g2/download"));
  const originalPrompt = await page.locator("[data-prompt]").inputValue();
  await page.locator("[data-language]").selectOption("zh");
  assert.equal(await page.locator("[data-generate-label]").textContent(), "生成动作");
  assert.equal(await page.locator("[data-prompt]").inputValue(), originalPrompt);
  assert.equal(await page.locator("[data-reasoning]").textContent(), "candidate 2 action plan");
  await page.locator("[data-clear-history]").click();
  assert.equal(await page.locator("[data-history] option").count(), 1);
  for (const width of [390, 900, 1280, 1440]) {
    await page.setViewportSize({ width, height: 1100 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `horizontal overflow at ${width}`);
    assert(await page.locator(".interaction-pane").evaluate((el) => el.scrollWidth <= el.clientWidth + 1), `workspace clipped at ${width}`);
  }
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const downloadButton = page.locator("[data-video-download]");
  assert.equal(await downloadButton.evaluate((el) => getComputedStyle(el).textDecorationLine), "none");
  await downloadButton.screenshot({ path: `${screenshotPath}.download.png` });
  assert.deepEqual(pageErrors, []);
  assert(requests.includes("/api/generations/g2/action"));
  assert(requests.includes("/api/generations/g2/sonic/sim"));
  assert(!requests.some((path) => /humanoid-mesh|approve|send/.test(path)));
  process.stdout.write(
    `${JSON.stringify({
      selected_generation: "g2",
      action_request: "/api/generations/g2/action",
      sonic_request: "/api/generations/g2/sonic/sim",
      canvas_changed: firstCanvas !== secondCanvas,
      webgl_renderer: await canvas.getAttribute("data-renderer"),
      sonic_render_forced: sonicRequestBodies.every((body) => !Object.hasOwn(body, "render")),
      screenshot: screenshotPath,
    })}\n`,
  );
} finally {
  await browser.close();
}
