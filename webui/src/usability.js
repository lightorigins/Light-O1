export const HISTORY_KEY = "light-deploy.public.prompt-history.v1";
const LANGUAGE_KEY = "light-deploy.public.language";
export function readHistory(storage) {
  try {
    const value = JSON.parse(storage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(value) ? [...new Set(value.filter((v) => typeof v === "string" && v.trim() && v.length <= 4000))].slice(0, 10) : [];
  } catch { return []; }
}
export function rememberPrompt(storage, prompt) {
  const text = prompt.trim();
  const history = readHistory(storage);
  if (!text || text.length > 4000) return history;
  const next = [text, ...history.filter((v) => v !== text)].slice(0, 10);
  try { storage.setItem(HISTORY_KEY, JSON.stringify(next)); } catch { /* Private browsing can disable storage. */ }
  return next;
}

const chinese = {
  "Action inference workspace": "动作推理工作台", WORKSPACE: "工作区", Control: "控制服务",
  "Inference only": "仅推理与仿真", Inference: "推理", "Model service": "模型服务",
  "Checkpoint path": "模型路径", "Loaded model": "已加载模型", "Action Representation": "动作表示",
  Device: "设备", Context: "上下文长度", "KV cache · GiB": "KV 缓存 · GiB",
  Deterministic: "确定性模式", "Eager reproducibility mode": "可复现执行模式",
  "Start service": "启动服务", "Connect service": "连接服务", Stop: "停止", Disconnect: "断开",
  "Generation controls": "生成参数", "Reasoning token budget": "推理 Token 预算",
  "Reasoning temperature": "推理温度", "Reasoning top-p": "推理 top-p", "Action min · sec": "最短动作 · 秒",
  "Action max · sec": "最长动作 · 秒", "Action temperature": "动作温度", "Action top-p": "动作 top-p",
  Seed: "随机种子", Candidates: "候选数量", "Sonic MuJoCo controls": "Sonic 仿真参数",
  "Replan frames": "重规划帧数", Lookahead: "前瞻帧数", ACTIVITY: "活动", "Session Log": "会话日志", Clear: "清空",
  INPUT: "输入", "Describe an action": "描述一个动作", Prompt: "动作描述", "Think before action": "先思考再生成动作",
  "Reasoning mode": "推理模式", "Generate action": "生成动作", "Generating…": "正在生成…", "Run Sonic": "运行 Sonic",
  "Ctrl + Enter to generate": "Ctrl + Enter 生成动作", "INFERENCE OUTPUT": "推理输出", "Reasoning & action": "推理与动作",
  Reasoning: "推理文本", Frames: "帧数", Duration: "时长", "Action ID": "动作 ID",
  "KINEMATICS · MESH": "运动学 · 人体网格", "Human Action View": "人体动作视图", "Reset view": "重置视角",
  "Candidate 1": "候选 1", "Candidate 2": "候选 2", "DYNAMICS · MUJOCO": "动力学 · MUJOCO",
  "MuJoCo dynamics preview": "MuJoCo 动力学预览", Templates: "动作模板", "Recent prompts": "最近使用",
  "Clear history": "清空历史", "History stays in this browser. Only successful prompts are saved.": "历史仅保存在当前浏览器，仅记录成功生成的描述。",
  "Download result ZIP": "下载结果 ZIP", "138D action · reasoning · generation settings": "138 维动作 · 推理文本 · 生成参数",
  "Download Sonic video": "下载 Sonic 视频", "Choose a template": "选择模板", "Choose a recent prompt": "选择历史描述",
  Connected: "已连接", Offline: "离线", Available: "可用", Unavailable: "不可用", "Server offline": "服务离线",
  "Service ready": "服务就绪", "Service stopped": "服务已停止", "Service starting": "服务启动中", "Service error": "服务错误",
  ready: "就绪", stopped: "已停止", starting: "启动中", error: "错误",
};
let language = "en";
let labels = [];
export function t(text) { return language === "zh" ? chinese[text] || text : text; }
export function initUsability({ prompt, onLanguage }) {
  let storage;
  try { storage = window.localStorage; language = storage.getItem(LANGUAGE_KEY) === "zh" ? "zh" : "en"; } catch { /* Optional persistence. */ }
  const select = document.querySelector("[data-language]");
  // Register static leaf labels only. Never translate model output, prompt text or logs.
  labels = [...document.querySelectorAll("span, strong, small, h1, h2, b, button, a")]
    .filter((el) => !el.children.length && chinese[el.textContent.trim()] && !el.matches("[data-system-state-label], [data-control-status], [data-inference-status], [data-sonic-status], [data-generate-label], [data-service-start], [data-service-stop]"))
    .map((el) => [el, el.textContent.trim()]);
  const templates = document.querySelector("[data-templates]");
  const history = document.querySelector("[data-history]");
  const examples = [
    ["Wave", "挥手", "a person raises the right arm and waves"],
    ["Walk forward", "向前走", "a person walks forward three steps and stops"],
    ["Turn left", "左转", "a person turns left in place"],
    ["Stretch", "伸展", "a person stretches both arms out to the sides"],
  ];
  function options(element, placeholder, entries) {
    element.replaceChildren(new Option(t(placeholder), ""), ...entries.map(([label, value]) => new Option(label, value)));
  }
  function refreshHistory() {
    options(history, "Choose a recent prompt", readHistory(storage).map((text) => [text, text]));
  }
  function refresh() {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    select.value = language;
    labels.forEach(([el, text]) => { el.textContent = t(text); });
    options(templates, "Choose a template", examples.map(([en, zh, text]) => [language === "zh" ? zh : en, text]));
    refreshHistory();
    onLanguage();
  }
  select.addEventListener("change", () => {
    language = select.value === "zh" ? "zh" : "en";
    try { storage.setItem(LANGUAGE_KEY, language); } catch { /* Optional persistence. */ }
    refresh();
  });
  for (const element of [templates, history]) element.addEventListener("change", () => {
    if (element.value) { prompt.value = element.value; prompt.focus(); }
    element.value = "";
  });
  document.querySelector("[data-clear-history]").addEventListener("click", () => {
    try { storage.removeItem(HISTORY_KEY); } catch { /* Optional persistence. */ }
    refreshHistory();
  });
  refresh();
  return { remember(text) { rememberPrompt(storage, text); refreshHistory(); } };
}
