const $ = (id) => document.getElementById(id);
let selectedJobId = null;
let flexcacheSchema = {};
let flexcacheAvailability = {};
let qualityAvailability = {};
let renderedFlexcacheStrategy = "";
let renderedFlexcacheSchemaKey = "";
let appliedRequestDefaultsKey = "";
let refreshTimer = null;
let shuttingDown = false;
const CANCELLABLE_JOB_STATUSES = new Set(["queued", "dispatching", "running", "cancelling"]);

function parseLogRanks(value) {
  const text = value.trim();
  if (text === "all" || text === "*") return text;
  return text.split(/[;,]/).map((item) => item.trim()).filter(Boolean).map((item) => Number(item));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function renderStaticConfig(cfg) {
  $("configPath").textContent = cfg.config_path;
  const ready = Boolean(cfg.ready);
  const stage = cfg.worker_stage || {};
  const status = stage.message || cfg.worker_status?.message || (ready ? "Worker is ready." : "Worker is loading.");
  $("workerStatus").textContent = status;
  renderWorkerDetails(stage, ready);
  $("submitBtn").disabled = !ready;
  $("submitBtn").textContent = ready ? "Start Request" : "Waiting For Worker";
  const rows = [
    ["model", cfg.model?.name],
    ["ckpt", cfg.model?.ckpt_dir],
    ["gpus", cfg.launch?.gpus_per_node],
    ["nodes", cfg.launch?.num_nodes],
    ["cfp", cfg.parallel?.cfp],
    ["cp", cfg.derived?.cp_size],
    ["attn", cfg.infer?.attn_type],
  ];
  $("staticConfig").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v ?? ""}</dd>`).join("");
  applyRequestDefaults(cfg.model?.name || "", cfg.request_defaults || {});
  flexcacheSchema = cfg.flexcache?.schema || {};
  flexcacheAvailability = cfg.flexcache?.availability || {};
  qualityAvailability = cfg.quality?.availability || {};
  const currentStrategy = $("flexcacheStrategy").value;
  const strategies = cfg.flexcache?.strategies || Object.keys(flexcacheSchema);
  $("flexcacheStrategy").innerHTML = strategies.map((name) => {
    const availability = flexcacheAvailability[name] || { available: true };
    const disabled = availability.available === false ? "disabled" : "";
    const label = availability.available === false ? `${name} (${availability.reason})` : name;
    return `<option value="${name}" ${disabled}>${label}</option>`;
  }).join("");
  const currentAvailable = (flexcacheAvailability[currentStrategy] || { available: true }).available !== false;
  if (strategies.includes(currentStrategy) && currentAvailable) {
    $("flexcacheStrategy").value = currentStrategy;
  } else {
    $("flexcacheStrategy").value = "origin";
  }
  renderFlexcacheParams({ preserveValues: true });
  if (!$("runLog").dataset.initialized) {
    $("runLog").checked = Boolean(cfg.output?.run_log);
    $("memory").checked = Boolean(cfg.output?.memory);
    $("timer").checked = Boolean(cfg.output?.timer);
    $("outputRoot").value = cfg.output?.root_dir || "outputs";
    const ranks = cfg.output?.log_ranks;
    $("logRanks").value = Array.isArray(ranks) ? ranks.join(",") : (ranks || "0");
    $("runLog").dataset.initialized = "1";
  }
}

function renderWorkerDetails(stage, ready) {
  const box = $("workerDetails");
  if (ready || !stage) {
    box.hidden = true;
    box.textContent = "";
    return;
  }
  const lines = [];
  if (stage.stage) lines.push(`stage: ${stage.stage}`);
  if (stage.command) lines.push(stage.command);
  if (Array.isArray(stage.recent) && stage.recent.length > 0) {
    lines.push(...stage.recent.slice(-8));
  }
  box.textContent = lines.join("\n");
  box.hidden = lines.length === 0;
}

function applyRequestDefaults(modelName, defaults) {
  const defaultsKey = `${modelName}:${JSON.stringify(defaults)}`;
  if (!modelName || appliedRequestDefaultsKey === defaultsKey) return;
  $("role").value = defaults.role ?? "user";
  $("prompt").value = defaults.prompt ?? "";
  $("negativePrompt").value = defaults.negative_prompt ?? "";
  $("seed").value = defaults.seed ?? "";
  $("width").value = defaults.width ?? 1024;
  $("height").value = defaults.height ?? 1024;
  $("frames").value = defaults.frame_num ?? 1;
  $("steps").value = defaults.num_inference_steps ?? "";
  $("solver").value = defaults.sample_solver ?? "";
  appliedRequestDefaultsKey = defaultsKey;
}

function collectFlexcacheValues() {
  const values = {};
  document.querySelectorAll("[data-flexcache-param]").forEach((input) => {
    values[input.dataset.flexcacheParam] = input.value;
  });
  return values;
}

function renderFlexcacheParams({ preserveValues = false } = {}) {
  const strategy = $("flexcacheStrategy").value;
  const fields = flexcacheSchema[strategy] || [];
  const schemaKey = JSON.stringify(fields.map((field) => [field.name, field.default, field.type]));
  if (preserveValues && strategy === renderedFlexcacheStrategy && schemaKey === renderedFlexcacheSchemaKey) {
    return;
  }
  const previousValues = preserveValues ? collectFlexcacheValues() : {};
  $("flexcacheParams").innerHTML = fields.map((field) => {
    const value = previousValues[field.name] ?? field.default ?? "";
    const numeric = field.type.includes("int") || field.type.includes("float");
    const step = field.type.includes("float") ? "0.01" : "1";
    const type = numeric ? "number" : "text";
    return `
      <label>
        ${field.name}
        <input data-flexcache-param="${field.name}" type="${type}" value="${value}" step="${step}">
      </label>
    `;
  }).join("");
  renderedFlexcacheStrategy = strategy;
  renderedFlexcacheSchemaKey = schemaKey;
}

function renderJobs(jobs) {
  const container = $("jobs");
  const activeIds = new Set();
  let nextNode = container.firstElementChild;
  jobs.forEach((job) => {
    activeIds.add(job.job_id);
    const card = ensureJobCard(job);
    updateJobCard(card, job);
    if (card !== nextNode) {
      container.insertBefore(card, nextNode);
    } else {
      nextNode = nextNode?.nextElementSibling || null;
    }
  });
  container.querySelectorAll(".job[data-job-id]").forEach((card) => {
    if (!activeIds.has(card.dataset.jobId)) card.remove();
  });
}

function ensureJobCard(job) {
  let card = document.querySelector(`.job[data-job-id="${job.job_id}"]`);
  if (card) return card;
  card = document.createElement("div");
  card.className = "job";
  card.dataset.jobId = job.job_id;
  card.innerHTML = `
    <div class="jobTop">
      <strong data-role="requestId"></strong>
      <span class="strategy" data-role="strategy"></span>
      <span class="status" data-role="status"></span>
    </div>
    <div data-role="preview"></div>
    <div data-role="metrics"></div>
    <div class="jobEval" data-role="evalPanel">
      <label>
        Reference Task ID
        <input data-role="evalReference" type="text" placeholder="origin job or request id">
      </label>
      <div class="toggles compactToggles">
        <label><input data-quality-metric="fid" type="checkbox"> FID</label>
        <label><input data-quality-metric="fvd" type="checkbox"> FVD</label>
        <label><input data-quality-metric="psnr" type="checkbox"> PSNR</label>
        <label><input data-quality-metric="ssim" type="checkbox"> SSIM</label>
        <label><input data-quality-metric="lpips" type="checkbox" checked> LPIPS</label>
        <label><input data-quality-metric="vbench" type="checkbox"> VBench</label>
      </div>
      <button type="button" data-role="runEval">Run Eval</button>
      <div class="evalResult" data-role="evalResult"></div>
    </div>
    <div class="jobMeta" data-role="meta"></div>
    <div class="jobActions">
      <button type="button" data-role="viewLog">View Log</button>
      <button class="danger" type="button" data-role="cancelJob">Stop</button>
      <a class="linkButton" data-role="downloadLog">Download Log</a>
      <a class="linkButton" data-role="downloadResult">Download Result</a>
    </div>
  `;
  card.querySelector('[data-role="viewLog"]').addEventListener("click", () => {
    selectedJobId = job.job_id;
    loadLog();
  });
  card.querySelector('[data-role="cancelJob"]').addEventListener("click", async () => {
    await cancelJob(card.dataset.jobId);
  });
  card.querySelector('[data-role="runEval"]').addEventListener("click", async () => {
    await runEval(card.dataset.jobId);
  });
  $("jobs").appendChild(card);
  return card;
}

function updateJobCard(card, job) {
  const statusClass = job.status === "failed" ? "failed" : job.status === "running" ? "running" : "";
  card.querySelector('[data-role="requestId"]').textContent = job.request_id;
  card.querySelector('[data-role="strategy"]').textContent = job.strategy_summary || "origin";
  const status = card.querySelector('[data-role="status"]');
  status.textContent = job.status;
  status.className = `status ${statusClass}`.trim();
  updatePreview(card.querySelector('[data-role="preview"]'), job.preview);
  card.querySelector('[data-role="metrics"]').innerHTML = renderMetrics(job);
  updateEvalPanel(card, job);
  card.querySelector('[data-role="meta"]').textContent = job.output_dir || job.run_dir || "";
  const logLink = card.querySelector('[data-role="downloadLog"]');
  logLink.href = `/api/jobs/${job.job_id}/download/log`;
  logLink.download = `${job.request_id || job.job_id}.log`;
  const resultLink = card.querySelector('[data-role="downloadResult"]');
  if (job.preview?.download_url) {
    resultLink.href = job.preview.download_url;
    resultLink.download = job.preview.path || "";
    resultLink.hidden = false;
  } else {
    resultLink.removeAttribute("href");
    resultLink.hidden = true;
  }
  const cancelButton = card.querySelector('[data-role="cancelJob"]');
  const cancellable = CANCELLABLE_JOB_STATUSES.has(job.status);
  cancelButton.hidden = !cancellable;
  cancelButton.disabled = !cancellable || job.status === "cancelling";
}

function updateEvalPanel(card, job) {
  const panel = card.querySelector('[data-role="evalPanel"]');
  const canEval = Boolean(job.output_dir) && !CANCELLABLE_JOB_STATUSES.has(job.status);
  panel.hidden = !canEval;
  if (!canEval) return;
  const referenceInput = card.querySelector('[data-role="evalReference"]');
  if (!referenceInput.value && job.eval?.reference_task_id) {
    referenceInput.value = job.eval.reference_task_id;
  }
  const button = card.querySelector('[data-role="runEval"]');
  card.querySelectorAll("[data-quality-metric]").forEach((input) => {
    const metric = input.dataset.qualityMetric;
    const availability = qualityAvailability[metric] || { available: true };
    input.disabled = availability.available === false;
    input.title = availability.reason || "";
    input.closest("label").title = availability.reason || "";
    if (availability.available === false) input.checked = false;
  });
  button.disabled = job.eval?.status === "running";
  button.textContent = job.eval?.status === "running" ? "Evaluating" : "Run Eval";
  const unavailable = Object.entries(qualityAvailability)
    .filter(([_metric, availability]) => availability?.available === false)
    .map(([metric, availability]) => `${metric.toUpperCase()}: ${availability.reason}`);
  const result = renderEvalResult(job);
  card.querySelector('[data-role="evalResult"]').innerHTML = [result, ...unavailable].filter(Boolean).join("<br>");
}

function formatSeconds(ms) {
  if (typeof ms !== "number" || Number.isNaN(ms)) return "-";
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatGb(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${value.toFixed(2)}GB`;
}

function renderMetrics(job) {
  const items = [];
  const timing = job.timing;
  if (timing) {
    items.push(`E2E ${formatSeconds(timing.e2e_ms)}`);
    items.push(`DiT FW ${formatSeconds(timing.dit_forward_total_ms)}`);
  }
  const memory = job.memory;
  if (memory) {
    items.push(`Peak GPU ${formatGb(memory.peak_gpu_allocated_gb)}`);
    if (typeof memory.peak_gpu_reserved_gb === "number") {
      items.push(`Reserved ${formatGb(memory.peak_gpu_reserved_gb)}`);
    }
    if (typeof memory.peak_flexcache_gb === "number") {
      items.push(`Cache ${formatGb(memory.peak_flexcache_gb)}`);
    }
  }
  if (items.length === 0) return "";
  return `<div class="timing">${items.map((item) => `<span>${item}</span>`).join("")}</div>`;
}

function formatScore(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "";
  return Number.isFinite(value) ? value.toFixed(4) : String(value);
}

function renderEvalResult(job) {
  const state = job.eval || {};
  const quality = job.quality || state.result || {};
  const lines = [];
  if (state.status === "running") lines.push("Eval running...");
  if (state.status === "failed") lines.push(`Eval failed: ${state.error || "unknown error"}`);
  Object.entries(quality).forEach(([metric, payload]) => {
    if (!payload || typeof payload !== "object") return;
    const result = payload.result && typeof payload.result === "object" ? payload.result : payload;
    const status = payload.status || result.status || "unknown";
    const score = result.mean_score ?? result.score;
    const suffix = score === undefined ? status : `${status} ${formatScore(score)}`;
    lines.push(`${metric.toUpperCase()}: ${suffix}`);
  });
  return lines.join("<br>");
}

function renderPreview(preview) {
  if (!preview) return "";
  if (preview.kind === "video") {
    return `<video class="preview" src="${preview.url}" controls muted playsinline></video>`;
  }
  return `<img class="preview" src="${preview.url}" alt="Generated preview">`;
}

function updatePreview(host, preview) {
  const url = preview?.url || "";
  const kind = preview?.kind || "";
  if (host.dataset.url === url && host.dataset.kind === kind) return;
  host.dataset.url = url;
  host.dataset.kind = kind;
  host.innerHTML = renderPreview(preview);
}

async function refresh() {
  if (shuttingDown) return;
  const [cfg, jobsData] = await Promise.all([api("/api/config"), api("/api/jobs")]);
  renderStaticConfig(cfg);
  renderJobs(jobsData.jobs);
  if (!selectedJobId && jobsData.jobs.length > 0) selectedJobId = jobsData.jobs[0].job_id;
  if (selectedJobId) await loadLog();
}

async function shutdownService() {
  if (shuttingDown) return;
  const ok = window.confirm("Shutdown this ChituDiffusion service and release the Slurm/GPU allocation?");
  if (!ok) return;
  shuttingDown = true;
  if (refreshTimer !== null) clearInterval(refreshTimer);
  $("shutdownBtn").disabled = true;
  $("refreshBtn").disabled = true;
  $("submitBtn").disabled = true;
  $("workerStatus").textContent = "Shutdown requested. Releasing worker resources...";
  try {
    await api("/api/service/shutdown", {
      method: "POST",
      body: JSON.stringify({ reason: "requested from web UI" }),
    });
    $("workerStatus").textContent = "Shutdown started. This page may disconnect when the server exits.";
  } catch (error) {
    $("workerStatus").textContent = `Shutdown request failed: ${error.message}`;
    shuttingDown = false;
    refreshTimer = setInterval(refresh, 5000);
    $("shutdownBtn").disabled = false;
    $("refreshBtn").disabled = false;
  }
}

async function cancelJob(jobId) {
  if (!jobId) return;
  const ok = window.confirm("Stop the current generation for this job? The service will keep running.");
  if (!ok) return;
  try {
    await api(`/api/jobs/${jobId}/cancel`, {
      method: "POST",
      body: JSON.stringify({ reason: "requested from web UI" }),
    });
    selectedJobId = jobId;
    await refresh();
    pollJobUntilSettled(jobId);
  } catch (error) {
    $("log").textContent = error.message;
  }
}

async function runEval(jobId) {
  const card = document.querySelector(`.job[data-job-id="${jobId}"]`);
  if (!card) return;
  const referenceTaskId = card.querySelector('[data-role="evalReference"]').value.trim();
  const metrics = Array.from(card.querySelectorAll("[data-quality-metric]"))
    .filter((input) => input.checked && !input.disabled)
    .map((input) => input.dataset.qualityMetric);
  if (metrics.length === 0) {
    $("log").textContent = "No available eval metric selected.";
    return;
  }
  try {
    const job = await api(`/api/jobs/${jobId}/eval`, {
      method: "POST",
      body: JSON.stringify({ reference_task_id: referenceTaskId, metrics }),
    });
    selectedJobId = job.job_id;
    await refresh();
  } catch (error) {
    $("log").textContent = error.message;
  }
}

function pollJobUntilSettled(jobId) {
  let attempts = 0;
  const timer = setInterval(async () => {
    attempts += 1;
    try {
      await refresh();
      const card = document.querySelector(`.job[data-job-id="${jobId}"]`);
      const status = card?.querySelector('[data-role="status"]')?.textContent || "";
      if (!CANCELLABLE_JOB_STATUSES.has(status) || attempts >= 20) {
        clearInterval(timer);
      }
    } catch (_error) {
      if (attempts >= 20) clearInterval(timer);
    }
  }, 1000);
}

async function loadLog() {
  if (!selectedJobId) return;
  const data = await api(`/api/jobs/${selectedJobId}/log`);
  $("log").textContent = data.text || "";
  $("log").scrollTop = $("log").scrollHeight;
}

$("requestForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const request = {
    request_id: $("requestId").value.trim() || undefined,
    role: $("role").value.trim() || "user",
    prompt: $("prompt").value.trim(),
    negative_prompt: $("negativePrompt").value.trim() || undefined,
    seed: $("seed").value === "" ? undefined : Number($("seed").value),
    size: [Number($("width").value), Number($("height").value)],
    frame_num: Number($("frames").value),
    num_inference_steps: $("steps").value === "" ? undefined : Number($("steps").value),
    sample_solver: $("solver").value || undefined,
  };
  request.flexcache_strategy = $("flexcacheStrategy").value;
  const availability = flexcacheAvailability[request.flexcache_strategy] || { available: true };
  if (availability.available === false) {
    $("log").textContent = availability.reason || `${request.flexcache_strategy} is not available for this instance.`;
    return;
  }
  if (request.flexcache_strategy !== "origin") {
    request.flexcache_params = { strategy: request.flexcache_strategy };
    document.querySelectorAll("[data-flexcache-param]").forEach((input) => {
      const name = input.dataset.flexcacheParam;
      if (input.value === "") return;
      request.flexcache_params[name] = input.type === "number" ? Number(input.value) : input.value;
    });
  }
  request.output = {
    root_dir: $("outputRoot").value.trim() || "outputs",
    run_log: $("runLog").checked,
    memory: $("memory").checked,
    timer: $("timer").checked,
    log_ranks: parseLogRanks($("logRanks").value),
  };
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ request }),
    });
    selectedJobId = job.job_id;
    await refresh();
  } catch (error) {
    $("log").textContent = error.message;
  }
});

$("refreshBtn").addEventListener("click", refresh);
$("shutdownBtn").addEventListener("click", shutdownService);
$("flexcacheStrategy").addEventListener("change", () => renderFlexcacheParams({ preserveValues: false }));
refreshTimer = setInterval(refresh, 5000);
refresh();
