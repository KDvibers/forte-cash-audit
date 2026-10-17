"use strict";

// This exercises the actual application state machine with controlled I/O.
// It is not a browser: layout, media decoding and native focus remain UI checks.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { KEY, VERSION } = require("../recovery.js");
const { validate } = require("../results.js");

const UPLOAD = "c36eb75d-dbc2-4bd7-9a7e-1fc80477d737";
const JOB = "cf088334-70f2-41d5-a74d-f953fdf73915";
const EPOCH = 1_792_200_000_000;
const frontend = path.resolve(__dirname, "..");

function saved(overrides = {}) {
  return { version: VERSION, uploadId: UPLOAD, jobId: JOB, mode: "mock", startedAt: EPOCH,
    file: { name: "neutral.mov", size: 12345, lastModified: EPOCH - 1000 }, ...overrides };
}

function bootstrap(overrides = {}) {
  return { csrf_token: "session-csrf-token-only", mode: "mock", analyzer_ready: true,
    message: "Ready", max_upload_bytes: 100 * 1024 * 1024, poll_interval_ms: 1000,
    job_timeout_seconds: 60, ...overrides };
}

function metadata(overrides = {}) {
  return { job_id: JOB, status: "processing", mode: "mock", file_name: "neutral.mov",
    file_size: 12345, created_at: EPOCH / 1000, ...overrides };
}

const completed = () => ({ job_id: JOB, status: "done", verdict: "no_violation", violations: [] });
const processing = () => ({ job_id: JOB, status: "processing" });

function response(data, status = 200, headers = {}) {
  const entries = new Map(Object.entries(headers).map(([key, value]) => [key.toLowerCase(), value]));
  return { ok: status >= 200 && status < 300, status, headers: { get: (name) => entries.get(name.toLowerCase()) ?? null },
    json: async () => data };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

class Target {
  constructor() { this.listeners = new Map(); }
  addEventListener(name, callback, options = {}) {
    const list = this.listeners.get(name) || [];
    list.push({ callback, once: options.once });
    this.listeners.set(name, list);
  }
  removeEventListener(name, callback) {
    this.listeners.set(name, (this.listeners.get(name) || []).filter((entry) => entry.callback !== callback));
  }
  emit(name, event = {}) {
    for (const entry of [...(this.listeners.get(name) || [])]) {
      entry.callback({ target: this, preventDefault() {}, ...event });
      if (entry.once) this.removeEventListener(name, entry.callback);
    }
  }
}

function makeHarness({ record = saved(), session = bootstrap(), lookup = metadata(), route, deniedStorage = false } = {}) {
  let now = 0, nextTimer = 0, pendingGets = 0, maxPendingGets = 0;
  const timers = new Map(), elements = new Map(), storage = new Map(), requests = [], posts = [], shown = [];
  if (record !== null) storage.set(KEY, typeof record === "string" ? record : JSON.stringify(record));
  class Element extends Target {
    constructor(id) {
      super();
      this.id = id; this.hidden = false; this.disabled = false; this.dataset = {};
      this.textContent = ""; this.value = ""; this.files = []; this.attributes = new Map();
      this.classList = { add() {}, remove() {}, toggle() {} };
    }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    removeAttribute(name) { this.attributes.delete(name); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    querySelector(selector) { return selector === "use" ? get(this.id + "-use") : null; }
    focus() { document.activeElement = this; }
    click() { if (!this.disabled) this.emit("click"); }
    showModal() { this.open = true; }
    close() { this.open = false; this.emit("close"); }
    scrollIntoView() {}
  }
  function get(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); }
  const html = fs.readFileSync(path.join(frontend, "index.html"), "utf8");
  for (const match of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    const node = get(match[1]);
    node.hidden = /\bhidden\b/.test(match[0]);
    node.disabled = /\bdisabled\b/.test(match[0]);
  }
  const document = Object.assign(new Target(), { getElementById: get, activeElement: null });
  function schedule(fn, delay = 0, repeat = false) {
    const id = ++nextTimer;
    timers.set(id, { fn, when: now + Math.max(0, Number(delay)), interval: repeat ? Math.max(1, delay) : 0 });
    return id;
  }
  const browser = new Target();
  browser.setTimeout = (fn, delay) => schedule(fn, delay);
  browser.setInterval = (fn, delay) => schedule(fn, delay, true);
  browser.clearTimeout = browser.clearInterval = (id) => timers.delete(id);
  browser.location = { reload() { browser.reloaded = true; } };
  browser.crypto = { randomUUID: () => "9962dc31-e78f-4395-a9b7-91413c055c04" };
  const localStorage = {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: (key) => storage.delete(key),
  };
  Object.defineProperty(browser, "localStorage", { get() {
    if (deniedStorage) throw new Error("Storage denied");
    return localStorage;
  } });
  browser.CashAuditResults = {
    validate,
    show(data, options) { shown.push({ data, options }); return true; },
    clear() {},
  };
  class XHR extends Target {
    constructor() { super(); this.upload = new Target(); this.headers = {}; }
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(name, value) { this.headers[name] = value; }
    getResponseHeader(name) { return this.responseHeaders?.[name.toLowerCase()] ?? null; }
    send(body) { this.body = body; posts.push(this); }
    abort() { this.emit("abort"); }
    respond(status, data, headers = {}) {
      this.status = status;
      this.responseText = JSON.stringify(data);
      this.responseHeaders = Object.fromEntries(Object.entries(headers).map(([name, value]) => [name.toLowerCase(), value]));
      this.emit("load");
    }
  }
  async function fetch(url, options = {}) {
    requests.push({ url, method: options.method || "GET", options });
    const isJob = url.startsWith("/ui/jobs/");
    if (isJob) { pendingGets += 1; maxPendingGets = Math.max(maxPendingGets, pendingGets); }
    try {
      const operation = Promise.resolve().then(() => {
        if (route) {
          const override = route(url, options);
          if (override !== undefined) return override;
        }
        if (url === "/ui/session") return response(session);
        if (url === `/ui/recovery/${UPLOAD}`) return response(lookup);
        if (url === `/ui/jobs/${JOB}`) return response(completed(), 200, { "X-Cash-Audit-Job-Mode": lookup.mode ?? "unknown" });
        throw new Error(`Unexpected URL: ${url}`);
      });
      const aborted = new Promise((resolve, reject) => {
        options.signal?.addEventListener("abort", () => reject(new Error("Request aborted")), { once: true });
      });
      return await Promise.race([operation, aborted]);
    } finally { if (isJob) pendingGets -= 1; }
  }
  class ClockDate extends Date { static now() { return EPOCH + now; } }
  const context = vm.createContext({ window: browser, document, fetch, XMLHttpRequest: XHR,
    FormData: class { append() {} }, performance: { now: () => now }, Date: ClockDate,
    AbortController, console, URL, Intl, localStorage, setTimeout: browser.setTimeout,
    clearTimeout: browser.clearTimeout });
  vm.runInContext(fs.readFileSync(path.join(frontend, "recovery.js"), "utf8"), context, { filename: "recovery.js" });
  vm.runInContext(fs.readFileSync(path.join(frontend, "app.js"), "utf8"), context, { filename: "app.js" });
  async function settle() { for (let i = 0; i < 30; i += 1) await Promise.resolve(); }
  async function advance(milliseconds) {
    const target = now + milliseconds;
    for (let steps = 0; steps < 500; steps += 1) {
      await settle();
      const next = [...timers].filter(([, value]) => value.when <= target).sort((a, b) => a[1].when - b[1].when)[0];
      if (!next) { now = target; await settle(); return; }
      const [id, timer] = next;
      now = timer.when;
      if (timer.interval) timer.when += timer.interval; else timers.delete(id);
      timer.fn();
    }
    throw new Error("Too many scheduled callbacks; possible polling loop");
  }
  return { get, requests, posts, shown, storage, browser, settle, advance, timers,
    maxPendingGets: () => maxPendingGets, jobRequests: () => requests.filter((entry) => entry.url.startsWith("/ui/jobs/")) };
}

test("refresh restores processing then done using GET requests only", async () => {
  let attempt = 0;
  const page = makeHarness({ route(url) {
    if (url === `/ui/jobs/${JOB}`) return response(++attempt === 1 ? processing() : completed(), 200, { "X-Cash-Audit-Job-Mode": "mock" });
  } });
  await page.settle();
  assert.equal(page.shown.length, 0);
  await page.advance(1500);
  assert.equal(page.shown.length, 1);
  assert.equal(page.shown[0].options.file, null, "Refresh must not invent a browser File");
  assert.equal(page.posts.length, 0);
  assert.ok(page.requests.every((entry) => entry.method === "GET"));
  assert.equal(page.jobRequests().length, 2);
  assert.equal(page.maxPendingGets(), 1);
});

test("stored result remains available while the current analyzer is unavailable", async () => {
  const page = makeHarness({ session: bootstrap({ analyzer_ready: false }) });
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.equal(page.posts.length, 0);
  assert.equal(page.get("choose-video").disabled, true);
  assert.equal(page.get("connection-retry").hidden, false, "Unavailable analyzer must still allow a readiness retry");
});

test("an interrupted upload with no job ID is recovered by upload ID without POST", async () => {
  const page = makeHarness({ record: saved({ jobId: null }) });
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.ok(page.requests.some((entry) => entry.url === `/ui/recovery/${UPLOAD}`));
  assert.equal(page.posts.length, 0);
});

test("network failures stop automatic polling and manual resume requests the existing job", async () => {
  let offline = true;
  const page = makeHarness({ route(url) {
    if (url === `/ui/jobs/${JOB}` && offline) throw new Error("Offline");
  } });
  await page.settle();
  assert.match(page.get("connection-message").textContent, /Не удаётся получить состояние задания/);
  assert.doesNotMatch(page.get("connection-message").textContent, /Сервер готов|Сервер отвечает/);
  await page.advance(30_000);
  const attempts = page.jobRequests().length;
  assert.ok(attempts >= 1 && attempts <= 5, "GET retries must be bounded");
  assert.equal(page.get("retry-action").hidden, false);
  assert.equal(page.get("connection-message").textContent, page.get("result-description").textContent);
  await page.advance(30_000);
  assert.equal(page.jobRequests().length, attempts);
  offline = false;
  page.get("retry-action").click();
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.match(page.get("connection-message").textContent, /Сервер отвечает/);
  assert.equal(page.get("connection-retry").hidden, true);
  assert.equal(page.posts.length, 0);
  assert.equal(page.maxPendingGets(), 1);
});

for (const status of [401, 404]) {
  test(`expired recovery (${status}) clears saved state without resubmitting`, async () => {
    const page = makeHarness({ route(url) {
      if (url === `/ui/recovery/${UPLOAD}`) return response({ detail: "Unavailable" }, status);
    } });
    await page.settle();
    assert.equal(page.storage.has(KEY), false);
    assert.equal(page.shown.length, 0);
    assert.equal(page.posts.length, 0);
    assert.equal(page.jobRequests().length, 0);
  });
  test(`expired job response (${status}) clears a previously recovered job`, async () => {
    const page = makeHarness({ route(url) {
      if (url === `/ui/jobs/${JOB}`) return response({ detail: "Unavailable" }, status);
    } });
    await page.settle();
    assert.equal(page.storage.has(KEY), false);
    assert.equal(page.shown.length, 0);
    assert.equal(page.posts.length, 0);
  });
}

test("an unknown upload is kept for manual lookup but never automatically resent after refresh", async () => {
  const page = makeHarness({ record: saved({ jobId: null }), route(url) {
    if (url === `/ui/recovery/${UPLOAD}`) return response({ detail: "Not yet found" }, 404);
  } });
  await page.settle();
  assert.equal(page.storage.has(KEY), true);
  assert.equal(page.get("retry-action").hidden, false);
  assert.equal(page.get("retry-upload").hidden, true, "No in-memory File remains after refresh");
  assert.equal(page.posts.length, 0);
});

test("invalid or inaccessible storage leaves a usable fresh page", async () => {
  for (const options of [{ record: "{bad json" }, { deniedStorage: true }]) {
    const page = makeHarness(options);
    await page.settle();
    assert.equal(page.requests.filter((entry) => entry.url.startsWith("/ui/recovery/")).length, 0);
    assert.equal(page.posts.length, 0);
    assert.equal(page.get("choose-video").disabled, false);
  }
});

test("original mock result is never relabelled real after a server mode change", async () => {
  const page = makeHarness({ session: bootstrap({ mode: "real" }), record: saved({ mode: "real" }),
    lookup: metadata({ mode: "mock", status: "done" }) });
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.equal(page.shown[0].options.mode, "mock");
  assert.equal(page.posts.length, 0);
});

test("unknown historical mode cannot be inferred from the current real server", async () => {
  const page = makeHarness({ session: bootstrap({ mode: "real" }), lookup: metadata({ mode: null }),
    record: saved({ mode: "real" }) });
  await page.settle();
  assert.equal(page.shown.length, 0);
  assert.equal(page.posts.length, 0);
});

test("job mode response header is authoritative over cached recovery metadata", async () => {
  const page = makeHarness({ session: bootstrap({ mode: "real" }), lookup: metadata({ mode: "real" }),
    route(url) {
      if (url === `/ui/jobs/${JOB}`) return response(completed(), 200, { "X-Cash-Audit-Job-Mode": "mock" });
    } });
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.equal(page.shown[0].options.mode, "mock");
});

test("explicitly unknown job mode does not reuse a cached known mode", async () => {
  const page = makeHarness({ route(url) {
    if (url === `/ui/jobs/${JOB}`) return response(completed(), 200, { "X-Cash-Audit-Job-Mode": "unknown" });
  } });
  await page.settle();
  assert.equal(page.shown.length, 0);
  assert.equal(page.posts.length, 0);
});

test("uncertain initial POST is reconciled by GET before any second upload", async () => {
  const page = makeHarness({ record: null, route(url) {
    if (url.startsWith("/ui/recovery/")) return response(metadata());
  } });
  await page.settle();
  const input = page.get("video-input");
  input.files = [{ name: "neutral.mov", size: 12345, lastModified: EPOCH - 1000 }];
  input.emit("change");
  await page.settle();
  assert.equal(page.posts.length, 1);
  assert.ok(page.storage.has(KEY), "Recovery metadata must be saved before a POST can be interrupted");
  page.posts[0].emit("timeout");
  await page.settle();
  if (!page.get("retry-action").hidden) page.get("retry-action").click();
  await page.settle();
  assert.equal(page.posts.length, 1, "The upload timeout must not trigger another POST");
  assert.ok(page.requests.some((entry) => entry.url.startsWith("/ui/recovery/")));
  assert.equal(page.shown.length, 1);
});

test("manual lookup permits one pending GET and ignores its response after reset", async () => {
  let pending = null, online = false;
  const page = makeHarness({ route(url) {
    if (url === `/ui/recovery/${UPLOAD}`) {
      if (!online) throw new Error("Offline");
      return pending.promise;
    }
  } });
  await page.settle();
  await page.advance(30_000);
  online = true;
  pending = deferred();
  const attempts = page.requests.filter((entry) => entry.url.startsWith("/ui/recovery/")).length;
  page.get("retry-action").click();
  await page.settle();
  page.get("retry-action").click();
  await page.settle();
  assert.equal(page.requests.filter((entry) => entry.url.startsWith("/ui/recovery/")).length, attempts + 1);
  assert.equal(page.get("new-video").hidden, false, "An uncertain check can be explicitly abandoned");
  page.get("new-video").click();
  await page.settle();
  pending.resolve(response(metadata()));
  await page.settle();
  assert.equal(page.shown.length, 0, "A superseded request must not restore an abandoned result");
  assert.equal(page.storage.has(KEY), false);
  assert.equal(page.posts.length, 0);
  assert.equal(page.jobRequests().length, 0, "The abandoned lookup must not start a job poll");
});

for (const status of [429, 503]) {
  test(`POST ${status} requires an explicit retry and a second lookup before resending the same upload`, async () => {
    let lookups = 0;
    const confirmation = deferred();
    const page = makeHarness({ record: null, route(url) {
      if (url.startsWith("/ui/recovery/")) {
        lookups += 1;
        return lookups === 1 ? response({ detail: "Not found" }, 404) : confirmation.promise;
      }
    } });
    await page.settle();
    const input = page.get("video-input");
    input.files = [{ name: "neutral.mov", size: 12345, lastModified: EPOCH - 1000 }];
    input.emit("change");
    await page.settle();
    assert.equal(page.posts.length, 1);
    const uploadId = page.posts[0].headers["X-Upload-ID"];
    page.posts[0].respond(status, { detail: "Server busy" });
    await page.settle();
    assert.equal(page.get("result-live").dataset.state, "connection-lost");
    assert.equal(page.get("connection-message").textContent, "Server busy");
    assert.doesNotMatch(page.get("result-title").textContent, /Связь.*прервалась/);
    assert.equal(page.get("retry-action").hidden, false);
    assert.equal(page.get("retry-upload").hidden, true);
    assert.equal(page.posts.length, 1);

    page.get("retry-action").click();
    await page.settle();
    assert.equal(lookups, 1);
    assert.equal(page.posts.length, 1, "A missing lookup is not permission for an automatic POST");
    assert.equal(page.get("retry-upload").hidden, false);
    assert.match(page.get("connection-message").textContent, /Задание пока не найдено/);
    assert.doesNotMatch(page.get("connection-message").textContent, /Нет связи|Связь.*прервалась/);

    page.get("retry-upload").click();
    page.get("retry-upload").click();
    await page.settle();
    assert.equal(lookups, 2, "Rapid clicks must start only one confirmation lookup");
    assert.equal(page.posts.length, 1, "Resend must wait for the second lookup response");
    confirmation.resolve(response({ detail: "Still not found" }, 404));
    await page.settle();
    assert.equal(page.posts.length, 2);
    assert.equal(page.posts[1].headers["X-Upload-ID"], uploadId);
    assert.equal(JSON.parse(page.storage.get(KEY)).uploadId, uploadId);
    page.get("retry-upload").click();
    await page.settle();
    assert.equal(page.posts.length, 2, "A hidden retry must not submit a third POST");

    page.posts[1].respond(200, { job_id: JOB }, { "X-Cash-Audit-Job-Mode": "mock" });
    await page.settle();
    assert.equal(page.shown.length, 1);
    assert.equal(page.shown[0].options.mode, "mock");
  });
}

test("refresh restores a failed check as a technical failure, never as no violations", async () => {
  const page = makeHarness({ lookup: metadata({ status: "failed" }), route(url) {
    if (url === `/ui/jobs/${JOB}`) return response({ job_id: JOB, status: "failed" }, 200,
      { "X-Cash-Audit-Job-Mode": "mock" });
  } });
  await page.settle();
  assert.equal(page.shown.length, 0, "Failed jobs must not render a completed verdict report");
  assert.equal(page.get("result-live").dataset.state, "failed");
  assert.equal(page.get("result-live").dataset.verdict, undefined);
  assert.equal(page.get("result-count").textContent, "");
  assert.match(page.get("result-description").textContent, /не сформировано/);
  assert.doesNotMatch(page.get("result-title").textContent, /не обнаружено|no_violation/i);
  assert.equal(page.posts.length, 0);
});

test("starting a new check after an unavailable worker reconnects and enables file selection", async () => {
  let sessions = 0;
  const page = makeHarness({ route(url) {
    if (url === "/ui/session") return response(bootstrap({ analyzer_ready: ++sessions > 1 }));
  } });
  await page.settle();
  assert.equal(page.shown.length, 1, "Stored results do not require an available worker");
  assert.equal(page.get("choose-video").disabled, true);
  assert.equal(sessions, 1);
  page.shown[0].options.onNewVideo();
  await page.settle();
  assert.equal(sessions, 2, "Reset must reconnect instead of keeping stale worker readiness");
  assert.equal(page.get("choose-video").disabled, false);
  assert.equal(page.get("upload-empty").hidden, false);
  assert.equal(page.get("result-live").hidden, true);
  assert.equal(page.storage.has(KEY), false);
  assert.equal(page.posts.length, 0);
});

test("choosing MOV starts one automatic upload and ignores a second selection while it is active", async () => {
  const page = makeHarness({ record: null });
  await page.settle();
  const input = page.get("video-input");
  const file = { name: "neutral.mov", size: 12345, lastModified: EPOCH - 1000 };
  input.files = [file];
  input.emit("change");
  await page.settle();
  assert.equal(page.posts.length, 1, "Choosing the file must start the upload without another button");
  assert.equal(page.posts[0].method, "POST");
  assert.equal(page.posts[0].url, "/ui/analyze");
  assert.equal(page.posts[0].headers["X-CSRF-Token"], bootstrap().csrf_token);
  assert.equal(page.posts[0].headers["X-API-Key"], undefined, "The shared API key does not belong in the browser");
  assert.equal(page.get("choose-video").disabled, true);
  assert.equal(page.get("drop-zone").getAttribute("aria-busy"), "true");

  input.files = [{ ...file, name: "second.mov" }];
  input.emit("change");
  await page.settle();
  assert.equal(page.posts.length, 1);
  page.posts[0].respond(200, { job_id: JOB }, { "X-Cash-Audit-Job-Mode": "mock" });
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.equal(page.shown[0].options.file, file, "The result must retain the file actually submitted");
  assert.equal(page.get("result-live").dataset.state, "done");
  assert.equal(page.get("new-video").hidden, false);
});

for (const [label, files, message] of [
  ["non-MOV", [{ name: "neutral.mp4", size: 12345 }], /формате MOV/],
  ["empty", [{ name: "empty.mov", size: 0 }], /пустой/],
  ["oversized", [{ name: "large.mov", size: 100 * 1024 * 1024 + 1 }], /меньшего размера/],
  ["multiple", [{ name: "one.mov", size: 1 }, { name: "two.mov", size: 1 }], /одно видео/],
]) {
  test(`${label} selection explains the problem before any upload and allows a valid file next`, async () => {
    const page = makeHarness({ record: null });
    await page.settle();
    const input = page.get("video-input");
    input.files = files;
    input.emit("change");
    await page.settle();
    assert.equal(page.posts.length, 0);
    assert.equal(page.storage.has(KEY), false);
    assert.equal(page.get("upload-error").hidden, false);
    assert.match(page.get("upload-error").textContent, message);
    assert.equal(page.get("choose-video").disabled, false);

    input.files = [{ name: "neutral.mov", size: 12345, lastModified: EPOCH - 1000 }];
    input.emit("change");
    await page.settle();
    assert.equal(page.posts.length, 1);
    assert.equal(page.get("upload-error").hidden, true);
  });
}

test("a rejected damaged MOV shows the server explanation and permits a new check", async () => {
  const page = makeHarness({ record: null });
  await page.settle();
  const input = page.get("video-input");
  input.files = [{ name: "damaged.mov", size: 12345, lastModified: EPOCH - 1000 }];
  input.emit("change");
  await page.settle();
  page.posts[0].respond(422, { detail: "Не удалось прочитать видео. Выберите корректный MOV с видеодорожкой." });
  await page.settle();
  assert.equal(page.shown.length, 0);
  assert.equal(page.storage.has(KEY), false, "A rejected file must not be restored as an accepted job");
  assert.equal(page.get("result-live").dataset.state, "failed");
  assert.match(page.get("upload-error").textContent, /корректный MOV с видеодорожкой/);
  assert.equal(page.get("connection-message").textContent, page.get("upload-error").textContent);
  assert.equal(page.get("new-video").hidden, false);

  page.get("new-video").click();
  await page.settle();
  assert.equal(page.get("upload-empty").hidden, false);
  assert.equal(page.get("choose-video").disabled, false);
  assert.match(page.get("connection-message").textContent, /Выберите видео/);
  assert.doesNotMatch(page.get("connection-message").textContent, /Не удалось прочитать/);
  input.files = [{ name: "neutral.mov", size: 54321, lastModified: EPOCH }];
  input.emit("change");
  await page.settle();
  assert.equal(page.posts.length, 2);
  page.posts[1].respond(200, { job_id: JOB }, { "X-Cash-Audit-Job-Mode": "mock" });
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.equal(page.shown[0].options.file.name, "neutral.mov");
});

test("an incomplete completed response never becomes a verdict and can be retrieved again without uploading", async () => {
  let malformed = true;
  const page = makeHarness({ route(url) {
    if (url === `/ui/jobs/${JOB}`) return response(malformed
      ? { job_id: JOB, status: "done", verdict: "violation", violations: [] }
      : completed(), 200, { "X-Cash-Audit-Job-Mode": "mock" });
  } });
  await page.settle();
  assert.equal(page.shown.length, 0);
  assert.equal(page.get("result-live").dataset.state, "connection-lost");
  assert.equal(page.get("result-live").dataset.verdict, undefined);
  assert.match(page.get("result-description").textContent, /неполный результат/);
  assert.equal(page.get("connection-message").textContent, page.get("result-description").textContent);
  const attempts = page.jobRequests().length;
  await page.advance(30_000);
  assert.equal(page.jobRequests().length, attempts, "A malformed final result must not trigger an endless polling loop");
  malformed = false;
  page.get("retry-action").click();
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.match(page.get("connection-message").textContent, /Сервер отвечает/);
  assert.equal(page.posts.length, 0);
  assert.ok(page.jobRequests().every((entry) => entry.url === `/ui/jobs/${JOB}`));
});

test("exceeding the UI wait limit pauses without a verdict and resumes only the existing job", async () => {
  let done = false;
  const page = makeHarness({ session: bootstrap({ job_timeout_seconds: 1 }), route(url) {
    if (url === `/ui/jobs/${JOB}`) return response(done ? completed() : processing(), 200,
      { "X-Cash-Audit-Job-Mode": "mock" });
  } });
  await page.settle();
  await page.advance(35_000);
  assert.equal(page.shown.length, 0);
  assert.equal(page.get("result-live").dataset.state, "connection-lost");
  assert.equal(page.get("result-live").dataset.verdict, undefined);
  assert.match(page.get("result-description").textContent, /Время ожидания/);
  const attempts = page.jobRequests().length;
  await page.advance(30_000);
  assert.equal(page.jobRequests().length, attempts);
  done = true;
  page.get("retry-action").click();
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.equal(page.posts.length, 0);
  assert.equal(page.maxPendingGets(), 1);
  assert.ok(page.jobRequests().every((entry) => entry.url === `/ui/jobs/${JOB}`));
});

test("an unavailable server gives a visitor an actionable retry without redirecting them to localhost", async () => {
  let online = false;
  const page = makeHarness({ record: null, route(url) {
    if (url === "/ui/session" && !online) throw new Error("Offline");
  } });
  await page.settle();
  assert.equal(page.get("choose-video").disabled, true);
  assert.equal(page.get("connection-retry").hidden, false);
  assert.match(page.get("connection-message").textContent, /Подключиться снова/);
  assert.doesNotMatch(page.get("connection-message").textContent, /127\.0\.0\.1|localhost|backend\/start\.ps1/);
  online = true;
  page.get("connection-retry").click();
  await page.settle();
  assert.equal(page.get("choose-video").disabled, false);
  assert.equal(page.get("connection-retry").hidden, true);
  assert.equal(page.posts.length, 0);
});

test("failed upload recovery updates the connection banner and successful GET recovery clears the stale error", async () => {
  let offline = true;
  const page = makeHarness({ record: saved({ jobId: null }), route(url) {
    if (url === `/ui/recovery/${UPLOAD}` && offline) throw new Error("Offline");
  } });
  await page.settle();
  assert.match(page.get("connection-message").textContent, /Не удаётся получить состояние загрузки/);
  assert.doesNotMatch(page.get("connection-message").textContent, /Сервер готов|Связь восстановлена/);
  await page.advance(30_000);
  assert.equal(page.get("result-live").dataset.state, "connection-lost");
  assert.equal(page.get("connection-message").textContent, page.get("result-description").textContent);
  assert.equal(page.get("retry-action").hidden, false);

  offline = false;
  page.get("retry-action").click();
  await page.settle();
  assert.equal(page.shown.length, 1);
  assert.match(page.get("connection-message").textContent, /Сервер отвечает/);
  assert.doesNotMatch(page.get("connection-message").textContent, /Не удаётся|Не удалось/);
  assert.equal(page.get("connection-retry").hidden, true);
  assert.equal(page.posts.length, 0, "Recovery must query the original upload without resending video");
});
