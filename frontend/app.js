(() => {
  "use strict";

  const el = (id) => document.getElementById(id);
  const input = el("video-input");
  const choose = el("choose-video");
  const replace = el("replace-video");
  const remove = el("remove-video");
  const newVideo = el("new-video");
  const retry = el("retry-action");
  const drop = el("drop-zone");
  const progress = el("transfer-progress");
  const connectionRetry = el("connection-retry");
  const dialog = el("help-dialog");
  const openHelp = el("open-help");
  const resultPanel = el("result-live");
  let config = null;
  let ready = false;
  let file = null;
  let uploadId = null;
  let jobId = null;
  let active = false;
  let retryKind = null;
  let generation = 0;
  let connectionGeneration = 0;
  let dragDepth = 0;
  let pollTimer = null;
  let elapsedTimer = null;
  let beganAt = 0;
  let pollDeadline = 0;
  let queryFailures = 0;
  let dialogReturnTarget = openHelp;

  const text = (id, value) => { el(id).textContent = value; };
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");

  function clearDrag() {
    dragDepth = 0;
    drop.classList.remove("is-dragging");
  }

  function clearError() {
    el("upload-error").hidden = true;
    text("upload-error", "");
  }

  function error(message, focus = false) {
    text("upload-error", message);
    el("upload-error").hidden = false;
    if (focus) el("upload-error").focus();
  }

  function size(bytes) {
    if (bytes < 1024) return `${bytes} Б`;
    const unit = bytes < 1024 * 1024 ? "КиБ" : "МиБ";
    const value = bytes / (unit === "КиБ" ? 1024 : 1024 * 1024);
    return `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 }).format(value)} ${unit}`;
  }

  function controls() {
    const disabled = !ready || active;
    input.disabled = choose.disabled = replace.disabled = disabled;
    remove.disabled = active;
    drop.setAttribute("aria-disabled", String(disabled));
    drop.setAttribute("aria-busy", String(active));
    replace.hidden = remove.hidden = true;
    newVideo.hidden = active || !file;
  }

  function result(state, title, description, count = "", icon = "i-clock") {
    el("result-empty").hidden = true;
    resultPanel.hidden = false;
    resultPanel.dataset.state = state;
    delete resultPanel.dataset.verdict;
    text("result-title", title);
    text("result-description", description);
    text("result-count", count);
    el("result-count").hidden = !count;
    text("result-job-id", jobId ? `Задание: ${jobId}` : "");
    const use = el("result-icon").querySelector("use");
    if (use) use.setAttribute("href", `#${icon}`);
    text("result-state", { processing: "Проверяем", done: "Готово", failed: "Не завершено", "connection-lost": "Нужно действие" }[state]);
  }

  function stopTimers() {
    window.clearTimeout(pollTimer);
    window.clearInterval(elapsedTimer);
    pollTimer = elapsedTimer = null;
  }

  function elapsed() {
    const seconds = Math.max(0, Math.floor((performance.now() - beganAt) / 1000));
    text("elapsed-time", `С начала загрузки: ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`);
  }

  function startElapsed() {
    window.clearInterval(elapsedTimer);
    elapsed();
    elapsedTimer = window.setInterval(elapsed, 1000);
  }

  function showRetry(kind, label) {
    retryKind = kind;
    retry.textContent = label;
    retry.hidden = retry.disabled = false;
  }

  function hideRetry() {
    retryKind = null;
    retry.hidden = true;
    retry.disabled = false;
  }

  function reset(focus = true) {
    if (active) return;
    generation += 1;
    stopTimers();
    file = uploadId = jobId = null;
    input.value = "";
    text("file-name", "");
    text("file-size", "");
    el("upload-selected").hidden = true;
    el("upload-empty").hidden = el("result-empty").hidden = false;
    resultPanel.hidden = true;
    text("result-state", "Ожидает видео");
    drop.dataset.state = "idle";
    progress.hidden = true;
    text("transfer-label", "");
    text("elapsed-time", "");
    hideRetry();
    clearError();
    clearDrag();
    text("upload-status", "Можно выбрать новое видео. Проверка начнётся автоматически.");
    controls();
    if (focus) choose.focus();
  }

  function uploadError(status, data) {
    const messages = {
      400: "Сервер не смог прочитать загрузку. Попробуйте выбрать файл ещё раз.",
      401: "Сессия недействительна. Обновите страницу перед новой загрузкой.",
      403: "Сессия недействительна. Обновите страницу перед новой загрузкой.",
      409: "Эта загрузка уже принимается сервером. Немного подождите и повторите проверку загрузки.",
      413: `Видео превышает лимит ${size(config.max_upload_bytes)}. Выберите файл меньше.`,
      415: "Нужен файл MOV. Выберите видео с расширением .mov.",
      422: "Не удалось прочитать видео. Выберите корректный MOV с видеодорожкой.",
      429: "Сервер занят. Подождите немного и повторите загрузку.",
      503: "Анализатор сейчас недоступен. Повторите загрузку после восстановления сервера.",
    };
    return typeof data?.detail === "string" && data.detail.length <= 500 ? data.detail
      : messages[status] || "Сервер не принял видео. Попробуйте позже.";
  }

  async function getJson(path, timeout = 12000) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, { credentials: "same-origin", cache: "no-store", signal: controller.signal, headers: { Accept: "application/json" } });
      let data = null;
      try { data = await response.json(); } catch { /* A standalone preview returns HTML, not the API. */ }
      return { response, data };
    } finally {
      window.clearTimeout(timer);
    }
  }

  function validSession(data) {
    return data && typeof data.csrf_token === "string" && data.csrf_token.length > 0 &&
      ["real", "mock"].includes(data.mode) && typeof data.analyzer_ready === "boolean" &&
      Number.isFinite(data.max_upload_bytes) && data.max_upload_bytes > 0 &&
      Number.isFinite(data.job_timeout_seconds) && data.job_timeout_seconds > 0;
  }

  async function connect() {
    if (active) return;
    const attempt = ++connectionGeneration;
    ready = false;
    connectionRetry.disabled = connectionRetry.hidden = true;
    text("connection-message", "Подключаемся к серверу…");
    text("mode-badge", "Подключение");
    controls();
    const until = performance.now() + 30000;
    do {
      try {
        const { response, data } = await getJson("/ui/session", 8000);
        if (attempt !== connectionGeneration) return;
        if (!response.ok || !validSession(data)) throw new Error("Unavailable UI API");
        config = data;
        text("mode-badge", data.mode === "mock" ? "Тестовый режим" : "Анализ видео");
        el("mode-badge").dataset.mode = data.mode;
        el("mode-notice").hidden = data.mode !== "mock";
        text("mode-notice", "Тестовый режим: сервер возвращает синтетический результат. Содержимое вашего видео модель пока не анализирует.");
        text("max-file-size", `MOV · до ${size(data.max_upload_bytes)}`);
        ready = data.analyzer_ready;
        if (ready) {
          text("connection-message", "Сервер готов. После выбора видео проверка начнётся автоматически.");
          connectionRetry.hidden = true;
          connectionRetry.disabled = false;
          controls();
          return;
        }
        text("connection-message", typeof data.message === "string" ? data.message : "Анализатор запускается. Выбор видео станет доступен, когда сервер будет готов.");
      } catch {
        if (attempt !== connectionGeneration) return;
        text("connection-message", "Нет связи с сервером. Запустите backend/start.ps1 и откройте приложение по адресу сервера, обычно http://127.0.0.1:8000. Отдельный предпросмотр интерфейса не выполняет анализ.");
        break;
      }
      if (performance.now() >= until) break;
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
    } while (attempt === connectionGeneration && performance.now() < until);
    if (attempt !== connectionGeneration) return;
    connectionRetry.hidden = connectionRetry.disabled = false;
    controls();
  }

  function pauseUpload(message) {
    stopTimers();
    progress.hidden = true;
    text("transfer-label", "Подтверждение загрузки не получено");
    text("selection-state", "Проверим состояние загрузки");
    text("selection-note", "Нажмите ниже, чтобы повторно подтвердить загрузку.");
    result("connection-lost", "Связь с сервером прервалась", message, "", "i-info");
    text("upload-status", message);
    error(message);
    showRetry("upload", "Проверить загрузку ещё раз");
    controls();
    retry.focus();
  }

  function rejectUpload(status, data) {
    if (status >= 500 || status === 409 || status === 0) {
      pauseUpload(uploadError(status, data));
      return;
    }
    active = false;
    stopTimers();
    progress.hidden = true;
    text("transfer-label", "Загрузка не принята");
    const message = uploadError(status, data);
    text("selection-state", "Видео не принято");
    text("selection-note", "Исправьте причину ошибки или выберите другое видео.");
    result("failed", "Проверка не началась", message, "", "i-info");
    text("upload-status", message);
    error(message, true);
    if (status === 429) showRetry("upload", "Повторить загрузку");
    if (status === 401 || status === 403) {
      ready = false;
      text("connection-message", message);
      connectionRetry.hidden = connectionRetry.disabled = false;
    }
    controls();
  }

  function serverModeChanged(mode) {
    if (!mode || mode === config.mode) return false;
    generation += 1;
    stopTimers();
    ready = false;
    active = true;
    progress.hidden = true;
    clearError();
    const message = "Режим работы сервера изменился. Перезагрузите страницу перед новой проверкой, чтобы результат был показан с правильной отметкой режима.";
    text("mode-badge", "Режим изменился");
    el("mode-badge").dataset.mode = "unknown";
    el("mode-notice").hidden = true;
    text("connection-message", message);
    text("selection-state", "Проверка приостановлена");
    text("selection-note", "Перезагрузите страницу, чтобы начать новую проверку.");
    text("transfer-label", "Режим сервера изменился");
    result("connection-lost", "Режим сервера изменился", "Не показываем заключение, поскольку режим сервера изменился во время проверки.", "", "i-info");
    text("upload-status", message);
    showRetry("reload", "Перезагрузить страницу");
    controls();
    retry.focus();
    return true;
  }

  function upload() {
    if (!file || !config || !uploadId) return;
    const run = ++generation;
    active = true;
    clearError();
    hideRetry();
    controls();
    startElapsed();
    progress.value = 0;
    progress.max = 100;
    progress.hidden = false;
    text("transfer-label", "Загружаем видео: 0%");
    text("selection-state", "Загружаем видео");
    text("selection-note", "Оставьте эту страницу открытой до получения результата.");
    result("processing", "Передаём видео на сервер", "После загрузки сервер проверит файл и автоматически начнёт анализ.");
    text("upload-status", "Началась загрузка видео.");
    const body = new FormData();
    body.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/ui/analyze");
    xhr.withCredentials = true;
    xhr.timeout = 180000;
    xhr.setRequestHeader("X-CSRF-Token", config.csrf_token);
    xhr.setRequestHeader("X-Upload-ID", uploadId);
    xhr.setRequestHeader("Accept", "application/json");
    function validating() {
      if (run !== generation) return;
      progress.value = 100;
      text("transfer-label", "Видео передано: 100%. Сервер проверяет файл…");
      text("selection-state", "Проверяем файл");
      result("processing", "Видео передано", "Сервер проверяет формат и видеодорожку. Это может занять несколько секунд.");
    }
    xhr.upload.addEventListener("progress", (event) => {
      if (run !== generation) return;
      if (event.lengthComputable) {
        const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
        progress.value = percent;
        text("transfer-label", `Загружаем видео: ${percent}%`);
        if (event.loaded === event.total) validating();
      } else {
        progress.removeAttribute("value");
        text("transfer-label", "Загружаем видео…");
      }
    });
    xhr.upload.addEventListener("load", validating);
    xhr.addEventListener("load", () => {
      if (run !== generation) return;
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* Never display an HTML error response. */ }
      if (xhr.status < 200 || xhr.status >= 300) {
        rejectUpload(xhr.status, data);
        return;
      }
      if (serverModeChanged(xhr.getResponseHeader("X-Cash-Audit-Mode"))) return;
      if (!data || typeof data.job_id !== "string" || !/^[0-9a-f-]{36}$/i.test(data.job_id)) {
        pauseUpload("Сервер ответил, но идентификатор задания не получен. Проверьте загрузку ещё раз.");
        return;
      }
      jobId = data.job_id;
      queryFailures = 0;
      pollDeadline = performance.now() + (config.job_timeout_seconds + 30) * 1000;
      progress.hidden = true;
      text("transfer-label", "Видео загружено");
      text("selection-state", "Идёт проверка");
      text("selection-note", "Результат появится автоматически. Дополнительную кнопку нажимать не нужно.");
      result("processing", "Проверяем кассовые операции", "Видео находится в очереди или уже анализируется. Страница обновит результат автоматически.");
      text("upload-status", "Видео принято. Ожидаем результат анализа.");
      pollJob(run);
    });
    const interrupted = () => {
      if (run !== generation) return;
      pauseUpload("Не удалось подтвердить загрузку. Сервер мог уже принять видео. Нажмите «Проверить загрузку ещё раз», чтобы продолжить с тем же идентификатором.");
    };
    xhr.addEventListener("error", interrupted);
    xhr.addEventListener("timeout", interrupted);
    xhr.addEventListener("abort", interrupted);
    xhr.send(body);
  }

  function pauseQuery(message) {
    stopTimers();
    text("selection-state", "Ожидаем подтверждение результата");
    text("selection-note", "Видео повторно не загружается. Проверим уже созданное задание.");
    result("connection-lost", "Результат пока недоступен", message, "", "i-info");
    text("upload-status", message);
    showRetry("query", "Проверить результат");
    controls();
    retry.focus();
  }

  function complete(data) {
    active = false;
    stopTimers();
    elapsed();
    hideRetry();
    clearError();
    if (data.status === "failed") {
      text("selection-state", "Анализ не завершён");
      text("selection-note", "Можно попробовать новое видео. Техническая ошибка не является заключением о нарушениях.");
      result("failed", "Не удалось завершить анализ", "Сервер сообщил об ошибке обработки. Заключение по видео не сформировано.", "", "i-info");
      text("upload-status", "Анализ завершился ошибкой. Заключение по видео не сформировано.");
    } else {
      const violation = data.verdict === "violation";
      const mock = config.mode === "mock";
      text("selection-state", "Проверка завершена");
      text("selection-note", "Можно выбрать следующее видео.");
      const title = violation ? "Обнаружены нарушения" : "Нарушений не обнаружено";
      result("done", mock ? `Тест: ${title.toLowerCase()}` : title,
        mock ? "Это синтетический результат тестового анализатора, он не описывает содержимое загруженного видео."
          : "Анализ завершён. Ниже показано количество событий, которые вернул анализатор.",
        `Событий: ${data.violations.length}`, violation ? "i-info" : "i-check");
      resultPanel.dataset.verdict = data.verdict;
      text("upload-status", `${mock ? "Тестовая проверка" : "Проверка"} завершена. ${violation ? "Обнаружены нарушения." : "Нарушений не обнаружено."} Событий: ${data.violations.length}.`);
    }
    controls();
    resultPanel.focus();
  }

  async function pollJob(run) {
    if (run !== generation || !active || !jobId) return;
    if (performance.now() >= pollDeadline) {
      pauseQuery("Время ожидания подтверждения истекло. Это не заключение по видео. Нажмите «Проверить результат», чтобы запросить состояние этого же задания.");
      return;
    }
    try {
      const { response, data } = await getJson(`/ui/jobs/${encodeURIComponent(jobId)}`);
      if (run !== generation) return;
      if (response.ok && serverModeChanged(response.headers.get("X-Cash-Audit-Mode"))) return;
      if ([401, 403, 404].includes(response.status)) {
        pauseQuery(response.status === 404
          ? "Задание не найдено в текущей сессии или срок его хранения истёк. Обновите страницу, чтобы начать новую проверку."
          : "Сессия недействительна. Обновите страницу для новой проверки. Текущий результат получить не удалось.");
        showRetry("reload", "Начать новую проверку");
        return;
      }
      if (!response.ok || !data || data.job_id !== jobId) throw new Error("Invalid job response");
      queryFailures = 0;
      if (data.status === "failed") { complete(data); return; }
      if (data.status === "done") {
        if (!["violation", "no_violation"].includes(data.verdict) || !Array.isArray(data.violations) ||
            (data.verdict === "no_violation" && data.violations.length !== 0) ||
            (data.verdict === "violation" && data.violations.length === 0)) {
          pauseQuery("Сервер вернул неполный результат. Не делаем вывод по видео; можно повторить запрос этого же задания.");
          return;
        }
        complete(data);
        return;
      }
      if (data.status !== "processing") throw new Error("Unknown job status");
      result("processing", "Проверяем кассовые операции", "Видео находится в очереди или уже анализируется. Результат появится автоматически.");
      const interval = Number.isFinite(config.poll_interval_ms) ? config.poll_interval_ms : 2000;
      pollTimer = window.setTimeout(() => pollJob(run), Math.max(1000, Math.min(10000, interval)));
    } catch {
      if (run !== generation) return;
      queryFailures += 1;
      if (queryFailures >= 4 || performance.now() >= pollDeadline) {
        pauseQuery("Не удаётся связаться с сервером. Видео уже принято; повторно отправлять его не нужно. Восстановите соединение и нажмите «Проверить результат».");
        return;
      }
      result("processing", "Восстанавливаем связь", "Видео уже на сервере. Повторяем только запрос состояния задания.");
      pollTimer = window.setTimeout(() => pollJob(run), Math.min(10000, 2000 * 2 ** (queryFailures - 1)));
    }
  }

  function acceptFiles(files) {
    clearDrag();
    if (!ready || active) return;
    if (files.length !== 1) { error("Выберите одно видео. Несколько файлов одновременно добавить нельзя.", true); return; }
    const candidate = files[0];
    if (!/\.mov$/i.test(candidate.name)) { error("Нужен файл в формате MOV. Выберите видео с расширением .mov.", true); return; }
    if (candidate.size === 0) { error("Этот файл пустой. Выберите другое видео.", true); return; }
    if (candidate.size > config.max_upload_bytes) { error(`Видео больше ${size(config.max_upload_bytes)}. Выберите файл меньшего размера.`, true); return; }
    if (!window.crypto?.randomUUID) { error("Для безопасной загрузки откройте приложение через localhost или HTTPS.", true); return; }
    reset(false);
    file = candidate;
    uploadId = window.crypto.randomUUID();
    beganAt = performance.now();
    text("file-name", file.name);
    text("file-size", size(file.size));
    el("upload-empty").hidden = true;
    el("upload-selected").hidden = false;
    drop.dataset.state = "selected";
    upload();
  }

  function picker() {
    if (!ready || active) return;
    input.value = "";
    input.click();
  }

  choose.addEventListener("click", picker);
  replace.addEventListener("click", picker);
  input.addEventListener("change", () => { if (input.files.length > 0) acceptFiles(input.files); });
  remove.addEventListener("click", () => reset());
  newVideo.addEventListener("click", () => reset());
  connectionRetry.addEventListener("click", connect);
  retry.addEventListener("click", () => {
    const kind = retryKind;
    if (!kind) return;
    hideRetry();
    clearError();
    if (kind === "reload") window.location.reload();
    else if (kind === "upload") upload();
    else {
      queryFailures = 0;
      pollDeadline = performance.now() + (config.job_timeout_seconds + 30) * 1000;
      startElapsed();
      result("processing", "Запрашиваем результат", "Проверяем ранее созданное задание. Видео повторно не отправляется.");
      pollJob(generation);
    }
  });

  document.addEventListener("dragover", (event) => { if (hasFiles(event)) event.preventDefault(); });
  document.addEventListener("drop", (event) => { if (hasFiles(event)) event.preventDefault(); clearDrag(); });
  window.addEventListener("dragend", clearDrag);
  window.addEventListener("blur", clearDrag);
  drop.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    if (!ready || active) return;
    dragDepth += 1;
    drop.classList.add("is-dragging");
  });
  drop.addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = ready && !active ? "copy" : "none";
  });
  drop.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) drop.classList.remove("is-dragging");
  });
  drop.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    acceptFiles(event.dataTransfer.files);
  });

  openHelp.addEventListener("click", () => { dialogReturnTarget = openHelp; dialog.showModal(); });
  el("close-help").addEventListener("click", () => dialog.close());
  el("dialog-upload").addEventListener("click", () => { dialogReturnTarget = choose; dialog.close(); });
  dialog.addEventListener("close", () => {
    if (dialogReturnTarget !== choose) { dialogReturnTarget.focus(); return; }
    if (active) resultPanel.focus();
    else if (file && ready) newVideo.focus();
    else if (ready) choose.focus();
    else openHelp.focus();
  });

  controls();
  connect();
})();
