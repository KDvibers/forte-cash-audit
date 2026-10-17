(() => {
  "use strict";

  const el = (id) => document.getElementById(id);
  const input = el("video-input");
  const choose = el("choose-video");
  const replace = el("replace-video");
  const remove = el("remove-video");
  const newVideo = el("new-video");
  const retry = el("retry-action");
  const retryUpload = el("retry-upload");
  const drop = el("drop-zone");
  const progress = el("transfer-progress");
  const connectionRetry = el("connection-retry");
  const dialog = el("help-dialog");
  const openHelp = el("open-help");
  const resultPanel = el("result-live");
  const report = window.CashAuditResults;
  const recovery = window.CashAuditRecovery;
  const initialRecord = recovery?.read() || null;
  let config = null;
  let ready = false;
  let file = null;
  let sourceFileInfo = null;
  let jobMode = null;
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
  let allowAbandon = false;
  let restoreOnConnect = false;
  let connecting = false;
  let completedResult = false;

  const text = (id, value) => { el(id).textContent = value; };
  const hasFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");
  const validId = (value) => typeof value === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);

  function storageNotice(message) {
    text("recovery-notice", message);
    el("recovery-notice").hidden = !message;
  }

  function persist() {
    if (!uploadId || !sourceFileInfo) return;
    const saved = recovery?.write({ version: 1, uploadId, jobId, mode: jobMode, startedAt: beganAt, file: sourceFileInfo });
    if (!saved) storageNotice("Браузер не сохранил сведения о проверке. Работа продолжается, но после закрытия страницы восстановление может быть недоступно.");
    else storageNotice("");
  }

  function forgetSaved() {
    if (uploadId && recovery && !recovery.clear(uploadId)) {
      storageNotice("Не удалось удалить сохранённые сведения о проверке в этом браузере.");
    }
    restoreOnConnect = false;
  }

  function showMode() {
    const mode = uploadId ? jobMode : config?.mode;
    text("mode-badge", mode === "mock" ? "Тестовый режим" : mode === "real" ? "Анализ видео" : "Режим не определён");
    el("mode-badge").dataset.mode = mode || "unknown";
    el("mode-notice").hidden = mode !== "mock";
    text("mode-notice", "Эта проверка выполнена в тестовом режиме: события и кадры синтетические и не описывают загруженное видео.");
  }

  function showSelection() {
    text("file-name", sourceFileInfo?.name || "Исходная запись");
    text("file-size", sourceFileInfo ? size(sourceFileInfo.size) : "");
    el("upload-empty").hidden = true;
    el("upload-selected").hidden = false;
    drop.dataset.state = "selected";
  }

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
    newVideo.hidden = !sourceFileInfo || (active && !allowAbandon);
    newVideo.textContent = active ? "Начать новую проверку" : "Проверить другое видео";
  }

  function result(state, title, description, count = "", icon = "i-clock") {
    if (state !== "done") {
      report?.clear();
      el("open-report").hidden = true;
    }
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
    const seconds = Math.max(0, Math.floor((Date.now() - beganAt) / 1000));
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
    retryUpload.hidden = true;
    retryUpload.disabled = false;
  }

  function reset(focus = true) {
    if (active && !allowAbandon) return;
    generation += 1;
    connectionGeneration += 1;
    connecting = false;
    stopTimers();
    forgetSaved();
    active = false;
    allowAbandon = false;
    completedResult = false;
    file = uploadId = jobId = null;
    sourceFileInfo = jobMode = null;
    report?.clear();
    el("open-report").hidden = true;
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
    showMode();
    controls();
    if (focus) choose.focus();
    if (!ready) connect();
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
    if (connecting || (active && !allowAbandon && !restoreOnConnect)) return;
    const attempt = ++connectionGeneration;
    connecting = true;
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
        showMode();
        text("max-file-size", `MOV · до ${size(data.max_upload_bytes)}`);
        ready = data.analyzer_ready;
        if (restoreOnConnect && uploadId) {
          connecting = false;
          restoreOnConnect = false;
          text("connection-message", ready ? "Связь восстановлена. Проверяем последнее задание." : "Анализатор сейчас недоступен. Сохранённый результат можно получить без запуска новой проверки.");
          connectionRetry.hidden = ready;
          connectionRetry.disabled = false;
          restoreUpload();
          return;
        }
        if (ready) {
          text("connection-message", "Сервер готов. После выбора видео проверка начнётся автоматически.");
          connectionRetry.hidden = true;
          connectionRetry.disabled = false;
          connecting = false;
          controls();
          return;
        }
        text("connection-message", typeof data.message === "string" ? data.message : "Анализатор запускается. Выбор видео станет доступен, когда сервер будет готов.");
      } catch {
        if (attempt !== connectionGeneration) return;
        text("connection-message", "Нет связи с сервером. Запустите backend/start.ps1 и откройте приложение по адресу сервера, обычно http://127.0.0.1:8000. Отдельный предпросмотр интерфейса не выполняет анализ.");
        if (uploadId && !completedResult) {
          restoreOnConnect = true;
          allowAbandon = true;
          result("connection-lost", "Сохранённую проверку пока не удалось восстановить", "Восстановите связь с сервером. Видео повторно отправляться не будет.", "", "i-info");
          text("selection-note", "Можно дождаться соединения или начать новую проверку. Предыдущее задание при этом не отменяется.");
          showRetry("connect", "Восстановить связь");
        }
        break;
      }
      if (performance.now() >= until) break;
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
    } while (attempt === connectionGeneration && performance.now() < until);
    if (attempt !== connectionGeneration) return;
    connecting = false;
    connectionRetry.hidden = connectionRetry.disabled = false;
    controls();
  }

  function pauseUpload(message) {
    stopTimers();
    active = true;
    allowAbandon = true;
    restoreOnConnect = true;
    connectionRetry.hidden = connectionRetry.disabled = false;
    progress.hidden = true;
    text("transfer-label", "Подтверждение загрузки не получено");
    text("selection-state", "Проверим состояние загрузки");
    text("selection-note", "Сначала проверим, принял ли сервер видео. Новая проверка не отменит прежнюю загрузку, если она ещё обрабатывается.");
    result("connection-lost", "Связь с сервером прервалась", message, "", "i-info");
    text("upload-status", message);
    error(message);
    showRetry("recover", "Проверить состояние загрузки");
    controls();
    retry.focus();
  }

  function rejectUpload(status, data) {
    if (status >= 500 || status === 409 || status === 429 || status === 0) {
      pauseUpload(uploadError(status, data));
      return;
    }
    active = false;
    allowAbandon = false;
    forgetSaved();
    uploadId = null;
    stopTimers();
    progress.hidden = true;
    text("transfer-label", "Загрузка не принята");
    const message = uploadError(status, data);
    text("selection-state", "Видео не принято");
    text("selection-note", "Исправьте причину ошибки или выберите другое видео.");
    result("failed", "Проверка не началась", message, "", "i-info");
    text("upload-status", message);
    error(message, true);
    if (status === 401 || status === 403) {
      ready = false;
      text("connection-message", message);
      connectionRetry.hidden = connectionRetry.disabled = false;
    }
    controls();
  }

  function expireRecovery(message) {
    stopTimers();
    forgetSaved();
    uploadId = null;
    active = false;
    completedResult = false;
    allowAbandon = false;
    hideRetry();
    progress.hidden = true;
    text("selection-state", "Сохранённая проверка недоступна");
    text("selection-note", "Запись восстановления удалена из браузера. Можно начать новую проверку.");
    result("connection-lost", "Проверка больше недоступна", message, "", "i-info");
    text("upload-status", message);
    connectionRetry.hidden = connectionRetry.disabled = false;
    controls();
    newVideo.focus();
  }

  function unknownMode() {
    stopTimers();
    allowAbandon = true;
    jobMode = null;
    persist();
    showMode();
    hideRetry();
    text("selection-state", "Режим этой проверки не сохранён");
    text("selection-note", "Начните новую проверку: нельзя считать старый результат реальным, если его режим неизвестен.");
    result("connection-lost", "Режим результата неизвестен", "Сервер не сохранил, был ли этот результат тестовым или реальным. Заключение не отображается.", "", "i-info");
    controls();
    newVideo.focus();
  }

  function readJobMode(response) {
    const value = response.headers.get("X-Cash-Audit-Job-Mode");
    if (value !== null) jobMode = ["mock", "real"].includes(value) ? value : null;
    persist();
    showMode();
  }

  function startPolling(run) {
    queryFailures = 0;
    pollDeadline = performance.now() + (config.job_timeout_seconds + 30) * 1000;
    active = true;
    restoreOnConnect = false;
    allowAbandon = false;
    startElapsed();
    progress.hidden = true;
    text("transfer-label", "Видео уже принято сервером");
    text("selection-state", "Проверяем сохранённое задание");
    text("selection-note", file ? "Результат появится автоматически." : "Видео не отправляется повторно. После восстановления результата можно открыть исходный файл локально.");
    result("processing", "Получаем состояние проверки", "Запрашиваем ранее созданное задание. Повторная загрузка не нужна.");
    controls();
    pollJob(run);
  }

  function restoreUpload(resendOnMissing = false) {
    if (!uploadId || !config) return;
    const run = ++generation;
    stopTimers();
    active = true;
    restoreOnConnect = false;
    hideRetry();
    clearError();
    startElapsed();
    text("selection-state", "Восстанавливаем проверку");
    result("processing", "Ищем последнее задание", "Проверяем состояние загрузки на сервере. Видео повторно не отправляется.");
    controls();
    lookupUpload(run, 0, resendOnMissing);
  }

  async function lookupUpload(run, failures, resendOnMissing) {
    if (run !== generation || !uploadId) return;
    try {
      const { response, data } = await getJson(`/ui/recovery/${encodeURIComponent(uploadId)}`);
      if (run !== generation) return;
      if ([401, 403].includes(response.status)) {
        ready = false;
        expireRecovery("Сессия истекла. Результат недоступен в текущем браузере; подключитесь снова для новой проверки.");
        return;
      }
      if (response.status === 404) {
        if (jobId) {
          expireRecovery("Сохранённое задание не найдено или срок его хранения истёк.");
          return;
        }
        if (resendOnMissing && file && ready) { upload(); return; }
        pauseUpload("Задание пока не найдено. Загрузка могла ещё не завершиться. Можно проверить состояние ещё раз; новая проверка не отменяет предыдущую.");
        retryUpload.hidden = !(file && ready);
        return;
      }
      if (!response.ok || !data || !validId(data.job_id) || !["processing", "done", "failed"].includes(data.status) ||
          ![null, "mock", "real"].includes(data.mode) || (jobId && data.job_id !== jobId)) throw new Error("Invalid recovery response");
      jobId = data.job_id;
      jobMode = data.mode;
      persist();
      showMode();
      startPolling(run);
    } catch {
      if (run !== generation) return;
      if (failures >= 3) {
        pauseUpload("Не удаётся получить состояние загрузки. Восстановите связь и повторите запрос. Видео не отправлялось повторно.");
        restoreOnConnect = true;
        connectionRetry.hidden = connectionRetry.disabled = false;
        return;
      }
      pollTimer = window.setTimeout(() => lookupUpload(run, failures + 1, resendOnMissing), Math.min(10000, 2000 * 2 ** failures));
    }
  }

  function upload() {
    if (!file || !config || !uploadId) return;
    const run = ++generation;
    active = true;
    completedResult = false;
    allowAbandon = false;
    restoreOnConnect = false;
    persist();
    clearError();
    hideRetry();
    controls();
    startElapsed();
    progress.value = 0;
    progress.max = 100;
    progress.hidden = false;
    text("transfer-label", "Загружаем видео: 0%");
    text("selection-state", "Загружаем видео");
    text("selection-note", "После завершения загрузки проверку можно восстановить, даже если закрыть страницу.");
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
      if (!data || !validId(data.job_id)) {
        pauseUpload("Сервер ответил, но идентификатор задания не получен. Проверьте загрузку ещё раз.");
        return;
      }
      jobId = data.job_id;
      const mode = xhr.getResponseHeader("X-Cash-Audit-Job-Mode") || xhr.getResponseHeader("X-Cash-Audit-Mode");
      jobMode = ["mock", "real"].includes(mode) ? mode : null;
      persist();
      showMode();
      text("upload-status", "Видео принято. Ожидаем результат анализа.");
      startPolling(run);
    });
    const interrupted = () => {
      if (run !== generation) return;
      pauseUpload("Не удалось подтвердить загрузку. Сервер мог уже принять видео. Сначала проверьте состояние загрузки; повторная отправка автоматически не выполняется.");
    };
    xhr.addEventListener("error", interrupted);
    xhr.addEventListener("timeout", interrupted);
    xhr.addEventListener("abort", interrupted);
    try { xhr.send(body); } catch { interrupted(); }
  }

  function pauseQuery(message) {
    stopTimers();
    allowAbandon = true;
    restoreOnConnect = true;
    connectionRetry.hidden = connectionRetry.disabled = false;
    text("selection-state", "Ожидаем подтверждение результата");
    text("selection-note", "Видео повторно не загружается. Новая проверка не отменяет текущее задание.");
    result("connection-lost", "Результат пока недоступен", message, "", "i-info");
    text("upload-status", message);
    showRetry("query", "Проверить результат");
    controls();
    retry.focus();
  }

  function complete(data) {
    if (data.status === "done") {
      try {
        if (!report?.show(data, { file, mode: jobMode, sourceFileInfo, onNewVideo: () => reset() })) {
          pauseQuery("Не удалось открыть полный результат. Можно повторно запросить это же задание.");
          return;
        }
      } catch {
        report?.clear();
        pauseQuery("Не удалось отобразить результат. Можно повторно запросить это же задание.");
        return;
      }
    }
    active = false;
    completedResult = true;
    allowAbandon = false;
    restoreOnConnect = false;
    persist();
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
      const mock = jobMode === "mock";
      text("selection-state", "Проверка завершена");
      text("selection-note", file ? "Можно выбрать следующее видео." : "Результат восстановлен. Для просмотра исходного видео выберите его локально в отчёте.");
      const title = violation ? "Обнаружены признаки возможных нарушений" : "Нарушений не обнаружено";
      result("done", mock ? `Тест: ${title.toLowerCase()}` : title,
        mock ? "Это синтетический результат тестового анализатора, он не описывает содержимое загруженного видео."
          : "Анализ завершён. Изучите события, полные объяснения и кадры в результате проверки.",
        `Событий: ${data.violations.length}`, violation ? "i-info" : "i-check");
      resultPanel.dataset.verdict = data.verdict;
      el("open-report").hidden = false;
      text("open-report", violation ? "События и кадры" : "Открыть результат");
      text("upload-status", `${mock ? "Тестовая проверка" : "Проверка"} завершена. ${violation ? "Обнаружены признаки возможных нарушений." : "Нарушений не обнаружено."} Событий: ${data.violations.length}.`);
    }
    controls();
    if (data.status === "done") el("report").focus();
    else resultPanel.focus();
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
      if ([401, 403, 404].includes(response.status)) {
        if (response.status !== 404) ready = false;
        expireRecovery(response.status === 404
          ? "Задание не найдено или срок его хранения истёк. Можно начать новую проверку."
          : "Сессия истекла. Подключитесь снова, чтобы начать новую проверку.");
        return;
      }
      if (!response.ok || !data || data.job_id !== jobId) throw new Error("Invalid job response");
      readJobMode(response);
      queryFailures = 0;
      if (data.status === "failed") { complete(data); return; }
      if (data.status === "done") {
        if (!jobMode) { unknownMode(); return; }
        if (!report?.validate(data)) {
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
    sourceFileInfo = { name: file.name, size: file.size, lastModified: file.lastModified };
    jobMode = config.mode;
    uploadId = window.crypto.randomUUID();
    beganAt = Date.now();
    showSelection();
    showMode();
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
    if (kind === "connect") { restoreOnConnect = Boolean(uploadId); connect(); }
    else if (kind === "recover") restoreUpload();
    else if (kind === "query") { stopTimers(); startPolling(++generation); }
  });
  retryUpload.addEventListener("click", () => {
    if (!file || !uploadId || retryUpload.hidden || retryUpload.disabled) return;
    retryUpload.disabled = true;
    restoreUpload(true);
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
    else if (sourceFileInfo) newVideo.focus();
    else if (ready) choose.focus();
    else openHelp.focus();
  });

  if (initialRecord) {
    sourceFileInfo = initialRecord.file;
    uploadId = initialRecord.uploadId;
    jobId = initialRecord.jobId;
    jobMode = initialRecord.mode;
    beganAt = initialRecord.startedAt;
    active = true;
    restoreOnConnect = true;
    showSelection();
    text("selection-state", "Восстанавливаем последнюю проверку");
    text("selection-note", "Запросим сохранённое задание. Видео повторно не отправляется.");
    elapsed();
  }
  controls();
  connect();
})();
