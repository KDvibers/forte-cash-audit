(() => {
  "use strict";

  const MAX_IMAGE_LENGTH = 8 * 1024 * 1024;
  const MAX_TOTAL_IMAGE_BYTES = 30 * 1024 * 1024;
  let current = null;

  const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  const validTime = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
  const validText = (value) => typeof value === "string" && value.trim().length > 0 && value.length <= 20000;

  function jpegBase64(value) {
    if (typeof value !== "string" || value.length < 8 || value.length > MAX_IMAGE_LENGTH ||
        value.length % 4 !== 0 || !value.startsWith("/9j/") || /[^A-Za-z0-9+/=]/.test(value)) return false;
    const padding = value.indexOf("=");
    return padding === -1 || (padding === value.length - 1) ||
      (padding === value.length - 2 && value.endsWith("=="));
  }

  function validate(data) {
    if (!record(data) || data.status !== "done" || !["violation", "no_violation"].includes(data.verdict) ||
        !Array.isArray(data.violations) || data.violations.length > 100 ||
        (data.verdict === "violation") !== (data.violations.length > 0)) return false;
    let totalImageBytes = 0;
    for (const event of data.violations) {
      if (!record(event) || !validText(event.type) || !validText(event.description) ||
          !validTime(event.timecode_sec) || !Array.isArray(event.evidence_frames) ||
          event.evidence_frames.length < 1 || event.evidence_frames.length > 3) return false;
      for (const frame of event.evidence_frames) {
        if (!record(frame) || !validTime(frame.timestamp_sec) || !jpegBase64(frame.image_base64)) return false;
        const padding = frame.image_base64.endsWith("==") ? 2 : frame.image_base64.endsWith("=") ? 1 : 0;
        totalImageBytes += frame.image_base64.length * 3 / 4 - padding;
        if (totalImageBytes > MAX_TOTAL_IMAGE_BYTES) return false;
      }
    }
    return true;
  }

  function formatTimestamp(seconds) {
    if (!validTime(seconds)) return "—";
    const millis = Math.round(seconds * 1000);
    if (!Number.isSafeInteger(millis)) {
      return `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 3 }).format(seconds)} с`;
    }
    const hours = Math.floor(millis / 3600000);
    const minutes = Math.floor(millis / 60000) % 60;
    const wholeSeconds = Math.floor(millis / 1000) % 60;
    const fraction = String(millis % 1000).padStart(3, "0");
    const clock = `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${fraction}`;
    return hours ? `${String(hours).padStart(2, "0")}:${clock}` : clock;
  }

  function chronologicalEvents(events) {
    return events.map((event, index) => ({ event, index }))
      .sort((a, b) => a.event.timecode_sec - b.event.timecode_sec || a.index - b.index)
      .map(({ event }) => event);
  }

  function element(tag, className = "", value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function button(className, label, action) {
    const node = element("button", className, label);
    node.type = "button";
    node.addEventListener("click", action);
    return node;
  }

  function live(state) {
    return current === state && !state.disposed;
  }

  function clear() {
    const previous = current;
    current = null;
    if (previous) {
      previous.disposed = true;
      window.clearTimeout(previous.mediaTimer);
      if (previous.dialog?.open) previous.dialog.close();
      if (previous.video) {
        previous.video.pause();
        previous.video.removeAttribute("src");
        previous.video.load();
      }
      if (previous.videoUrl) URL.revokeObjectURL(previous.videoUrl);
      previous.report.querySelectorAll("img").forEach((image) => image.removeAttribute("src"));
    }
    const report = previous?.report || (typeof document !== "undefined" ? document.getElementById("report") : null);
    if (report) {
      report.replaceChildren();
      report.hidden = true;
      delete report.dataset.verdict;
    }
  }

  function feedback(state, message) {
    if (live(state)) state.videoFeedback.textContent = message;
  }

  function finishSeek(state) {
    if (!live(state) || state.requestedSeek === null || !state.video) return;
    const target = state.requestedSeek;
    state.requestedSeek = null;
    const actual = state.video.currentTime;
    if (!validTime(actual)) {
      feedback(state, "Не удалось определить позицию видео. Таймкоды и кадры остаются доступны ниже.");
      return;
    }
    const playback = state.video.paused ? "Видео на паузе." : "Воспроизведение продолжается.";
    feedback(state, Math.abs(actual - target) > 0.25
      ? `Браузер открыл ближайшую доступную позицию ${formatTimestamp(actual)} вместо ${formatTimestamp(target)}. ${playback}`
      : `Видео на отметке ${formatTimestamp(actual)}. ${playback}`);
  }

  function seek(state, seconds) {
    if (!live(state) || !validTime(seconds)) return;
    if (!state.video || state.mediaFailed) {
      feedback(state, "Переход недоступен: браузер не открыл исходное видео. Таймкоды и кадры можно изучить отдельно.");
      return;
    }
    state.video.pause();
    if (state.video.readyState < 1) {
      state.pendingSeek = seconds;
      feedback(state, `Ожидаем сведения о видео для перехода к ${formatTimestamp(seconds)}. Воспроизведение не начнётся автоматически.`);
      return;
    }
    state.pendingSeek = null;
    const duration = state.video.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      feedback(state, "Браузер не определил длительность видео. Переход по таймкоду недоступен; кадры и описание сохранены.");
      return;
    }
    if (seconds > duration) {
      state.requestedSeek = null;
      feedback(state, `Отметка ${formatTimestamp(seconds)} выходит за длительность видео ${formatTimestamp(duration)}. Переход к этой отметке не выполнен.`);
      return;
    }
    try {
      state.requestedSeek = seconds;
      state.video.currentTime = seconds;
      feedback(state, `Переходим к ${formatTimestamp(seconds)}…`);
      if (!state.video.seeking) finishSeek(state);
    } catch {
      state.requestedSeek = null;
      feedback(state, "Браузер не смог перейти к этой отметке. Таймкод, описание и кадры остаются доступными.");
    }
  }

  function timeButton(state, seconds, prefix = "") {
    const node = button("report-time-button", `${prefix}${formatTimestamp(seconds)}`, () => {
      if (!live(state)) return;
      seek(state, seconds);
      state.videoSection.scrollIntoView({ block: "center" });
    });
    node.setAttribute("aria-label", `Перейти к отметке ${formatTimestamp(seconds)} в исходном видео`);
    return node;
  }

  function makeVideo(state, file) {
    const section = element("section", "report-video-review");
    state.videoSection = section;
    const heading = element("h3", "", "Исходное видео");
    heading.id = "report-video-heading";
    section.setAttribute("aria-labelledby", heading.id);
    section.append(heading);
    const note = element("p", "report-video-note", "Нажмите на событие или таймкод кадра, чтобы перейти к этому моменту. Воспроизведение запускается только кнопкой плеера.");
    state.videoFeedback = element("p", "report-video-feedback");
    state.videoFeedback.setAttribute("role", "status");
    state.videoFeedback.setAttribute("aria-live", "polite");
    if (file) {
      try {
        const video = element("video", "report-video");
        state.video = video;
        video.controls = true;
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.setAttribute("aria-label", "Исходное видео для проверки таймкодов");
        video.addEventListener("loadedmetadata", () => {
          if (!live(state)) return;
          window.clearTimeout(state.mediaTimer);
          if (state.pendingSeek !== null) seek(state, state.pendingSeek);
          else feedback(state, Number.isFinite(video.duration) && video.duration > 0
            ? `Длительность: ${formatTimestamp(video.duration)}. Выберите таймкод для перехода.`
            : "Браузер не определил длительность записи. Кадры и описание можно изучить без плеера.");
        });
        video.addEventListener("seeked", () => finishSeek(state));
        video.addEventListener("error", () => {
          if (!live(state)) return;
          state.mediaFailed = true;
          state.pendingSeek = state.requestedSeek = null;
          window.clearTimeout(state.mediaTimer);
          video.hidden = true;
          feedback(state, "Браузер не смог воспроизвести MOV: его кодек может не поддерживаться. Результат анализа, описание и кадры доступны независимо от плеера.");
        });
        state.videoUrl = URL.createObjectURL(file);
        video.src = state.videoUrl;
        section.append(video);
        state.videoFeedback.textContent = "Открываем исходное видео в браузере…";
        state.mediaTimer = window.setTimeout(() => {
          if (live(state) && video.readyState < 1 && !state.mediaFailed) {
            feedback(state, "Браузер пока не открыл видео. Если кодек MOV не поддерживается, изучите описание и кадры: они не зависят от плеера.");
          }
        }, 12000);
      } catch {
        state.mediaFailed = true;
        state.video = null;
        state.videoFeedback.textContent = "Не удалось открыть исходное видео в браузере. Описание и кадры результата доступны ниже.";
      }
    } else {
      state.videoFeedback.textContent = "Исходный файл недоступен на этой странице. Можно изучить описание событий и кадры с таймкодами.";
    }
    section.append(note, state.videoFeedback);
    return section;
  }

  function makeDialog(state) {
    const dialog = element("dialog", "report-lightbox");
    state.dialog = dialog;
    const header = element("header", "report-lightbox-header");
    const title = element("h2", "", "Кадр события");
    title.id = "report-lightbox-title";
    dialog.setAttribute("aria-labelledby", title.id);
    const close = button("report-dialog-close icon-button", "×", () => dialog.close());
    close.setAttribute("aria-label", "Закрыть увеличенный кадр");
    header.append(title, close);
    const media = element("div", "report-lightbox-media");
    const caption = element("p", "report-lightbox-caption");
    caption.id = "report-lightbox-caption";
    caption.setAttribute("aria-live", "polite");
    dialog.setAttribute("aria-describedby", caption.id);
    const actions = element("div", "report-lightbox-actions");
    const previous = button("button button-secondary", "Предыдущий кадр", () => renderDialogFrame(state, state.dialogIndex - 1));
    const next = button("button button-secondary", "Следующий кадр", () => renderDialogFrame(state, state.dialogIndex + 1));
    actions.append(previous, next);
    dialog.append(header);
    if (state.mode === "mock") dialog.append(element("p", "report-mode-notice", "Тестовый режим: кадр синтетический и не является доказательством из загруженного видео."));
    dialog.append(media, caption, actions);
    state.dialogParts = { title, media, caption, previous, next, close };
    dialog.addEventListener("close", () => {
      if (!live(state)) return;
      if (state.dialogOpener?.isConnected && !state.dialogOpener.disabled) state.dialogOpener.focus();
      else state.report.focus();
    });
    dialog.addEventListener("keydown", (event) => {
      if (!dialog.open || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        renderDialogFrame(state, state.dialogIndex + (event.key === "ArrowLeft" ? -1 : 1));
      }
    });
    return dialog;
  }

  function renderDialogFrame(state, index) {
    if (!live(state) || index < 0 || index >= state.frames.length) return;
    state.dialogIndex = index;
    const frame = state.frames[index];
    const parts = state.dialogParts;
    parts.title.textContent = `Кадр ${index + 1} из ${state.frames.length}`;
    parts.caption.textContent = `${state.events[state.selectedIndex].type} · ${formatTimestamp(frame.timestamp_sec)}`;
    parts.previous.disabled = index === 0;
    parts.next.disabled = index === state.frames.length - 1;
    parts.previous.hidden = parts.next.hidden = state.frames.length === 1;
    parts.media.querySelectorAll("img").forEach((old) => old.removeAttribute("src"));
    const image = element("img", "report-lightbox-image");
    image.alt = `Кадр события на отметке ${formatTimestamp(frame.timestamp_sec)}`;
    const failure = element("p", "report-lightbox-error", state.frames.length > 1
      ? "Браузер не смог показать этот JPEG. Можно открыть соседний кадр; таймкод и описание события сохранены."
      : "Браузер не смог показать этот JPEG. Таймкод и описание события сохранены.");
    failure.hidden = true;
    failure.setAttribute("role", "status");
    image.addEventListener("error", () => {
      if (!live(state) || !image.isConnected) return;
      image.hidden = true;
      failure.hidden = false;
    });
    parts.media.replaceChildren(image, failure);
    image.src = `data:image/jpeg;base64,${frame.image_base64}`;
  }

  function openFrame(state, index, opener) {
    if (!live(state)) return;
    state.dialogOpener = opener;
    renderDialogFrame(state, index);
    state.dialog.showModal();
    state.dialogParts.close.focus();
  }

  function selectEvent(state, index, jump) {
    if (!live(state)) return;
    state.selectedIndex = index;
    const event = state.events[index];
    state.frames = event.evidence_frames.map((frame, order) => ({ frame, order }))
      .sort((a, b) => a.frame.timestamp_sec - b.frame.timestamp_sec || a.order - b.order)
      .map(({ frame }) => frame);
    state.eventButtons.forEach((node, i) => {
      if (i === index) node.setAttribute("aria-current", "true");
      else node.removeAttribute("aria-current");
    });
    state.details.querySelectorAll("img").forEach((image) => image.removeAttribute("src"));
    state.details.replaceChildren();
    const article = element("article", "report-event-detail");
    const heading = element("h3", "report-event-title", event.type);
    heading.id = "report-event-title";
    article.setAttribute("aria-labelledby", heading.id);
    const description = element("p", "report-description", event.description);
    article.append(heading, timeButton(state, event.timecode_sec, "Событие · "), description);
    const evidence = element("section", "report-evidence");
    const evidenceTitle = element("h4", "", `Кадры события · ${state.frames.length}`);
    evidenceTitle.id = "report-evidence-title";
    evidence.setAttribute("aria-labelledby", evidenceTitle.id);
    const grid = element("div", "report-frame-grid");
    state.frames.forEach((frame, frameIndex) => {
      const figure = element("figure", "report-frame");
      const open = button("report-frame-open", "", () => openFrame(state, frameIndex, open));
      open.setAttribute("aria-label", `Увеличить кадр ${frameIndex + 1} на отметке ${formatTimestamp(frame.timestamp_sec)}`);
      open.setAttribute("aria-haspopup", "dialog");
      const image = element("img", "report-frame-image");
      image.alt = `Кадр ${frameIndex + 1} события «${event.type}» на отметке ${formatTimestamp(frame.timestamp_sec)}`;
      image.loading = "lazy";
      const imageError = element("p", "report-frame-error", "JPEG не удалось показать. Описание и таймкод остаются доступными.");
      imageError.hidden = true;
      imageError.setAttribute("role", "status");
      image.addEventListener("error", () => {
        if (!live(state) || !image.isConnected) return;
        image.hidden = true;
        open.hidden = true;
        imageError.hidden = false;
      });
      open.append(image);
      const caption = element("figcaption");
      caption.append(element("span", "report-frame-label", `Кадр ${frameIndex + 1}`), timeButton(state, frame.timestamp_sec));
      figure.append(open, imageError, caption);
      grid.append(figure);
      image.src = `data:image/jpeg;base64,${frame.image_base64}`;
    });
    evidence.append(evidenceTitle, grid);
    state.details.append(article, evidence);
    state.selectionStatus.textContent = `Выбрано событие ${index + 1} из ${state.events.length}: ${event.type}, ${formatTimestamp(event.timecode_sec)}.`;
    if (jump) seek(state, event.timecode_sec);
  }

  function show(data, { file = null, mode = "real", onNewVideo } = {}) {
    clear();
    if (!validate(data) || !["mock", "real"].includes(mode) || typeof document === "undefined") return false;
    const report = document.getElementById("report");
    if (!report) return false;
    const state = {
      report, mode, events: chronologicalEvents(data.violations), eventButtons: [], frames: [], selectedIndex: 0,
      video: null, videoUrl: null, mediaTimer: null, pendingSeek: null, requestedSeek: null,
      mediaFailed: false, dialog: null, disposed: false,
    };
    current = state;
    report.dataset.verdict = data.verdict;
    const header = element("header", "report-header");
    const group = element("div", "report-heading-group");
    const heading = element("h2", "", "Результат проверки");
    heading.id = "report-heading";
    group.append(element("p", "report-eyebrow", "Подробный отчёт"), heading,
      element("p", "report-filename", file && typeof file.name === "string" ? file.name : "Исходная запись"));
    const actions = element("div", "report-header-actions");
    const verdict = element("div", "report-verdict");
    verdict.dataset.verdict = data.verdict;
    const verdictText = data.verdict === "violation" ? "Обнаружены признаки возможных нарушений" : "Нарушений не обнаружено";
    verdict.append(element("strong", "", mode === "mock" ? `Тест: ${verdictText.toLowerCase()}` : verdictText),
      element("span", "report-total", `Событий: ${state.events.length}`));
    actions.append(verdict);
    if (typeof onNewVideo === "function") actions.append(button("report-new-video button button-secondary", "Проверить другое видео", () => {
      if (live(state)) onNewVideo();
    }));
    header.append(group, actions);
    report.append(header);
    if (mode === "mock") report.append(element("p", "report-mode-notice", "Тестовый режим. События и кадры синтетические и не описывают содержимое загруженного видео."));
    const layout = element("div", "report-layout");
    const main = element("div", "report-main");
    state.details = element("div", "report-event-content");
    state.details.id = "report-event-content";
    state.selectionStatus = element("p", "sr-only");
    state.selectionStatus.setAttribute("role", "status");
    state.selectionStatus.setAttribute("aria-live", "polite");
    if (state.events.length) {
      const sidebar = element("aside", "report-sidebar");
      const title = element("h3", "report-list-title", "События по времени");
      title.id = "report-list-title";
      sidebar.setAttribute("aria-labelledby", title.id);
      const list = element("ol", "report-event-list");
      state.events.forEach((event, index) => {
        const item = element("li");
        const select = button("report-event-button", "", () => selectEvent(state, index, true));
        select.setAttribute("aria-controls", state.details.id);
        select.append(element("span", "report-event-number", String(index + 1).padStart(2, "0")),
          element("span", "report-event-name", event.type),
          element("span", "report-event-meta", `${formatTimestamp(event.timecode_sec)} · кадров: ${event.evidence_frames.length}`));
        state.eventButtons.push(select);
        item.append(select);
        list.append(item);
      });
      sidebar.append(title, list);
      layout.append(sidebar);
      main.append(state.details, state.selectionStatus);
    } else {
      layout.classList.add("report-layout-clean");
      const clean = element("div", "report-clean-state");
      clean.append(element("h3", "", mode === "mock" ? "Тестовый анализатор не вернул событий" : "Анализатор не вернул событий"),
        element("p", "", mode === "mock" ? "Это синтетический ответ. Список событий и кадры для него отсутствуют."
          : "Для этого видео анализатор не указал признаков нарушений. Список событий и кадры отсутствуют."));
      main.append(clean);
    }
    main.append(makeVideo(state, file));
    layout.append(main);
    report.append(layout, makeDialog(state));
    if (state.events.length) selectEvent(state, 0, false);
    report.hidden = false;
    return true;
  }

  const api = Object.freeze({ validate, show, clear });
  if (typeof window !== "undefined") window.CashAuditResults = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { ...api, formatTimestamp, chronologicalEvents };
  }
})();
