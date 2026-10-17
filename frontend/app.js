(() => {
  "use strict";

  const maxFileBytes = 100 * 1024 * 1024;
  const videoInput = document.querySelector("#video-input");
  const chooseVideo = document.querySelector("#choose-video");
  const dropZone = document.querySelector("#drop-zone");
  const uploadEmpty = document.querySelector("#upload-empty");
  const uploadSelected = document.querySelector("#upload-selected");
  const fileName = document.querySelector("#file-name");
  const fileSize = document.querySelector("#file-size");
  const uploadError = document.querySelector("#upload-error");
  const uploadStatus = document.querySelector("#upload-status");
  const helpDialog = document.querySelector("#help-dialog");
  const openHelp = document.querySelector("#open-help");

  let selectedFile = null;
  let dragDepth = 0;
  let dialogReturnTarget = openHelp;

  const hasFiles = (event) =>
    Array.from(event.dataTransfer?.types || []).includes("Files");

  function clearDragState() {
    dragDepth = 0;
    dropZone.classList.remove("is-dragging");
  }

  function clearError() {
    uploadError.hidden = true;
    uploadError.textContent = "";
  }

  function showError(message) {
    uploadError.textContent = message;
    uploadError.hidden = false;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} Б`;
    const units = bytes < 1024 * 1024 ? "КиБ" : "МиБ";
    const value = bytes / (units === "КиБ" ? 1024 : 1024 * 1024);
    return `${new Intl.NumberFormat("ru-RU", {
      maximumFractionDigits: 1,
    }).format(value)} ${units}`;
  }

  function acceptFiles(files, { moveFocus = false } = {}) {
    clearDragState();
    if (files.length !== 1) {
      showError("Выберите одно видео. Несколько файлов одновременно добавить нельзя.");
      return;
    }

    const file = files[0];
    if (!/\.mov$/i.test(file.name)) {
      showError("Нужен файл в формате MOV. Выберите видео с расширением .mov.");
      return;
    }
    if (file.size === 0) {
      showError("Этот файл пустой. Выберите другое видео.");
      return;
    }
    if (file.size > maxFileBytes) {
      showError("Видео больше 100 МиБ. Выберите файл меньшего размера.");
      return;
    }

    selectedFile = file;
    clearError();
    fileName.textContent = selectedFile.name;
    fileSize.textContent = formatSize(selectedFile.size);
    uploadEmpty.hidden = true;
    uploadSelected.hidden = false;
    dropZone.dataset.state = "selected";
    uploadStatus.textContent = `Видео «${selectedFile.name}» выбрано. Файл не отправлен на сервер.`;
    if (moveFocus) document.querySelector("#replace-video").focus();
  }

  function openFilePicker() {
    // Allow selecting the same file again after removing it or fixing an error.
    videoInput.value = "";
    videoInput.click();
  }

  chooseVideo.addEventListener("click", openFilePicker);
  document.querySelector("#replace-video").addEventListener("click", openFilePicker);
  videoInput.addEventListener("change", () => {
    // Cancelling the picker keeps the previously accepted file.
    if (videoInput.files.length > 0) acceptFiles(videoInput.files, { moveFocus: true });
  });

  document.querySelector("#remove-video").addEventListener("click", () => {
    selectedFile = null;
    videoInput.value = "";
    fileName.textContent = "";
    fileSize.textContent = "";
    uploadSelected.hidden = true;
    uploadEmpty.hidden = false;
    dropZone.dataset.state = "idle";
    clearError();
    clearDragState();
    uploadStatus.textContent = "Видео удалено из выбора. Можно выбрать другой файл.";
    chooseVideo.focus();
  });

  // Dropping a file anywhere on the page must not navigate away from the app.
  document.addEventListener("dragover", (event) => {
    if (hasFiles(event)) event.preventDefault();
  });
  document.addEventListener("drop", (event) => {
    if (hasFiles(event)) event.preventDefault();
    clearDragState();
  });
  window.addEventListener("dragend", clearDragState);
  window.addEventListener("blur", clearDragState);

  dropZone.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    dropZone.classList.add("is-dragging");
  });
  dropZone.addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  dropZone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) dropZone.classList.remove("is-dragging");
  });
  dropZone.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    acceptFiles(event.dataTransfer.files);
  });

  openHelp.addEventListener("click", () => {
    dialogReturnTarget = openHelp;
    helpDialog.showModal();
  });
  document.querySelector("#close-help").addEventListener("click", () => helpDialog.close());
  document.querySelector("#dialog-upload").addEventListener("click", () => {
    dialogReturnTarget = chooseVideo;
    helpDialog.close();
  });
  helpDialog.addEventListener("close", () => {
    // The empty picker button is hidden while a video is selected.
    if (dialogReturnTarget === chooseVideo && selectedFile) {
      document.querySelector("#replace-video").focus();
    } else {
      dialogReturnTarget.focus();
    }
  });
})();
