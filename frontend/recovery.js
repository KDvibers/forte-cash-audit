(() => {
  "use strict";
  const VERSION = 1;
  const KEY = "forte-audit:last-check:v1";
  const uuid = (value) => typeof value === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
  const plain = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

  function normalize(value) {
    if (!plain(value) || value.version !== VERSION || !uuid(value.uploadId) ||
        !(value.jobId === null || uuid(value.jobId)) || ![null, "mock", "real"].includes(value.mode) ||
        !Number.isFinite(value.startedAt) || value.startedAt < 0 || !plain(value.file) ||
        typeof value.file.name !== "string" || !value.file.name.trim() || value.file.name.length > 2048 ||
        !Number.isSafeInteger(value.file.size) || value.file.size <= 0 ||
        !Number.isSafeInteger(value.file.lastModified) || value.file.lastModified < 0) return null;
    return {
      version: VERSION, uploadId: value.uploadId, jobId: value.jobId, mode: value.mode, startedAt: value.startedAt,
      file: { name: value.file.name, size: value.file.size, lastModified: value.file.lastModified },
    };
  }

  function createStore(storageOrGetter) {
    const storage = () => typeof storageOrGetter === "function" ? storageOrGetter() : storageOrGetter;
    function clear(expectedUploadId) {
      try {
        if (expectedUploadId) {
          let existing = null;
          try { existing = normalize(JSON.parse(storage().getItem(KEY))); } catch { /* Remove corrupt records. */ }
          if (existing && existing.uploadId !== expectedUploadId) return true;
        }
        storage().removeItem(KEY);
        return true;
      } catch { return false; }
    }
    function read() {
      try {
        const raw = storage().getItem(KEY);
        if (raw === null) return null;
        if (typeof raw !== "string" || raw.length > 16384) { clear(); return null; }
        const value = normalize(JSON.parse(raw));
        if (!value) clear();
        return value;
      } catch { clear(); return null; }
    }
    function write(value) {
      const safe = normalize(value);
      if (!safe) return false;
      try {
        let existing = null;
        try { existing = normalize(JSON.parse(storage().getItem(KEY))); } catch { /* Replace corrupt records. */ }
        if (existing && existing.uploadId !== safe.uploadId && existing.startedAt >= safe.startedAt) return true;
        storage().setItem(KEY, JSON.stringify(safe));
        return true;
      } catch { return false; }
    }
    return Object.freeze({ read, write, clear });
  }

  if (typeof window !== "undefined") window.CashAuditRecovery = createStore(() => window.localStorage);
  if (typeof module !== "undefined" && module.exports) module.exports = { normalize, createStore, KEY, VERSION };
})();
