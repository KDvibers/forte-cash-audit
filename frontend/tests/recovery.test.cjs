"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { normalize, createStore, KEY, VERSION } = require("../recovery.js");

function record(overrides = {}) {
  return {
    version: VERSION,
    uploadId: "c36eb75d-dbc2-4bd7-9a7e-1fc80477d737",
    jobId: "cf088334-70f2-41d5-a74d-f953fdf73915",
    mode: "mock", startedAt: 1_792_200_000_000,
    file: { name: "neutral.mov", size: 12345, lastModified: 1_792_100_000_000 },
    ...overrides,
  };
}

function memoryStorage() {
  const values = new Map();
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

test("stores only recovery metadata, never result bodies, credentials or file contents", () => {
  const storage = memoryStorage();
  const store = createStore(storage);
  const original = record();
  const input = {
    ...original, apiKey: "do-not-store-key", csrf_token: "do-not-store-csrf",
    violations: [{ image_base64: "do-not-store-image" }],
    file: { ...original.file, contents: "do-not-store-video" },
  };
  assert.equal(store.write(input), true);
  assert.deepEqual(JSON.parse(storage.getItem(KEY)), original);
  assert.deepEqual(store.read(), original);
  assert.doesNotMatch(storage.getItem(KEY), /do-not-store/);
  assert.deepEqual(input.violations, [{ image_base64: "do-not-store-image" }]);
});

test("supports an upload without a known job and preserves unknown historical mode", () => {
  const input = record({ jobId: null, mode: null });
  assert.deepEqual(normalize(input), input);
  assert.equal(normalize(input).mode, null);
  const storage = memoryStorage();
  const store = createStore(() => storage);
  assert.equal(store.write(input), true);
  assert.deepEqual(store.read(), input);
});

test("rejects corrupt schema and unsafe IDs before constructing recovery URLs", () => {
  const cases = [null, [], "record", {}, record({ version: VERSION + 1 }),
    record({ uploadId: "../../api/v1/analyze" }), record({ uploadId: "javascript:alert(1)" }),
    record({ uploadId: "------------------------------------" }), record({ jobId: "not-a-uuid" }),
    record({ mode: "pretend-real" }), record({ startedAt: NaN }), record({ startedAt: -1 }),
    record({ startedAt: "1792200000000" }), record({ file: null }),
  ];
  for (const input of cases) assert.equal(normalize(input), null, JSON.stringify(input));
});

test("validates filename and file metadata without coercing numbers", () => {
  for (const overrides of [
    { name: "" }, { name: "x".repeat(2049) }, { name: null },
    { size: 0 }, { size: -1 }, { size: "123" }, { size: Infinity },
    { size: Number.MAX_SAFE_INTEGER + 1 }, { size: 1.25 },
    { lastModified: -1 }, { lastModified: "0" }, { lastModified: NaN },
  ]) {
    assert.equal(normalize(record({ file: { ...record().file, ...overrides } })), null);
  }
  assert.equal(normalize(record({ file: { name: "<video>.mov", size: 1, lastModified: 0 } })).file.name, "<video>.mov");
});

test("corrupt saved state is removed without touching unrelated local storage", () => {
  for (const raw of ["{broken", "null", "[]", "42", JSON.stringify(record({ mode: "bad" }))]) {
    const storage = memoryStorage();
    storage.setItem("unrelated-setting", "keep");
    storage.setItem(KEY, raw);
    const store = createStore(storage);
    assert.equal(store.read(), null);
    assert.equal(storage.getItem(KEY), null);
    assert.equal(storage.getItem("unrelated-setting"), "keep");
  }
});

test("blocked storage getters and methods do not break the page", () => {
  const denied = () => { throw new Error("Storage denied"); };
  for (const input of [denied, { getItem: denied, setItem: denied, removeItem: denied }]) {
    const store = createStore(input);
    assert.equal(store.read(), null);
    assert.equal(store.write(record()), false);
    assert.equal(store.clear(), false);
  }
});

test("clearing a completed check removes only its own record", () => {
  const storage = memoryStorage();
  const store = createStore(storage);
  storage.setItem("unrelated-setting", "keep");
  assert.equal(store.write(record()), true);
  assert.equal(store.clear(), true);
  assert.equal(store.read(), null);
  assert.equal(storage.getItem("unrelated-setting"), "keep");
});

test("late writes and cleanup from another tab preserve a newer upload", () => {
  const storage = memoryStorage();
  const olderTab = createStore(storage);
  const newerTab = createStore(storage);
  const older = record();
  const newer = record({ uploadId: "9962dc31-e78f-4395-a9b7-91413c055c04", startedAt: older.startedAt + 1000 });
  assert.equal(olderTab.write(older), true);
  assert.equal(newerTab.write(newer), true);
  assert.equal(olderTab.write({ ...older, jobId: null }), true);
  assert.deepEqual(newerTab.read(), newer);
  assert.equal(olderTab.clear(older.uploadId), true);
  assert.deepEqual(newerTab.read(), newer);
  assert.equal(newerTab.clear(newer.uploadId), true);
  assert.equal(newerTab.read(), null);
});
