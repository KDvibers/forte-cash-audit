"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { matchSourceFile } = require("../results.js");

const original = { name: "касса.MOV", size: 4096, lastModified: 1729160000000 };

test("accepts MOV with matching saved metadata, without treating metadata as a content hash", () => {
  assert.equal(matchSourceFile({ ...original }, original), null);
  assert.equal(matchSourceFile({ ...original, content: "different bytes with the same metadata" }, original), null);
});

test("rejects each changed saved field and rejects empty or non-MOV files", () => {
  for (const changed of [
    { name: "other.mov" }, { size: 4097 }, { lastModified: original.lastModified + 1 },
    { name: "recording.mp4" }, { size: 0 }, { size: NaN },
  ]) assert.equal(typeof matchSourceFile({ ...original, ...changed }, original), "string");
  assert.equal(typeof matchSourceFile(null, original), "string");
});

test("compares only known valid metadata and supports timestamps equal to zero", () => {
  assert.equal(matchSourceFile(original, { name: original.name }), null);
  assert.equal(matchSourceFile(original, null), null);
  assert.equal(matchSourceFile(original, { size: null, lastModified: null }), null);
  assert.equal(matchSourceFile({ ...original, lastModified: 0 }, { lastModified: 0 }), null);
  assert.equal(typeof matchSourceFile(original, { lastModified: 0 }), "string");
});
