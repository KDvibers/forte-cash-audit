"use strict";

// Run from the repository root: node --test frontend/tests/results.test.cjs
// These are contract/formatting tests. Actual decoding, DOM rendering and media
// cleanup require a separate browser check; these tests do not simulate one.
const test = require("node:test");
const assert = require("node:assert/strict");
const { validate, formatTimestamp, chronologicalEvents } = require("../results.js");

// A neutral 1 x 1 JPEG generated with Pillow; no bank footage is used.
const JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q==";

function frame(timestamp = 1.5, image = JPEG) {
  return { timestamp_sec: timestamp, image_base64: image };
}

function event(overrides = {}) {
  return {
    type: "Событие с неизвестным клиенту названием",
    timecode_sec: 1.5,
    description: "Наблюдение. Объяснение. Ссылка на правило из ответа анализатора.",
    evidence_frames: [frame()],
    ...overrides,
  };
}

function done(events = [event()]) {
  return {
    job_id: "bd3f5834-d63a-4ca6-91fe-33936868eed3", status: "done",
    verdict: events.length ? "violation" : "no_violation", violations: events,
  };
}

function freeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

test("accepts positive and negative completed results without a closed type list", () => {
  assert.equal(validate(done()), true);
  assert.equal(validate(done([])), true);
  assert.equal(validate(done([event({ type: "новый_тип_из_анализатора_2030" })])), true);
  assert.equal(validate(done([event({ evidence_frames: [frame(0), frame(1.5), frame(3600.125)] })])), true);
});

test("validating a result preserves the original explanation, type and order", () => {
  const explanation = "Полный текст\nс нормативной ссылкой и деталями; " + "контекст ".repeat(200);
  const literalMarkup = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
  const payload = freeze(done([
    event({ type: literalMarkup, description: explanation, timecode_sec: 15 }),
    event({ timecode_sec: 2.25 }),
  ]));
  const original = JSON.stringify(payload);
  // Validation must not erase model text or attempt to interpret it as HTML.
  // The renderer's textContent safety still requires a separate DOM review.
  assert.equal(validate(payload), true);
  assert.equal(JSON.stringify(payload), original);
  assert.equal(payload.violations[0].type, literalMarkup);
  assert.equal(payload.violations[0].description, explanation);
});

test("rejects incomplete, contradictory and non-terminal result envelopes", () => {
  const cases = [
    null, [], {}, "done", { status: "processing" }, { status: "failed" },
    { ...done(), status: "processing" }, { ...done(), status: "failed" },
    { ...done(), verdict: "unknown" }, { ...done(), verdict: "no_violation" },
    { ...done([]), verdict: "violation" }, { ...done(), violations: null },
    { ...done(), violations: {} }, { ...done(), violations: [null] },
  ];
  for (const payload of cases) {
    assert.equal(validate(payload), false, `Unexpectedly accepted ${JSON.stringify(payload)}`);
  }
});

test("rejects missing or blank event descriptions and labels", () => {
  for (const field of ["type", "description"]) {
    for (const value of [undefined, null, "", " \n\t ", 42, {}, "x".repeat(20_001)]) {
      assert.equal(validate(done([event({ [field]: value })])), false, `${field}: ${typeof value}`);
    }
  }
});

test("accepts fractional timestamps but rejects coerced, negative and nonfinite times", () => {
  for (const timestamp of [0, 0.125, 59.999, 3600.125, Number.MAX_VALUE]) {
    assert.equal(validate(done([event({ timecode_sec: timestamp, evidence_frames: [frame(timestamp)] })])), true);
  }
  for (const timestamp of [undefined, null, "1.5", true, false, -0.001, NaN, Infinity, -Infinity]) {
    assert.equal(validate(done([event({ timecode_sec: timestamp })])), false, `event ${String(timestamp)}`);
    const badFrame = { timestamp_sec: timestamp, image_base64: JPEG };
    assert.equal(validate(done([event({ evidence_frames: [badFrame] })])), false, `frame ${String(timestamp)}`);
  }
});

test("requires between one and three evidence frames for every event", () => {
  for (const frames of [undefined, null, {}, [], [null], [frame(), frame(), frame(), frame()]]) {
    assert.equal(validate(done([event({ evidence_frames: frames })])), false);
  }
});

test("accepts JPEG base64 only, with no data URI or HTML payload", () => {
  const invalidImages = [undefined, null, "", true, 42,
    `data:image/jpeg;base64,${JPEG}`, "<img src=x onerror=alert(1)>",
    "iVBORw0KGgoAAAANSUhEUg==", `${JPEG}\n`, "/9j/===", "/9j/!!!!", "/9j/2Q=",
  ];
  for (const image of invalidImages) {
    const badFrame = { timestamp_sec: 1.5, image_base64: image };
    assert.equal(validate(done([event({ evidence_frames: [badFrame] })])), false, String(image));
  }
});

test("rejects more than 100 events instead of silently truncating evidence", () => {
  assert.equal(validate(done(Array.from({ length: 100 }, (_, i) => event({ timecode_sec: i })))), true);
  assert.equal(validate(done(Array.from({ length: 101 }, (_, i) => event({ timecode_sec: i })))), false);
});

test("rejects oversized images and aggregate evidence before rendering", () => {
  const tooLarge = "/9j/" + "A".repeat(12 * 1024 * 1024);
  assert.equal(validate(done([event({ evidence_frames: [frame(1, tooLarge)] })])), false);
  const medium = "/9j/" + "A".repeat(6 * 1024 * 1024 - 4);
  const largeReport = Array.from({ length: 8 }, (_, i) => event({
    timecode_sec: i, evidence_frames: [frame(i, medium)],
  }));
  assert.equal(validate(done(largeReport)), false);
});

test("sorts chronologically without changing server order or equal-time order", () => {
  const late = freeze(event({ type: "late", timecode_sec: 30 }));
  const early = freeze(event({ type: "early", timecode_sec: 1.25 }));
  const equal = freeze(event({ type: "equal", timecode_sec: 1.25 }));
  const input = Object.freeze([late, early, equal]);
  const output = chronologicalEvents(input);
  assert.notEqual(output, input);
  assert.deepEqual(output, [early, equal, late]);
  assert.deepEqual(input, [late, early, equal]);
  assert.equal(output[0], early);
  assert.deepEqual(chronologicalEvents([]), []);
});

test("formats fractional and hour timestamps with millisecond precision", () => {
  for (const [seconds, expected] of [
    [0, "00:00.000"], [1.5, "00:01.500"], [61.125, "01:01.125"],
    [3599.999, "59:59.999"], [3600, "01:00:00.000"],
    [3661.125, "01:01:01.125"], [90061.007, "25:01:01.007"],
  ]) {
    assert.equal(formatTimestamp(seconds), expected, `${seconds} seconds`);
  }
});

test("rounds milliseconds across minute and hour boundaries correctly", () => {
  assert.equal(formatTimestamp(59.9996), "01:00.000");
  assert.equal(formatTimestamp(3599.9996), "01:00:00.000");
  assert.equal(formatTimestamp(1.2344), "00:01.234");
});

test("invalid timestamps have no misleading numeric display", () => {
  for (const timestamp of [undefined, null, "12", true, false, NaN, Infinity, -Infinity, -1]) {
    assert.equal(formatTimestamp(timestamp), "—", String(timestamp));
  }
  const huge = formatTimestamp(Number.MAX_VALUE);
  assert.equal(typeof huge, "string");
  assert.ok(huge.length > 0);
  assert.doesNotMatch(huge, /NaN|Infinity|undefined/);
});
