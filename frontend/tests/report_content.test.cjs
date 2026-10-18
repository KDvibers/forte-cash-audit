"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { normalizeScope, scopeText, descriptionParts, renderDescription } = require("../results.js");

const scope = () => ({ rules: [{ id: "cash_after_departure", title: "Деньги после ухода" }],
  notice: "Проверены выбранные кадры.", complete_case_coverage: false });

test("scope metadata is bounded, copied and rendered only as literal text", () => {
  const value = scope();
  value.rules[0].title = '<img src=x onerror="alert(1)">';
  value.extra = "do not propagate";
  value.rules[0].extra = "do not propagate";
  const normalized = normalizeScope(value);
  assert.deepEqual(normalized, { rules: [{ id: "cash_after_departure", title: value.rules[0].title }],
    notice: value.notice, complete_case_coverage: false });
  assert.notEqual(normalized, value);
  assert.notEqual(normalized.rules[0], value.rules[0]);
  assert.match(scopeText(normalized), /<img src=x/);
  assert.match(scopeText(normalized), /не охватывает все возможные нарушения/);
  for (const invalid of [null, {}, { ...scope(), rules: [] }, { ...scope(), rules: [null] },
    { ...scope(), rules: [scope().rules[0], scope().rules[0]] },
    { ...scope(), rules: [{ id: "x".repeat(201), title: "name" }] },
    { ...scope(), rules: [{ id: "id", title: "x".repeat(501) }] },
    { ...scope(), notice: "x".repeat(4001) }, { ...scope(), complete_case_coverage: "false" }]) {
    assert.equal(normalizeScope(invalid), null);
    assert.match(scopeText(invalid), /Объём выполненных проверок не указан/);
  }
});

test("official Adilet links retain all description text, punctuation and fragments", () => {
  const source = "https://adilet.zan.kz/rus/docs/V1900019680#z112";
  const text = `Наблюдение\nИсточник: ${source}. Затем (${source}), пояснение. <b>Текст</b>`;
  const parts = descriptionParts(text);
  assert.equal(parts.map((part) => part.text).join(""), text);
  assert.deepEqual(parts.filter((part) => part.href).map((part) => part.href), [source, source]);
  assert.equal(descriptionParts("" ).length, 0);
});

test("malicious, non-HTTPS and lookalike source addresses remain non-clickable text", () => {
  const addresses = [
    "javascript:alert(1)", "data:text/html,<script>alert(1)</script>",
    "http://adilet.zan.kz/rus/docs/example", "https://adilet.zan.kz.evil.example/rule",
    "https://adilet.zan.kz@evil.example/rule", "https://evil.example@adilet.zan.kz/rule",
    "https://adilet.zan.kz:8443/rule", "https://adilet.zan.kz\\evil.example/rule",
    "https://adilet.zan.kz./rule", "//adilet.zan.kz/rule", "https://example.org/rule",
  ];
  for (const text of addresses) {
    const parts = descriptionParts(text);
    assert.equal(parts.map((part) => part.text).join(""), text);
    assert.equal(parts.some((part) => part.href), false, text);
  }
});

test("description renderer creates text nodes and safe anchors without interpreting HTML", () => {
  const previous = global.document;
  function node(tagName, text = "") {
    return { tagName, textContent: text, children: [], append(child) { this.children.push(child); } };
  }
  global.document = {
    createElement: (tag) => node(tag),
    createTextNode: (text) => node("#text", text),
  };
  try {
    const source = "https://adilet.zan.kz/rus/docs/V1900019680#z112";
    const text = `<img src=x onerror=alert(1)> ${source}. javascript:alert(2)`;
    const paragraph = renderDescription(text);
    assert.equal(paragraph.tagName, "p");
    assert.equal(paragraph.children.map((child) => child.textContent).join(""), text);
    const links = paragraph.children.filter((child) => child.tagName === "a");
    assert.equal(links.length, 1);
    assert.equal(links[0].href, source);
    assert.equal(links[0].target, "_blank");
    assert.equal(links[0].rel, "noopener noreferrer");
    assert.equal(paragraph.children.some((child) => child.tagName === "img"), false);
    assert.equal(paragraph.children.some((child) => "innerHTML" in child), false);
  } finally {
    if (previous === undefined) delete global.document;
    else global.document = previous;
  }
});
