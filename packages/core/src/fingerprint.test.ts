import { test } from "node:test";
import assert from "node:assert/strict";
import { fingerprintDocs } from "./fingerprint.js";
import type { Document } from "./types.js";

const A: Document = { id: "a", content: "alpha" };
const B: Document = { id: "b", content: "beta" };
const C: Document = { id: "c", content: "gamma" };

test("fingerprintDocs: same docs → same hash", () => {
  assert.equal(fingerprintDocs([A, B]), fingerprintDocs([A, B]));
});

test("fingerprintDocs: reordering changes the hash", () => {
  // We don't canonicalize order on purpose — callers who want order
  // independence sort before passing. This pins the contract.
  assert.notEqual(fingerprintDocs([A, B]), fingerprintDocs([B, A]));
});

test("fingerprintDocs: one-byte content change → different hash", () => {
  const A1: Document = { id: "a", content: "alphas" }; // one extra char
  assert.notEqual(fingerprintDocs([A]), fingerprintDocs([A1]));
});

test("fingerprintDocs: id change → different hash", () => {
  const A1: Document = { id: "a-renamed", content: "alpha" };
  assert.notEqual(fingerprintDocs([A]), fingerprintDocs([A1]));
});

test("fingerprintDocs: prefix-equal corpora don't collide", () => {
  // [A, B] and [A, B, C] start with the same data; the doc-count suffix
  // ensures they hash differently.
  assert.notEqual(fingerprintDocs([A, B]), fingerprintDocs([A, B, C]));
});

test("fingerprintDocs: id|content boundary is preserved", () => {
  // Without a field separator, {id:"ab",content:"cd"} would hash the
  // same as {id:"a",content:"bcd"}. The 0-byte separator prevents that.
  const x: Document = { id: "ab", content: "cd" };
  const y: Document = { id: "a", content: "bcd" };
  assert.notEqual(fingerprintDocs([x]), fingerprintDocs([y]));
});

test("fingerprintDocs: doc-record boundary is preserved", () => {
  // Without a record separator, two ([id="a",content="b"],[id="c",content="d"])
  // could hash the same as one ([id="abcd",content=""]). The 0xff separator prevents that.
  const split: Document[] = [{ id: "a", content: "b" }, { id: "c", content: "d" }];
  const merged: Document[] = [{ id: "abcd", content: "" }];
  assert.notEqual(fingerprintDocs(split), fingerprintDocs(merged));
});

test("fingerprintDocs: empty list is a stable, distinct value", () => {
  const fp = fingerprintDocs([]);
  assert.equal(typeof fp, "string");
  assert.equal(fingerprintDocs([]), fp);
  assert.notEqual(fingerprintDocs([A]), fp);
});

test("fingerprintDocs: handles unicode content without throwing", () => {
  const ja: Document = { id: "ja", content: "こんにちは世界" };
  const en: Document = { id: "en", content: "hello world" };
  const fpJa = fingerprintDocs([ja]);
  const fpEn = fingerprintDocs([en]);
  assert.equal(typeof fpJa, "string");
  assert.notEqual(fpJa, fpEn);
});

test("fingerprintDocs: output format is `${count}:${base36}`", () => {
  // Pinning the format so consumers (tests, debug printing) can rely on it.
  const fp = fingerprintDocs([A, B]);
  assert.match(fp, /^2:[0-9a-z]+$/);
});
