import { test } from "node:test";
import assert from "node:assert/strict";
import { sha256Hex, randomUuid, utf8ByteLength } from "./sha256.js";

test("sha256Hex: matches the canonical SHA-256 of an empty string", async () => {
  // RFC test vector: empty input → e3b0c4...855
  const out = await sha256Hex("");
  assert.equal(
    out,
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  );
});

test("sha256Hex: matches the canonical SHA-256 of 'abc'", async () => {
  // RFC test vector: "abc" → ba7816...ad15
  const out = await sha256Hex("abc");
  assert.equal(
    out,
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  );
});

test("sha256Hex: handles multi-byte UTF-8 correctly", async () => {
  // "你好" → utf-8 bytes E4 BD A0 E5 A5 BD. Expected SHA-256 is a published
  // value; tests our TextEncoder path, not just ASCII.
  const out = await sha256Hex("你好");
  assert.equal(
    out,
    "670d9743542cae3ea7ebe36af56bd53648b0a1126162e78d81a32934a711302e"
  );
});

test("sha256Hex: output is always 64 lowercase hex chars", async () => {
  const out = await sha256Hex("anything");
  assert.equal(out.length, 64);
  assert.match(out, /^[0-9a-f]{64}$/);
});

test("randomUuid: returns RFC 4122 v4 UUID shape", () => {
  const out = randomUuid();
  assert.match(
    out,
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
  );
});

test("randomUuid: two calls return different UUIDs", () => {
  const a = randomUuid();
  const b = randomUuid();
  assert.notEqual(a, b);
});

test("utf8ByteLength: ASCII is 1 byte per char", () => {
  assert.equal(utf8ByteLength("hello"), 5);
  assert.equal(utf8ByteLength(""), 0);
});

test("utf8ByteLength: multi-byte UTF-8 counts bytes, not chars", () => {
  // 你 = 3 bytes (UTF-8), 好 = 3 bytes → 6 bytes total even though string.length is 2.
  assert.equal("你好".length, 2);
  assert.equal(utf8ByteLength("你好"), 6);
});
