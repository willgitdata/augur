import { test } from "node:test";
import assert from "node:assert/strict";
import { tokenize } from "./embedder.js";

test("tokenize: lowercases, splits on whitespace + punctuation", () => {
  assert.deepEqual(
    tokenize("Hello, World! 42 things"),
    ["hello", "world", "42", "things"]
  );
});

test("tokenize: drops empty tokens", () => {
  assert.deepEqual(tokenize("   "), []);
  assert.deepEqual(tokenize(""), []);
});

test("tokenize: keeps unicode letters and numbers", () => {
  assert.deepEqual(tokenize("café 中文"), ["café", "中文"]);
});
