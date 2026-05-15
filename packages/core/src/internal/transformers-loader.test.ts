import { test } from "node:test";
import assert from "node:assert/strict";
import {
  MissingTransformersError,
  interpretImportError,
} from "./transformers-loader.js";

test("interpretImportError: ERR_MODULE_NOT_FOUND → MissingTransformersError", () => {
  const err = Object.assign(new Error("Cannot find package '@huggingface/transformers'"), {
    code: "ERR_MODULE_NOT_FOUND",
  });
  const out = interpretImportError(err);
  assert.ok(out instanceof MissingTransformersError);
  assert.equal(out!.cause, err);
});

test("interpretImportError: CJS MODULE_NOT_FOUND → MissingTransformersError", () => {
  const err = Object.assign(new Error("Cannot find module '@huggingface/transformers'"), {
    code: "MODULE_NOT_FOUND",
  });
  assert.ok(interpretImportError(err) instanceof MissingTransformersError);
});

test("interpretImportError: error message-only match (no code) → MissingTransformersError", () => {
  // Some runtimes throw without a code; rely on the message.
  const err = new Error(`Cannot find module '@huggingface/transformers' imported from foo.js`);
  assert.ok(interpretImportError(err) instanceof MissingTransformersError);
});

test("interpretImportError: unrelated error → null (caller rethrows original)", () => {
  const err = new Error("ONNX runtime crashed");
  assert.equal(interpretImportError(err), null);
});

test("interpretImportError: non-Error throwable → null", () => {
  assert.equal(interpretImportError("just a string"), null);
  assert.equal(interpretImportError(undefined), null);
  assert.equal(interpretImportError(null), null);
});

test("MissingTransformersError: message names the package and install command", () => {
  const err = new MissingTransformersError();
  assert.match(err.message, /@huggingface\/transformers/);
  assert.match(err.message, /npm install @huggingface\/transformers/);
  assert.equal(err.name, "MissingTransformersError");
});
