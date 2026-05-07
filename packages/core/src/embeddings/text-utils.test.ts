import { test } from "node:test";
import assert from "node:assert/strict";
import { stem, STOPWORDS, tokenizeAdvanced } from "./text-utils.js";

test("stem: basic suffixes", () => {
  // The Porter stemmer is *aggressive*. These are the canonical reductions.
  assert.equal(stem("running"), "run");
  assert.equal(stem("connections"), "connect");
  assert.equal(stem("agreed"), "agre");
  assert.equal(stem("happy"), "happi");
  assert.equal(stem("ponies"), "poni");
  assert.equal(stem("relational"), "relat");
});

test("stem: short words are returned as-is", () => {
  assert.equal(stem("the"), "the");
  assert.equal(stem("an"), "an");
  assert.equal(stem("at"), "at");
});

test("stem: irregular roots become the same stem", () => {
  // Same morphological root → same stem (the test of a stemmer's value).
  assert.equal(stem("connection"), stem("connections"));
  assert.equal(stem("running"), stem("runs"));
  assert.equal(stem("argued"), stem("argues"));
});

test("STOPWORDS: contains common closed-class words", () => {
  for (const w of ["the", "a", "and", "of", "to", "in", "for", "is", "are", "with"]) {
    assert.ok(STOPWORDS.has(w), `expected stopword: ${w}`);
  }
});

test("STOPWORDS: does not contain content words", () => {
  for (const w of ["postgres", "kubernetes", "redis", "compute", "rust"]) {
    assert.ok(!STOPWORDS.has(w), `expected NOT stopword: ${w}`);
  }
});

test("tokenizeAdvanced: default behavior is plain whitespace split", () => {
  assert.deepEqual(
    tokenizeAdvanced("How do I configure pgvector?"),
    ["how", "do", "i", "configure", "pgvector"]
  );
});

test("tokenizeAdvanced: dropStopwords removes common words", () => {
  const out = tokenizeAdvanced("How do I configure pgvector?", { dropStopwords: true });
  assert.deepEqual(out, ["configure", "pgvector"]);
});

test("tokenizeAdvanced: stem reduces inflectional forms", () => {
  const out = tokenizeAdvanced("running connections agreed", { stem: true });
  assert.deepEqual(out, ["run", "connect", "agre"]);
});

test("tokenizeAdvanced: stem + stopwords together", () => {
  const out = tokenizeAdvanced("the cats are running quickly", {
    stem: true,
    dropStopwords: true,
  });
  // "the", "are" stripped; "cats" → "cat" (stem); "running" → "run"; "quickly" → "quickli"
  assert.deepEqual(out, ["cat", "run", "quickli"]);
});
