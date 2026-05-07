import { test } from "node:test";
import assert from "node:assert/strict";
import { HeuristicRouter } from "./router.js";
import type { AdapterCapabilities } from "../adapters/adapter.js";

const fullCaps: AdapterCapabilities = {
  vector: true,
  keyword: true,
  hybrid: true,
  computesEmbeddings: false,
  filtering: true,
};

const vectorOnlyCaps: AdapterCapabilities = {
  vector: true,
  keyword: false,
  hybrid: false,
  computesEmbeddings: false,
  filtering: false,
};

test("router picks keyword for quoted phrases", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: 'find "exact phrase" in docs' }, fullCaps);
  assert.equal(d.strategy, "keyword");
});

test("router picks keyword for short specific queries", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "ERR_CONNECTION_TIMEOUT 504" }, fullCaps);
  assert.equal(d.strategy, "keyword");
});

test("router picks vector for natural-language questions", () => {
  const r = new HeuristicRouter();
  const d = r.decide(
    { query: "How do I configure database connection pooling?" },
    fullCaps
  );
  assert.equal(d.strategy, "vector");
});

test("router defaults to hybrid for ambiguous mid-length queries", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "deploy production rollout pipeline" }, fullCaps);
  assert.equal(d.strategy, "hybrid");
});

test("router falls back to vector when adapter only supports vector", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: 'find "exact phrase" here' }, vectorOnlyCaps);
  assert.equal(d.strategy, "vector");
  assert.ok(d.reasons.some((r) => r.includes("vector-only")));
});

test("router enables reranking when latency budget allows", () => {
  const r = new HeuristicRouter();
  const d = r.decide(
    { query: "How do I configure database pooling?", latencyBudgetMs: 2000 },
    fullCaps
  );
  assert.equal(d.reranked, true);
});

test("router skips reranking when latency budget is tight", () => {
  const r = new HeuristicRouter();
  const d = r.decide(
    { query: "How do I configure database pooling?", latencyBudgetMs: 200 },
    fullCaps
  );
  assert.equal(d.reranked, false);
});

test("forceStrategy is honored", () => {
  const r = new HeuristicRouter();
  const d = r.decide(
    { query: "anything goes here", forceStrategy: "keyword" },
    fullCaps
  );
  assert.equal(d.strategy, "keyword");
});
