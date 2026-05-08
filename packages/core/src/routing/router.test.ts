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

// ---------- new signal-driven branches ----------

test("router routes non-English queries to vector", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "重複行 削除 PostgreSQL" }, fullCaps);
  assert.equal(d.strategy, "vector");
  assert.ok(d.reasons.some((x) => x.includes("non-English")));
});

test("router treats code-like syntax as keyword on short queries", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "useState pattern react" }, fullCaps);
  assert.equal(d.strategy, "keyword");
  assert.ok(d.reasons.some((x) => x.includes("code-like")));
});

test("router routes date/version tokens (no code syntax) to hybrid", () => {
  const r = new HeuristicRouter();
  // Identifier-only short queries get hybrid: BM25 carries the literal
  // match, vector adds topical recall when the relevant doc discusses the
  // concept without naming the specific RFC/CVE. See router rule 5b.
  const d = r.decide({ query: "RFC 7519 JWT" }, fullCaps);
  assert.equal(d.strategy, "hybrid");
  assert.ok(d.reasons.some((x) => x.includes("date/version")));
});

test("router routes a CVE token to hybrid", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "CVE-2024-3094" }, fullCaps);
  assert.equal(d.strategy, "hybrid");
});

test("router keeps code-like specific tokens on keyword", () => {
  const r = new HeuristicRouter();
  // hasCodeLike fires (snake_case) → rule 5a wins over 5b.
  const d = r.decide({ query: "ssl: SSL_ERROR_SYSCALL" }, fullCaps);
  assert.equal(d.strategy, "keyword");
  assert.ok(d.reasons.some((x) => x.includes("code-like")));
});

test("router routes mid-query named entity to keyword (not a question)", () => {
  const r = new HeuristicRouter();
  // Query has no specific/code/date tokens, just a mid-query proper noun.
  const d = r.decide({ query: "setup Stripe payments production" }, fullCaps);
  assert.equal(d.strategy, "keyword");
  assert.ok(d.reasons.some((x) => x.includes("named entity")));
});

test("router routes definitional questions to vector", () => {
  const r = new HeuristicRouter();
  // No specific/code tokens so the short-keyword branch doesn't fire first.
  const d = r.decide({ query: "what is the rust borrow checker" }, fullCaps);
  assert.equal(d.strategy, "vector");
  assert.ok(d.reasons.some((x) => x.includes("definitional")));
});

test("router routes factoid questions to hybrid", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "where does redis store data on disk" }, fullCaps);
  assert.equal(d.strategy, "hybrid");
  assert.ok(d.reasons.some((x) => x.includes("factoid")));
});

test("router routes procedural questions to vector with reason 'procedural'", () => {
  const r = new HeuristicRouter();
  const d = r.decide(
    { query: "how do I tune autovacuum on a large table" },
    fullCaps
  );
  assert.equal(d.strategy, "vector");
  assert.ok(d.reasons.some((x) => x.includes("procedural")));
});

test("router preserves untyped question fallback (Can/Should/Is)", () => {
  const r = new HeuristicRouter();
  const d = r.decide(
    { query: "Can I deploy without taking the system down" },
    fullCaps
  );
  // "without" triggers negation → reranking forced; strategy still "vector".
  assert.equal(d.strategy, "vector");
});

test("default router reranks on every strategy (auto = best)", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "ssl: SSL_ERROR_SYSCALL" }, fullCaps);
  assert.equal(d.strategy, "keyword");
  assert.equal(d.reranked, true);
  assert.ok(
    d.reasons.some((x) => x.includes("alwaysRerank")),
    "trace should explain the routing decision"
  );
});

test("default router still respects tight latency budgets", () => {
  const r = new HeuristicRouter();
  const d = r.decide({ query: "kubectl apply", latencyBudgetMs: 50 }, fullCaps);
  assert.equal(d.reranked, false);
});

test("alwaysRerank=false restores fast keyword path", () => {
  const r = new HeuristicRouter({ alwaysRerank: false });
  const d = r.decide({ query: "ssl: SSL_ERROR_SYSCALL" }, fullCaps);
  assert.equal(d.strategy, "keyword");
  assert.equal(d.reranked, false);
});

test("alwaysRerank=false: negation still forces rerank on keyword", () => {
  const r = new HeuristicRouter({ alwaysRerank: false });
  const d = r.decide(
    { query: '"connection refused" without firewall' },
    fullCaps
  );
  assert.equal(d.strategy, "keyword");
  assert.equal(d.reranked, true);
  assert.ok(d.reasons.some((x) => x.includes("negation")));
});
