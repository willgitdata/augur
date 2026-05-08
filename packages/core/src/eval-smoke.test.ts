/**
 * Eval smoke test — minimal regression harness.
 *
 * **What this is.** A synthetic 16-doc / 12-query fixture with
 * hand-crafted relevance labels, run with a deterministic stub
 * embedder. Asserts NDCG@10 > 0.65 and "every query returns ≥1
 * result". Runs in <50ms as part of `pnpm test`.
 *
 * **What this is NOT.** This is *not* the eval that produced the
 * README's NDCG@10 = 0.920 / SciFact 0.707 / FiQA 0.338 numbers.
 * Those measure retrieval *quality* against real corpora with real
 * ONNX models (LocalEmbedder + LocalReranker + MetadataChunker +
 * stemmed BM25); the harness is preserved in git history under
 * `evaluations/` (removed from main in commit feffc73). To re-run
 * those numbers, check out `feffc73^` and run `pnpm eval` from the
 * `evaluations/` package — it's a ~30 min job over the bundled
 * 504-query corpus and the BEIR datasets it pulls.
 *
 * **The bar this enforces.** A refactor that breaks "vector queries
 * find vector-relevant docs" or "keyword queries find exact-match
 * docs" should fail this test. It's the structural regression net
 * for the routing / fusion / rerank pipeline — it can't tell you if
 * you've squeezed +0.005 NDCG out of a tweak (use the real eval for
 * that), but it WILL tell you in <50ms that the pipeline still
 * works end-to-end.
 *
 * **About the floor.** NDCG@10 > 0.65 is calibrated against the
 * current stub stack. The number is meaningless to compare against
 * the real-eval numbers — different embedder, different corpus,
 * different scale. If you intentionally regress retrieval, this
 * test will fire and you'll need to decide whether to update the
 * floor or revert the change.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { Augur } from "./index.js";
import { StubEmbedder } from "./test-fixtures.js";

const CORPUS = [
  { id: "pg-pool", content: "PostgreSQL connection pooling. PgBouncer multiplexes client connections in transaction mode. Three modes: session, transaction, statement." },
  { id: "pg-vacuum", content: "VACUUM in PostgreSQL reclaims dead tuples and prevents bloat. Autovacuum runs in the background." },
  { id: "pg-vector", content: "pgvector adds vector indexing to Postgres for similarity search. HNSW and IVFFlat are the two main index types." },
  { id: "redis-cache", content: "Redis as a cache: cache-aside, write-through, write-behind. Set TTLs to bound staleness. LRU eviction is the common policy." },
  { id: "redis-stream", content: "Redis Streams support consumer groups for at-least-once delivery. XADD writes; XREADGROUP consumes." },
  { id: "k8s-probe", content: "Kubernetes liveness probes restart unhealthy containers. Readiness probes gate traffic. Configure both for safe rollouts." },
  { id: "k8s-pdb", content: "PodDisruptionBudget protects availability during voluntary disruptions like node drains. minAvailable or maxUnavailable." },
  { id: "k8s-hpa", content: "HorizontalPodAutoscaler scales replica count from CPU or custom metrics. Set sane min and max bounds." },
  { id: "tls-tls13", content: "TLS 1.3 reduces handshake round trips and removes legacy ciphers. 0-RTT is opt-in due to replay risk." },
  { id: "tls-cert", content: "X.509 certificate chains validate up to a trusted root CA. Pinning ties a deployment to a specific public key." },
  { id: "http-503", content: "HTTP 503 Service Unavailable signals temporary overload. Retry-After header advises when to retry." },
  { id: "http-429", content: "HTTP 429 Too Many Requests is the rate-limit response. Servers should send Retry-After." },
  { id: "rust-async", content: "Rust async/await runs on top of an executor (tokio, async-std). Futures are zero-cost; .await yields to the runtime." },
  { id: "rust-borrow", content: "Rust's borrow checker enforces aliasing XOR mutation at compile time. References cannot outlive their referents." },
  { id: "py-gil", content: "CPython's GIL serializes bytecode execution per process. Threads still help for I/O; multiprocessing for CPU." },
  { id: "py-asyncio", content: "Python asyncio runs coroutines on an event loop. async def + await; gather for concurrency; tasks for fire-and-forget." },
];

// Hand-labeled judgments. Each query maps to the IDs of clearly relevant docs.
const QUERIES: Array<{ q: string; relevant: string[] }> = [
  { q: "How do I configure connection pooling in Postgres?", relevant: ["pg-pool"] },
  { q: "Postgres bloat from dead tuples", relevant: ["pg-vacuum"] },
  { q: "vector similarity search in postgres", relevant: ["pg-vector"] },
  { q: '"liveness probes"', relevant: ["k8s-probe"] },
  { q: "kubernetes autoscaling", relevant: ["k8s-hpa"] },
  { q: "503 Service Unavailable", relevant: ["http-503"] },
  { q: "rate limit response code", relevant: ["http-429"] },
  { q: "TLS 1.3 0-RTT", relevant: ["tls-tls13"] },
  { q: "rust borrow checker", relevant: ["rust-borrow"] },
  { q: "python global interpreter lock", relevant: ["py-gil"] },
  { q: "redis cache patterns", relevant: ["redis-cache"] },
  { q: "what is a PodDisruptionBudget", relevant: ["k8s-pdb"] },
];

// NDCG@k with binary relevance — robust enough for a smoke test.
function ndcg(retrievedIds: string[], relevant: Set<string>, k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, retrievedIds.length); i++) {
    if (relevant.has(retrievedIds[i]!)) dcg += 1 / Math.log2(i + 2);
  }
  let idcg = 0;
  for (let i = 0; i < Math.min(k, relevant.size); i++) {
    idcg += 1 / Math.log2(i + 2);
  }
  return idcg === 0 ? 0 : dcg / idcg;
}

test("eval smoke: routing pipeline meets NDCG@10 floor on synthetic corpus", async () => {
  const augr = new Augur({ embedder: new StubEmbedder() });
  await augr.index(CORPUS);

  const scores: number[] = [];
  for (const { q, relevant } of QUERIES) {
    const { results } = await augr.search({ query: q, topK: 10 });
    const ids = results.map((r) => r.chunk.documentId);
    scores.push(ndcg(ids, new Set(relevant), 10));
  }

  const meanNdcg = scores.reduce((a, b) => a + b, 0) / scores.length;
  // Floor: if you regress retrieval to "broken", this fires. The number
  // is calibrated to the current stub stack; intentional regressions
  // need to update it consciously.
  assert.ok(
    meanNdcg > 0.65,
    `eval smoke regressed: mean NDCG@10 = ${meanNdcg.toFixed(3)} (floor 0.65)`
  );
});

test("eval smoke: every query returns at least one result", async () => {
  // Cheap structural assertion — if the pipeline returns zero results
  // for a query that has a clearly relevant doc, something is broken
  // before we even talk about NDCG. Catches "filter ate everything"
  // regressions in particular.
  const augr = new Augur({ embedder: new StubEmbedder() });
  await augr.index(CORPUS);
  for (const { q } of QUERIES) {
    const { results } = await augr.search({ query: q, topK: 10 });
    assert.ok(results.length > 0, `zero results for query: ${q}`);
  }
});
