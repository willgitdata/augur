/**
 * Basic search — the "hello world" of Augur with the recommended stack.
 *
 * Run (from the repo root):
 *   pnpm install
 *   pnpm --filter example-basic-search start
 *
 * **Standalone copy of this example?** `LocalEmbedder` and
 * `LocalReranker` use the optional peer dep `@huggingface/transformers`.
 * The repo workspace already resolves it; if you copy this file out of
 * the repo into a fresh project, install it explicitly:
 *
 *   npm i @augur-rag/core @huggingface/transformers
 *
 * This example reproduces the headline configuration from the README:
 *
 *   - LocalEmbedder                  — 22 MB ONNX bi-encoder (Xenova/all-MiniLM-L6-v2)
 *   - LocalReranker                  — 22 MB ONNX cross-encoder, voting on every query
 *   - MetadataChunker(SentenceChunker) — prepends doc metadata so it's searchable
 *   - InMemoryAdapter({ useStemming: true }) — BM25 with English stemming + cosine
 *
 * Total on-device footprint: ~44 MB. No API keys, no network at query time.
 *
 * What this shows:
 *  - Index a few documents (with metadata that the chunker prepends)
 *  - Run several queries that exercise the router's different paths
 *  - Inspect the trace for each result
 */
import {
  Augur,
  InMemoryAdapter,
  LocalEmbedder,
  LocalReranker,
  MetadataChunker,
  SentenceChunker,
} from "@augur-rag/core";

async function main() {
  const augr = new Augur({
    embedder: new LocalEmbedder(),
    reranker: new LocalReranker(),
    chunker: new MetadataChunker({ base: new SentenceChunker() }),
    adapter: new InMemoryAdapter({ useStemming: true }),
  });

  await augr.index([
    {
      id: "pg-pooling",
      content:
        "PostgreSQL connection pooling. PgBouncer is the most common option. " +
        "It runs as a separate process and multiplexes client connections to a " +
        "smaller pool of backend connections. Three modes are available: session, " +
        "transaction, and statement. Most applications use transaction mode.",
      metadata: { topic: "postgres", title: "Connection pooling" },
    },
    {
      id: "redis-cache",
      content:
        "Redis is an in-memory key-value store often used as a cache in front of a " +
        "database. Common patterns include cache-aside (read-through) and write-through. " +
        "Set sensible TTLs to bound staleness.",
      metadata: { topic: "redis", title: "Caching patterns" },
    },
    {
      id: "k8s-probes",
      content:
        "Kubernetes liveness probes determine whether a container should be restarted. " +
        "Readiness probes determine whether the container should receive traffic. " +
        "Configure both — restarts without readiness probes can cause traffic to flow " +
        "to half-initialized pods.",
      metadata: { topic: "kubernetes", title: "Health probes" },
    },
    {
      id: "code-snippet",
      content:
        "Error code ERR_CONNECTION_REFUSED 4101 indicates a TCP-level rejection. " +
        "Check firewall rules, SELinux, and that the target port is bound.",
      metadata: { topic: "networking", title: "Connection errors" },
    },
  ]);

  const queries = [
    "How do I configure connection pooling in Postgres?",
    "ERR_CONNECTION_REFUSED 4101",
    '"liveness probes"',
    "kubernetes",
  ];

  for (const q of queries) {
    const { results, trace } = await augr.search({ query: q, topK: 2 });
    console.log("\n=== Query:", q);
    console.log(
      `Strategy: ${trace.decision.strategy}${trace.decision.reranked ? " (+rerank)" : ""}  ` +
        `· ${trace.totalMs.toFixed(1)} ms · ${trace.candidates} candidates`
    );
    console.log("Reasons:");
    for (const r of trace.decision.reasons) console.log("  -", r);
    console.log("Top results:");
    for (const r of results) {
      const preview = r.chunk.content.slice(0, 80).replace(/\s+/g, " ");
      console.log(`  [${r.score.toFixed(3)}] ${r.chunk.documentId}  "${preview}…"`);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
