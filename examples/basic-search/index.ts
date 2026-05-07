/**
 * Basic search — the "hello world" of Augur.
 *
 * Run:  pnpm --filter example-basic-search start
 *
 * What this shows:
 *  - Index a few documents
 *  - Run several queries that exercise the router's different paths
 *  - Inspect the trace for each result
 */
import { Augur } from "@augur/core";

async function main() {
  const augr = new Augur();

  await augr.index([
    {
      id: "pg-pooling",
      content:
        "PostgreSQL connection pooling. PgBouncer is the most common option. " +
        "It runs as a separate process and multiplexes client connections to a " +
        "smaller pool of backend connections. Three modes are available: session, " +
        "transaction, and statement. Most applications use transaction mode.",
    },
    {
      id: "redis-cache",
      content:
        "Redis is an in-memory key-value store often used as a cache in front of a " +
        "database. Common patterns include cache-aside (read-through) and write-through. " +
        "Set sensible TTLs to bound staleness.",
    },
    {
      id: "k8s-probes",
      content:
        "Kubernetes liveness probes determine whether a container should be restarted. " +
        "Readiness probes determine whether the container should receive traffic. " +
        "Configure both — restarts without readiness probes can cause traffic to flow " +
        "to half-initialized pods.",
    },
    {
      id: "code-snippet",
      content:
        "Error code ERR_CONNECTION_REFUSED 4101 indicates a TCP-level rejection. " +
        "Check firewall rules, SELinux, and that the target port is bound.",
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
