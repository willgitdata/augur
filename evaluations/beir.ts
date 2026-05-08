/**
 * BEIR runner — load a BEIR dataset, run our auto-routing pipeline, report
 * NDCG@10 / MRR / Recall@10 in the standard BEIR format. Apples-to-apples
 * with the published numbers for BM25, dense, and reranked baselines.
 *
 * Usage:
 *   pnpm exec tsx evaluations/beir.ts /tmp/beir/nfcorpus
 *   pnpm exec tsx evaluations/beir.ts /tmp/beir/scifact
 *
 * Expects a BEIR-format directory:
 *   <root>/corpus.jsonl       { _id, title?, text, metadata? }
 *   <root>/queries.jsonl      { _id, text }
 *   <root>/qrels/test.tsv     query-id\tcorpus-id\tscore
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import {
  Augur,
  HeuristicRouter,
  InMemoryAdapter,
  LocalEmbedder,
  LocalReranker,
  MetadataChunker,
  SentenceChunker,
} from "@augur/core";
import { dcgAt, mean, reciprocalRank } from "./metrics.js";

interface BeirDoc {
  _id: string;
  title?: string;
  text: string;
  metadata?: Record<string, unknown>;
}
interface BeirQuery {
  _id: string;
  text: string;
}

// Argv: <dataset-dir> [--model NAME] [--query-prefix STR] [--doc-prefix STR]
const argv = process.argv.slice(2);
const root = argv[0];
if (!root || root.startsWith("--")) {
  console.error(
    "Usage: tsx evaluations/beir.ts <beir-dataset-dir> [--model NAME] [--query-prefix STR] [--doc-prefix STR]"
  );
  process.exit(1);
}
function readFlag(name: string): string | undefined {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
}
const modelOverride = readFlag("model");
const queryPrefix = readFlag("query-prefix");
const docPrefix = readFlag("doc-prefix");
const dtype = readFlag("dtype");
const device = readFlag("device");
const fastKeyword = argv.includes("--fast-keyword");

const datasetName = root.split("/").filter(Boolean).pop()!;

console.log(`BEIR runner — ${datasetName}`);
if (modelOverride) console.log(`  embedder model : ${modelOverride}`);
if (queryPrefix) console.log(`  query prefix   : ${JSON.stringify(queryPrefix)}`);
if (docPrefix) console.log(`  doc prefix     : ${JSON.stringify(docPrefix)}`);
if (dtype) console.log(`  dtype          : ${dtype}`);
if (device) console.log(`  device         : ${device}`);
if (fastKeyword) console.log(`  fast-keyword   : on (skip rerank on keyword strategy)`);

// ---------- load ----------
function readJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as T);
}

const t0 = performance.now();
const corpus = readJsonl<BeirDoc>(join(root, "corpus.jsonl"));
const queries = readJsonl<BeirQuery>(join(root, "queries.jsonl"));

// qrels: query-id\tcorpus-id\tscore (with header row)
const qrelsRaw = readFileSync(join(root, "qrels/test.tsv"), "utf8")
  .split("\n")
  .filter((l) => l.trim().length > 0);
const qrelsHeader = qrelsRaw[0]!.toLowerCase();
const qrelLines = qrelsHeader.startsWith("query-id") ? qrelsRaw.slice(1) : qrelsRaw;
const qrels = new Map<string, Map<string, number>>();
for (const line of qrelLines) {
  const [qid, did, scoreStr] = line.split("\t");
  if (!qid || !did || !scoreStr) continue;
  const score = parseInt(scoreStr, 10);
  if (score <= 0) continue; // BEIR convention: 0 = not relevant
  let m = qrels.get(qid);
  if (!m) {
    m = new Map();
    qrels.set(qid, m);
  }
  m.set(did, score);
}

// Restrict to queries that have qrels in the test split.
const testQueries = queries.filter((q) => qrels.has(q._id));
console.log(
  `corpus=${corpus.length} docs · queries=${testQueries.length} test (of ${queries.length} total) · qrels=${[...qrels.values()].reduce((s, m) => s + m.size, 0)} judgements`
);
console.log(`load: ${(performance.now() - t0).toFixed(0)}ms\n`);

// ---------- index ----------
const embedderOpts: ConstructorParameters<typeof LocalEmbedder>[0] = {};
if (modelOverride) embedderOpts.model = modelOverride;
if (queryPrefix !== undefined) embedderOpts.queryPrefix = queryPrefix;
if (docPrefix !== undefined) embedderOpts.docPrefix = docPrefix;
if (dtype) embedderOpts.dtype = dtype as "fp32" | "fp16" | "q8" | "q4";
if (device) embedderOpts.device = device as "wasm" | "webgpu" | "cpu";

const augr = new Augur({
  embedder: new LocalEmbedder(embedderOpts),
  reranker: new LocalReranker(),
  chunker: new MetadataChunker({ base: new SentenceChunker() }),
  adapter: new InMemoryAdapter({ useStemming: true }),
  ...(fastKeyword ? { router: new HeuristicRouter({ alwaysRerank: false }) } : {}),
});

const docs = corpus.map((d) => ({
  id: d._id,
  content: d.title ? `${d.title}. ${d.text}` : d.text,
  ...(d.metadata ? { metadata: d.metadata } : {}),
}));

console.log("indexing…");
const i0 = performance.now();
const indexResult = await augr.index(docs);
const indexMs = performance.now() - i0;
console.log(
  `indexed ${indexResult.documents} docs → ${indexResult.chunks} chunks in ${(indexMs / 1000).toFixed(1)}s (${(indexResult.chunks / (indexMs / 1000)).toFixed(0)} chunks/sec)\n`
);

// ---------- query ----------
console.log(`querying ${testQueries.length} test queries with auto-routing…`);
const ndcgs: number[] = [];
const mrrs: number[] = [];
const recalls: number[] = [];
const lat: number[] = [];
const strategyCounts: Record<string, number> = {};

const q0 = performance.now();
let done = 0;
const reportEvery = Math.max(50, Math.floor(testQueries.length / 10));

for (const q of testQueries) {
  const qrel = qrels.get(q._id)!;
  const tQ = performance.now();
  const { results, trace } = await augr.search({ query: q.text, topK: 10 });
  lat.push(performance.now() - tQ);

  // Aggregate to doc level (multi-chunk docs count once at highest rank).
  const seen = new Set<string>();
  const top10Rel: number[] = [];
  for (const r of results) {
    const did = r.chunk.documentId;
    if (seen.has(did)) continue;
    seen.add(did);
    top10Rel.push(qrel.get(did) ?? 0);
  }

  // Graded NDCG: ideal is the qrel scores sorted desc, top 10.
  const ideal = [...qrel.values()].sort((a, b) => b - a).slice(0, 10);
  const dcg = dcgAt(top10Rel, 10);
  const idcg = dcgAt(ideal, 10);
  const ndcg = idcg === 0 ? 0 : dcg / idcg;
  ndcgs.push(ndcg);
  mrrs.push(reciprocalRank(top10Rel));
  const totalRelevant = qrel.size;
  const hits = top10Rel.filter((r) => r > 0).length;
  recalls.push(totalRelevant === 0 ? 0 : hits / totalRelevant);

  strategyCounts[trace.decision.strategy] = (strategyCounts[trace.decision.strategy] ?? 0) + 1;

  done++;
  if (done % reportEvery === 0) {
    process.stdout.write(`  ${done}/${testQueries.length}…\n`);
  }
}
const queryMs = performance.now() - q0;
const sortedLat = [...lat].sort((a, b) => a - b);
const ql = (p: number) => sortedLat[Math.min(sortedLat.length - 1, Math.floor(p * sortedLat.length))]!;

// ---------- report ----------
console.log("\n" + "=".repeat(60));
console.log(`BEIR ${datasetName} — auto-routing, full local stack`);
console.log("=".repeat(60));
console.log(`NDCG@10   ${mean(ndcgs).toFixed(3)}`);
console.log(`MRR@10    ${mean(mrrs).toFixed(3)}`);
console.log(`Recall@10 ${mean(recalls).toFixed(3)}`);
console.log();
console.log(`Strategy distribution:`);
const totalQ = Object.values(strategyCounts).reduce((a, b) => a + b, 0);
for (const [s, c] of Object.entries(strategyCounts)) {
  console.log(`  ${s.padEnd(8)} ${c}  (${((c / totalQ) * 100).toFixed(0)}%)`);
}
console.log();
console.log(
  `Latency: total=${(queryMs / 1000).toFixed(1)}s · QPS=${(testQueries.length / (queryMs / 1000)).toFixed(1)} · p50=${ql(0.5).toFixed(1)}ms · p95=${ql(0.95).toFixed(1)}ms · p99=${ql(0.99).toFixed(1)}ms`
);
