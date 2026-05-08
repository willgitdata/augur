/**
 * Eval CLI. Loads corpus + queries from disk, runs the Augur pipeline, and
 * prints aggregate + per-strategy + per-category metrics.
 *
 * Usage:
 *   pnpm eval                                   # default config
 *   pnpm eval -- --verbose                      # per-query lines
 *   pnpm eval -- --save out.json                # write metrics JSON
 *   pnpm eval -- --compare baseline.json        # diff vs baseline
 *   pnpm eval -- --embedder tfidf               # swap embedder (hash | tfidf)
 *   pnpm eval -- --metadata-chunker             # prepend doc metadata to chunks
 *   pnpm eval -- --embedder tfidf --metadata-chunker --save tfidf-meta.json
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  Augur,
  CascadedReranker,
  Doc2QueryChunker,
  GeminiEmbedder,
  HashEmbedder,
  HeuristicReranker,
  InMemoryAdapter,
  LocalEmbedder,
  LocalReranker,
  MetadataChunker,
  MMRReranker,
  SentenceChunker,
  TfIdfEmbedder,
  type Chunker,
  type Embedder,
  type Reranker,
  type SemanticChunker,
} from "@augur/core";
import { formatReport, runEval, type EvalDoc, type EvalQuery, type EvalReport } from "./runner.js";

const HERE = dirname(fileURLToPath(import.meta.url));

interface Args {
  verbose: boolean;
  save?: string;
  compare?: string;
  embedder: "hash" | "tfidf" | "gemini" | "local";
  reranker: "none" | "heuristic" | "local";
  localEmbedderModel?: string;
  localEmbedderQueryPrefix?: string;
  localEmbedderDocPrefix?: string;
  localRerankerModel?: string;
  geminiModel?: string;
  geminiThrottleMs?: number;
  geminiCacheDir?: string;
  limit?: number;
  metadataChunker: boolean;
  doc2query: boolean;
  doc2queryModel?: string;
  doc2queryNumQueries?: number;
  bm25Stem: boolean;
  mmr: boolean;
  mmrLambda: number;
}

function parseArgs(argv: string[]): Args {
  const out: Args = {
    verbose: false,
    embedder: "hash",
    reranker: "heuristic",
    metadataChunker: false,
    bm25Stem: false,
    mmr: false,
    mmrLambda: 0.7,
    doc2query: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--verbose" || a === "-v") out.verbose = true;
    else if (a === "--save") out.save = argv[++i];
    else if (a === "--compare") out.compare = argv[++i];
    else if (a === "--embedder") {
      const v = argv[++i];
      if (v !== "hash" && v !== "tfidf" && v !== "gemini" && v !== "local") {
        throw new Error(`--embedder must be 'hash' | 'tfidf' | 'gemini' | 'local', got ${v}`);
      }
      out.embedder = v;
    } else if (a === "--reranker") {
      const v = argv[++i];
      if (v !== "none" && v !== "heuristic" && v !== "local") {
        throw new Error(`--reranker must be 'none' | 'heuristic' | 'local', got ${v}`);
      }
      out.reranker = v;
    } else if (a === "--local-embedder-model") out.localEmbedderModel = argv[++i];
    else if (a === "--local-embedder-query-prefix") out.localEmbedderQueryPrefix = argv[++i];
    else if (a === "--local-embedder-doc-prefix") out.localEmbedderDocPrefix = argv[++i];
    else if (a === "--local-reranker-model") out.localRerankerModel = argv[++i];
    else if (a === "--gemini-model") out.geminiModel = argv[++i];
    else if (a === "--gemini-throttle") out.geminiThrottleMs = parseInt(argv[++i]!, 10);
    else if (a === "--gemini-cache-dir") out.geminiCacheDir = argv[++i];
    else if (a === "--limit") out.limit = parseInt(argv[++i]!, 10);
    else if (a === "--metadata-chunker") out.metadataChunker = true;
    else if (a === "--bm25-stem") out.bm25Stem = true;
    else if (a === "--mmr") out.mmr = true;
    else if (a === "--mmr-lambda") out.mmrLambda = parseFloat(argv[++i]!);
    else if (a === "--doc2query") out.doc2query = true;
    else if (a === "--doc2query-model") out.doc2queryModel = argv[++i];
    else if (a === "--doc2query-n") out.doc2queryNumQueries = parseInt(argv[++i]!, 10);
  }
  return out;
}

function buildEmbedder(args: Args): Embedder {
  if (args.embedder === "tfidf") return new TfIdfEmbedder();
  if (args.embedder === "gemini") {
    return new GeminiEmbedder({
      ...(args.geminiModel ? { model: args.geminiModel } : {}),
      ...(args.geminiThrottleMs !== undefined ? { throttleMs: args.geminiThrottleMs } : {}),
      ...(args.geminiCacheDir ? { cacheDir: args.geminiCacheDir } : {}),
    });
  }
  if (args.embedder === "local") {
    return new LocalEmbedder({
      ...(args.localEmbedderModel ? { model: args.localEmbedderModel } : {}),
      ...(args.localEmbedderQueryPrefix !== undefined
        ? { queryPrefix: args.localEmbedderQueryPrefix }
        : {}),
      ...(args.localEmbedderDocPrefix !== undefined
        ? { docPrefix: args.localEmbedderDocPrefix }
        : {}),
    });
  }
  return new HashEmbedder();
}

function buildReranker(args: Args): Reranker | null {
  if (args.reranker === "none") return null;
  if (args.reranker === "local") {
    return new LocalReranker({
      ...(args.localRerankerModel ? { model: args.localRerankerModel } : {}),
    });
  }
  return new HeuristicReranker();
}

function loadJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function diff(a: EvalReport, b: EvalReport): string {
  const fmt = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(3);
  const out: string[] = [];
  out.push(`Δ NDCG@10   ${fmt(a.aggregate.ndcg10 - b.aggregate.ndcg10)}`);
  out.push(`Δ MRR       ${fmt(a.aggregate.mrr - b.aggregate.mrr)}`);
  out.push(`Δ Recall@10 ${fmt(a.aggregate.recall10 - b.aggregate.recall10)}`);
  out.push("");
  out.push("By strategy (n / Δ NDCG@10):");
  const allStrategies = new Set([...Object.keys(a.byStrategy), ...Object.keys(b.byStrategy)]);
  for (const s of allStrategies) {
    const an = a.byStrategy[s]?.ndcg10 ?? 0;
    const bn = b.byStrategy[s]?.ndcg10 ?? 0;
    const ac = a.byStrategy[s]?.n ?? 0;
    const bc = b.byStrategy[s]?.n ?? 0;
    out.push(`  ${s.padEnd(8)} n: ${bc}→${ac}  ${fmt(an - bn)}`);
  }
  return out.join("\n");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const corpus = loadJson<EvalDoc[]>(join(HERE, "corpus.json"));
  const queries = loadJson<EvalQuery[]>(join(HERE, "queries.json"));

  const baseChunker = new SentenceChunker();
  let chunker: Chunker | SemanticChunker = baseChunker;
  if (args.metadataChunker) {
    chunker = new MetadataChunker({ base: chunker });
  }
  if (args.doc2query) {
    chunker = new Doc2QueryChunker({
      base: chunker,
      ...(args.doc2queryModel ? { model: args.doc2queryModel } : {}),
      ...(args.doc2queryNumQueries !== undefined ? { numQueries: args.doc2queryNumQueries } : {}),
    });
  }
  const embedder = buildEmbedder(args);
  const baseReranker = buildReranker(args);
  // If --mmr is set, cascade [base reranker → MMR] for diverse top-K. The
  // base reranker narrows to a wider pool (50) on relevance; MMR diversifies
  // the survivors down to the caller's topK.
  let reranker: Reranker | null = baseReranker;
  if (args.mmr) {
    const mmr = new MMRReranker({ lambda: args.mmrLambda });
    reranker = baseReranker
      ? new CascadedReranker([
          [baseReranker, 50],
          [mmr, 10],
        ])
      : mmr;
  }
  const adapter = new InMemoryAdapter({ useStemming: args.bm25Stem });

  console.log(
    `Config: embedder=${embedder.name}  chunker=${(chunker as { name: string }).name}  reranker=${reranker ? reranker.name : "none"}  bm25-stem=${args.bm25Stem}` +
      (args.metadataChunker ? "  (metadata-prepend ON)" : "")
  );
  console.log();

  const augur = new Augur({
    embedder,
    chunker,
    adapter,
    ...(reranker ? { reranker } : {}),
  });
  const report = await runEval(augur, corpus, queries);

  console.log(formatReport(report, { verbose: args.verbose }));

  if (args.save) {
    writeFileSync(args.save, JSON.stringify(report, null, 2));
    console.log(`\nSaved report → ${args.save}`);
  }
  if (args.compare) {
    const baseline = loadJson<EvalReport>(args.compare);
    console.log("\nDiff vs baseline:");
    console.log(diff(report, baseline));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
