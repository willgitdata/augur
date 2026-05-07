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
  HashEmbedder,
  MetadataChunker,
  SentenceChunker,
  TfIdfEmbedder,
  type Embedder,
} from "../src/index.js";
import { formatReport, runEval, type EvalDoc, type EvalQuery, type EvalReport } from "./runner.js";

const HERE = dirname(fileURLToPath(import.meta.url));

interface Args {
  verbose: boolean;
  save?: string;
  compare?: string;
  embedder: "hash" | "tfidf";
  metadataChunker: boolean;
}

function parseArgs(argv: string[]): Args {
  const out: Args = { verbose: false, embedder: "hash", metadataChunker: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--verbose" || a === "-v") out.verbose = true;
    else if (a === "--save") out.save = argv[++i];
    else if (a === "--compare") out.compare = argv[++i];
    else if (a === "--embedder") {
      const v = argv[++i];
      if (v !== "hash" && v !== "tfidf") {
        throw new Error(`--embedder must be 'hash' or 'tfidf', got ${v}`);
      }
      out.embedder = v;
    } else if (a === "--metadata-chunker") out.metadataChunker = true;
  }
  return out;
}

function buildEmbedder(kind: Args["embedder"]): Embedder {
  if (kind === "tfidf") return new TfIdfEmbedder();
  return new HashEmbedder();
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
  const chunker = args.metadataChunker
    ? new MetadataChunker({ base: baseChunker })
    : baseChunker;
  const embedder = buildEmbedder(args.embedder);

  console.log(
    `Config: embedder=${embedder.name}  chunker=${(chunker as { name: string }).name}` +
      (args.metadataChunker ? "  (metadata-prepend ON)" : "")
  );
  console.log();

  const augur = new Augur({ embedder, chunker });
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
