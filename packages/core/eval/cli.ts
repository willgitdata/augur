/**
 * Eval CLI. Loads corpus + queries from disk, runs the default Augur pipeline
 * (in-memory adapter, hash embedder, sentence chunker, heuristic router +
 * reranker), and prints aggregate + per-strategy + per-category metrics.
 *
 * Usage:
 *   pnpm --filter @augur/core eval                     # run with defaults
 *   pnpm --filter @augur/core eval -- --verbose        # per-query lines
 *   pnpm --filter @augur/core eval -- --save out.json  # write metrics JSON
 *   pnpm --filter @augur/core eval -- --compare baseline.json  # diff
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Augur } from "../src/index.js";
import { formatReport, runEval, type EvalDoc, type EvalQuery, type EvalReport } from "./runner.js";

const HERE = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv: string[]): { verbose: boolean; save?: string; compare?: string } {
  const out: { verbose: boolean; save?: string; compare?: string } = { verbose: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--verbose" || a === "-v") out.verbose = true;
    else if (a === "--save") out.save = argv[++i];
    else if (a === "--compare") out.compare = argv[++i];
  }
  return out;
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

  const augur = new Augur();
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
