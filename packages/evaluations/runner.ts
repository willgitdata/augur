/**
 * Eval runner. Indexes a corpus, runs every query, and computes
 * NDCG@10 / MRR / Recall@10 — overall, per category, and per chosen strategy.
 *
 * The runner is deliberately implementation-agnostic: it takes an Augur
 * instance so we can swap the embedder, adapter, router, or reranker
 * between runs and compare. That's how we'll measure whether changes to
 * signals.ts or the reranker actually help.
 */
import type { Augur, Document, RetrievalStrategy } from "@augur-rag/core";
import { mean, ndcgAt, reciprocalRank, recallAt } from "./metrics.js";

export interface EvalDoc extends Document {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface EvalQuery {
  query: string;
  /** Document IDs (not chunk IDs) considered relevant. */
  relevant: string[];
  /** Optional grouping label used for per-category breakdowns. */
  category?: string;
  /** Optional strategy hint — used only for reporting, not graded. */
  expectedStrategy?: RetrievalStrategy;
}

export interface PerQueryResult {
  query: string;
  category?: string;
  strategy: RetrievalStrategy;
  reranked: boolean;
  ndcg10: number;
  mrr: number;
  recall10: number;
  topDocId?: string;
  totalRelevant: number;
}

export interface EvalReport {
  corpus: number;
  queries: number;
  /** Aggregate metrics across all queries with at least one relevant doc. */
  aggregate: { ndcg10: number; mrr: number; recall10: number };
  /** Breakdown grouped by query.category. */
  byCategory: Record<string, { n: number; ndcg10: number; mrr: number; recall10: number }>;
  /** Breakdown grouped by router-chosen strategy. */
  byStrategy: Record<string, { n: number; ndcg10: number; mrr: number; recall10: number }>;
  /** Strategy distribution (counts). */
  strategyCounts: Record<RetrievalStrategy, number>;
  perQuery: PerQueryResult[];
}

const TOP_K = 10;

export async function runEval(
  augur: Augur,
  corpus: EvalDoc[],
  queries: EvalQuery[]
): Promise<EvalReport> {
  // Reset adapter so reruns don't double-index.
  await augur.clear();
  await augur.index(corpus);

  const perQuery: PerQueryResult[] = [];
  const strategyCounts: Record<RetrievalStrategy, number> = {
    vector: 0,
    keyword: 0,
    hybrid: 0,
    rerank: 0,
  };

  for (const q of queries) {
    const { results, trace } = await augur.search({ query: q.query, topK: TOP_K });
    const relevantSet = new Set(q.relevant);
    // Map each retrieved chunk back to its document. Multiple chunks from the
    // same doc count once at the highest-ranked position (standard practice).
    const seenDocs = new Set<string>();
    const docRelevance: number[] = [];
    for (const r of results) {
      const docId = r.chunk.documentId;
      if (seenDocs.has(docId)) continue;
      seenDocs.add(docId);
      docRelevance.push(relevantSet.has(docId) ? 1 : 0);
    }

    const totalRelevant = q.relevant.length;
    const ndcg10 = ndcgAt(docRelevance, totalRelevant, TOP_K);
    const mrr = reciprocalRank(docRelevance);
    const recall10 = recallAt(docRelevance, totalRelevant, TOP_K);

    const strategy = trace.decision.strategy;
    strategyCounts[strategy] = (strategyCounts[strategy] ?? 0) + 1;

    perQuery.push({
      query: q.query,
      ...(q.category !== undefined ? { category: q.category } : {}),
      strategy,
      reranked: trace.decision.reranked,
      ndcg10,
      mrr,
      recall10,
      ...(results[0] !== undefined ? { topDocId: results[0].chunk.documentId } : {}),
      totalRelevant,
    });
  }

  // Aggregate excludes queries with no relevant docs (none in current set,
  // but defensive — and the ratio is undefined for those).
  const scored = perQuery.filter((p) => p.totalRelevant > 0);
  const aggregate = {
    ndcg10: mean(scored.map((p) => p.ndcg10)),
    mrr: mean(scored.map((p) => p.mrr)),
    recall10: mean(scored.map((p) => p.recall10)),
  };

  const byCategory = groupedMetrics(scored, (p) => p.category ?? "uncategorized");
  const byStrategy = groupedMetrics(scored, (p) => p.strategy);

  return {
    corpus: corpus.length,
    queries: queries.length,
    aggregate,
    byCategory,
    byStrategy,
    strategyCounts,
    perQuery,
  };
}

function groupedMetrics(
  rows: PerQueryResult[],
  key: (p: PerQueryResult) => string
): Record<string, { n: number; ndcg10: number; mrr: number; recall10: number }> {
  const buckets = new Map<string, PerQueryResult[]>();
  for (const r of rows) {
    const k = key(r);
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k)!.push(r);
  }
  const out: Record<string, { n: number; ndcg10: number; mrr: number; recall10: number }> = {};
  for (const [k, group] of buckets) {
    out[k] = {
      n: group.length,
      ndcg10: mean(group.map((p) => p.ndcg10)),
      mrr: mean(group.map((p) => p.mrr)),
      recall10: mean(group.map((p) => p.recall10)),
    };
  }
  return out;
}

/** Pretty-print a report to a string. */
export function formatReport(report: EvalReport, opts: { verbose?: boolean } = {}): string {
  const fmt = (n: number) => n.toFixed(3);
  const lines: string[] = [];
  lines.push(`Augur eval — corpus=${report.corpus} docs, queries=${report.queries}`);
  lines.push("");
  lines.push("Aggregate:");
  lines.push(`  NDCG@10   ${fmt(report.aggregate.ndcg10)}`);
  lines.push(`  MRR       ${fmt(report.aggregate.mrr)}`);
  lines.push(`  Recall@10 ${fmt(report.aggregate.recall10)}`);
  lines.push("");
  lines.push("By strategy:");
  for (const [s, m] of Object.entries(report.byStrategy)) {
    lines.push(
      `  ${s.padEnd(8)} n=${String(m.n).padStart(2)}  NDCG@10=${fmt(m.ndcg10)}  MRR=${fmt(m.mrr)}  Recall@10=${fmt(m.recall10)}`
    );
  }
  lines.push("");
  lines.push("By category:");
  for (const [c, m] of Object.entries(report.byCategory)) {
    lines.push(
      `  ${c.padEnd(14)} n=${String(m.n).padStart(2)}  NDCG@10=${fmt(m.ndcg10)}  MRR=${fmt(m.mrr)}  Recall@10=${fmt(m.recall10)}`
    );
  }
  lines.push("");
  lines.push("Strategy distribution:");
  const total = Object.values(report.strategyCounts).reduce((a, b) => a + b, 0);
  for (const [s, c] of Object.entries(report.strategyCounts)) {
    const pct = total === 0 ? 0 : (c / total) * 100;
    lines.push(`  ${s.padEnd(8)} ${String(c).padStart(2)}  (${pct.toFixed(0)}%)`);
  }

  if (opts.verbose) {
    lines.push("");
    lines.push("Per query:");
    for (const r of report.perQuery) {
      const status = r.ndcg10 > 0 ? "✓" : r.totalRelevant === 0 ? "·" : "✗";
      const q = r.query.length > 60 ? r.query.slice(0, 57) + "..." : r.query;
      lines.push(
        `  ${status} ${r.strategy.padEnd(7)}${r.reranked ? "+r " : "   "}NDCG=${fmt(r.ndcg10)} MRR=${fmt(r.mrr)}  ${q}`
      );
    }
  }

  return lines.join("\n");
}
