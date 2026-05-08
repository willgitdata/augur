import type { Reranker } from "./reranker.js";
import type { SearchResult } from "../types.js";
import { tokenizeAdvanced } from "../embeddings/text-utils.js";

/**
 * MMRReranker — Maximal Marginal Relevance reranking for diversity.
 *
 * Carbonell & Goldstein (1998), still the standard "make top-K diverse"
 * pattern in production retrieval. The score for selecting the next doc
 * is:
 *
 *   MMR(d) = λ · score(d) − (1 − λ) · max_{d' in selected} sim(d, d')
 *
 * λ = 1.0 → pure relevance (input order).
 * λ = 0.0 → pure novelty (most-different first).
 * λ = 0.7 (default) → relevance-weighted with a strong diversity bonus.
 *
 * Why include this:
 *   - Ambiguous / multi-aspect queries often have several distinct
 *     relevant docs. A pure-relevance reranker concentrates on near-
 *     duplicates, leaving the user with three flavors of the same
 *     answer instead of three answers.
 *   - On the bundled eval, the `ambiguous` category has the lowest
 *     NDCG@10 across all configs — exactly because relevant docs are
 *     spread across multiple sub-topics. MMR re-orders so different
 *     sub-topics surface in the top-K.
 *
 * Stack with another reranker via `CascadedReranker`:
 *   `[LocalReranker, 50]` → `[MMRReranker, topK]`. The cross-encoder
 *   narrows on relevance, then MMR diversifies the survivors.
 *
 * Doc-doc similarity:
 *   Default uses a Jaccard overlap on stemmed-and-stopworded tokens —
 *   no embeddings needed, runs in O(n²) over the candidates (fine for
 *   ~50 candidates). Pass a custom `similarity` function for
 *   embedding-based or other measures.
 */
export class MMRReranker implements Reranker {
  readonly name: string;
  private lambda: number;
  private similarity: (a: SearchResult, b: SearchResult) => number;

  constructor(opts: {
    /** 0 = pure novelty, 1 = pure relevance. Default 0.7. */
    lambda?: number;
    /** Override the default Jaccard-on-stems similarity. */
    similarity?: (a: SearchResult, b: SearchResult) => number;
  } = {}) {
    this.lambda = opts.lambda ?? 0.7;
    this.similarity = opts.similarity ?? defaultJaccardSimilarity;
    this.name = `mmr(λ=${this.lambda})`;
  }

  async rerank(
    _query: string,
    results: SearchResult[],
    topK: number
  ): Promise<SearchResult[]> {
    if (results.length === 0) return [];
    const k = Math.min(topK, results.length);

    // Pre-compute pairwise similarities lazily as we go (stale once we cross
    // ~200 candidates; fine at our scale).
    const simCache = new Map<string, number>();
    const pairSim = (i: number, j: number): number => {
      const key = i < j ? `${i}|${j}` : `${j}|${i}`;
      let s = simCache.get(key);
      if (s === undefined) {
        s = this.similarity(results[i]!, results[j]!);
        simCache.set(key, s);
      }
      return s;
    };

    const selected: number[] = [];
    const remaining = new Set<number>();
    for (let i = 0; i < results.length; i++) remaining.add(i);

    while (selected.length < k && remaining.size > 0) {
      let bestIdx = -1;
      let bestScore = -Infinity;
      for (const i of remaining) {
        let maxSim = 0;
        for (const j of selected) {
          const s = pairSim(i, j);
          if (s > maxSim) maxSim = s;
        }
        const mmr = this.lambda * results[i]!.score - (1 - this.lambda) * maxSim;
        if (mmr > bestScore) {
          bestScore = mmr;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      selected.push(bestIdx);
      remaining.delete(bestIdx);
    }

    // Preserve the input scores (not the MMR meta-scores) so downstream code
    // sees calibrated relevance numbers; the new order is the only diff.
    return selected.map((i) => results[i]!);
  }
}

/** Token-level Jaccard on stemmed, stopword-filtered content. */
function defaultJaccardSimilarity(a: SearchResult, b: SearchResult): number {
  const ta = new Set(tokenizeAdvanced(a.chunk.content, { stem: true, dropStopwords: true }));
  const tb = new Set(tokenizeAdvanced(b.chunk.content, { stem: true, dropStopwords: true }));
  if (ta.size === 0 && tb.size === 0) return 1;
  let intersect = 0;
  for (const t of ta) if (tb.has(t)) intersect += 1;
  const union = ta.size + tb.size - intersect;
  return union === 0 ? 0 : intersect / union;
}
