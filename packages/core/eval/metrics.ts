/**
 * IR metrics for the eval harness. Binary relevance: a doc is either
 * relevant (1) or not (0). Graded relevance is a possible extension; the
 * NDCG implementation already supports it via `gain()`.
 */

/** Discounted Cumulative Gain at k. With binary relevance, gain = rel. */
export function dcgAt(relevance: number[], k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, relevance.length); i++) {
    const rel = relevance[i] ?? 0;
    // 2^rel - 1 form generalizes to graded relevance; equals rel when rel ∈ {0,1}.
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  return dcg;
}

/**
 * Normalized DCG at k. Returns 0 for queries with no relevant docs (the
 * IDCG denominator is 0 — those queries should typically be skipped, but
 * we return 0 so the caller can decide).
 */
export function ndcgAt(retrievedRelevance: number[], totalRelevant: number, k: number): number {
  if (totalRelevant === 0) return 0;
  const dcg = dcgAt(retrievedRelevance, k);
  // Ideal: as many 1s as totalRelevant, capped at k.
  const ideal = Array.from({ length: Math.min(totalRelevant, k) }, () => 1);
  const idcg = dcgAt(ideal, k);
  return idcg === 0 ? 0 : dcg / idcg;
}

/** Reciprocal rank of the first relevant doc (1-indexed). 0 if none. */
export function reciprocalRank(retrievedRelevance: number[]): number {
  for (let i = 0; i < retrievedRelevance.length; i++) {
    if ((retrievedRelevance[i] ?? 0) > 0) return 1 / (i + 1);
  }
  return 0;
}

/** Recall at k: fraction of all relevant docs found in the top-k. */
export function recallAt(retrievedRelevance: number[], totalRelevant: number, k: number): number {
  if (totalRelevant === 0) return 0;
  const hits = retrievedRelevance.slice(0, k).reduce((s, r) => s + (r > 0 ? 1 : 0), 0);
  return hits / totalRelevant;
}

/** Average a list of numbers; returns 0 for empty input. */
export function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}
