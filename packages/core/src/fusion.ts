/**
 * Pure helpers for the multi-stage retrieval pipeline.
 *
 * Extracted from `augur.ts` so each function is independently testable
 * without standing up an Augur instance / corpus / reranker. The router
 * decisions (which strategy to pick) and the fusion math (how to combine
 * vector + keyword candidates) are the two places where empirical
 * thresholds live; both deserve unit tests, not just system-level
 * eval-harness coverage.
 *
 * Numerical constants in this file (k=60 RRF smoothing, ±0.20 confidence
 * shift, 0.30 confidence-shift coefficient, 0.10–0.90 weight clamp,
 * 0.3/0.4/0.7 query-signal priors) are tuned on a 504-query / 182-doc
 * eval that's preserved in this repo's git history under the
 * `evaluations/` directory (removed from main in commit feffc73). They
 * are NOT principled — they're empirical. If you change them, run a
 * regression eval first.
 */

import type { QuerySignals, SearchResult } from "./types.js";

/** Clamp a number into [lo, hi]. */
export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/**
 * Merge user-supplied filter with the auto-language tag (if any) into a
 * single AND-style filter. User-supplied keys win on conflict; we don't
 * silently override an explicit `lang` if the caller pinned one.
 *
 *   composeFilter(undefined, "ja")       → { lang: "ja" }
 *   composeFilter({ topic: "k8s" }, "ja") → { lang: "ja", topic: "k8s" }
 *   composeFilter({ lang: "fr" }, "ja")   → { lang: "fr" }   (user wins)
 *   composeFilter(undefined, null)        → undefined
 */
export function composeFilter(
  userFilter: Record<string, unknown> | undefined,
  autoLang: string | null
): Record<string, unknown> | undefined {
  if (!autoLang) return userFilter;
  return userFilter ? { lang: autoLang, ...userFilter } : { lang: autoLang };
}

/**
 * Static prior on the vector/BM25 mix from query signals. Returns the
 * fraction of weight the vector side gets in fusion (BM25 gets 1 - x):
 *   - quoted / specific-token / code-like → 0.3 (lexical wins)
 *   - very short (≤2 tokens) → 0.4 (bi-encoders embed single terms poorly)
 *   - long natural-language question (≥6 tokens) → 0.7 (semantic wins)
 *   - otherwise → 0.5 (no strong signal either way)
 *
 * Thresholds (2, 6) and weights (0.3 / 0.4 / 0.5 / 0.7) are tuned on the
 * preserved eval — they're not principled. The shape (lexical-favored
 * vs. semantic-favored buckets) is the load-bearing part; the exact
 * numbers are negotiable.
 */
export function pickVectorWeight(signals: QuerySignals): number {
  if (signals.hasQuotedPhrase || signals.hasSpecificTokens || signals.hasCodeLike) return 0.3;
  if (signals.wordCount <= 2) return 0.4;
  if (signals.isQuestion && signals.wordCount >= 6) return 0.7;
  return 0.5;
}

/**
 * Reciprocal Rank Fusion of two ranked lists into one with a per-side
 * weight. k=60 is the canonical Cormack-2009 smoothing constant. The
 * weight (`vectorWeight` ∈ [0,1]) lets one side carry more influence
 * than the other — important because production hybrid systems are
 * never symmetric in practice (vector helps on natural-language
 * questions, BM25 helps on identifiers, the right balance is
 * query-dependent). `vectorWeight` is clamped to [0,1] defensively;
 * out-of-range input is silently corrected, not errored.
 */
export function weightedRrfFuse(
  vec: SearchResult[],
  kw: SearchResult[],
  vectorWeight: number,
  k: number = 60
): SearchResult[] {
  const wV = clamp(vectorWeight, 0, 1);
  const wK = 1 - wV;
  const fused = new Map<string, { result: SearchResult; score: number }>();
  vec.forEach((r, rank) => {
    fused.set(r.chunk.id, { result: r, score: wV * (1 / (k + rank + 1)) });
  });
  kw.forEach((r, rank) => {
    const score = wK * (1 / (k + rank + 1));
    const existing = fused.get(r.chunk.id);
    if (existing) existing.score += score;
    else fused.set(r.chunk.id, { result: r, score });
  });
  return Array.from(fused.values())
    .sort((x, y) => y.score - x.score)
    .map(({ result, score }) => ({ ...result, score }));
}

/**
 * Adjust the static (query-signal-derived) vector weight using observed
 * retrieval confidence. The intuition: when one side has a top result
 * that clearly stands out from the rest of its list (large score gap to
 * #2, normalized over the score range), we should trust that side more
 * for this specific query. When both sides look unsure, fall back to
 * the prior.
 *
 * Numbers (all empirically tuned, preserved-eval-derived):
 *   - 0.30 — coefficient that converts confidence-delta into weight-shift.
 *   - ±0.20 — bounded shift so confidence can't fully override the prior;
 *     they vote together rather than the dynamic signal flipping the
 *     query-signal classification.
 *   - [0.10, 0.90] — final weight clamp so neither side gets zeroed out
 *     even if both confidence and prior point hard one way. Preserves
 *     fusion's "second opinion" property.
 */
export function adaptWeightByConfidence(
  baseWeight: number,
  vec: SearchResult[],
  kw: SearchResult[]
): number {
  const vConf = topGapNormalized(vec);
  const kConf = topGapNormalized(kw);
  const shift = clamp((vConf - kConf) * 0.30, -0.20, 0.20);
  return clamp(baseWeight + shift, 0.10, 0.90);
}

/**
 * Confidence proxy: gap from #1 to #2, normalized by the dynamic range
 * of the top-10. A standout #1 → near-1.0; a flat list → near-0. Range-
 * normalizing makes this comparable across BM25 (unbounded scores) and
 * cosine ([-1,1]) score scales without per-retriever calibration.
 *
 * The "bottom of top-10" floor is heuristic — it estimates the noise
 * level of the retriever for this query. If the top-10 is all in a
 * tight band, the noise is low and even a modest gap is meaningful;
 * if the top-10 spans a wide range, you need a bigger absolute gap to
 * count as confidence.
 */
export function topGapNormalized(results: SearchResult[]): number {
  if (results.length < 2) return 0;
  const top = results[0]!.score;
  const second = results[1]!.score;
  const floor = results[Math.min(9, results.length - 1)]!.score;
  const range = top - floor;
  if (range <= 0) return 0;
  return clamp((top - second) / range, 0, 1);
}
