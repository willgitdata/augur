import { tokenize } from "../embeddings/embedder.js";
import { STOPWORDS } from "../embeddings/text-utils.js";
import type { SearchResult } from "../types.js";

/**
 * Reranker interface — takes the top-N candidates from initial retrieval and
 * re-scores them with a more expensive model.
 *
 * This is the standard "retrieve then rerank" pattern. Why it works:
 * - Initial retrieval is cheap and recall-oriented (get a wide net).
 * - Reranking is expensive and precision-oriented (resort the net).
 * - 1 expensive call over 50 candidates beats 50 expensive calls over the corpus.
 */
export interface Reranker {
  readonly name: string;
  rerank(query: string, results: SearchResult[], topK: number): Promise<SearchResult[]>;
}

/**
 * HeuristicReranker — re-orders by combining the original score with simple
 * lexical overlap features. Zero dependencies, sub-millisecond per candidate.
 *
 * This is *not* a state-of-the-art reranker. It's a useful baseline that:
 *   - boosts results with high token-overlap to the query (a partial fix
 *     for embedding-only "missed the keyword" failures)
 *   - down-weights results where the query terms appear far apart in the chunk
 *
 * For real production use, plug in `LocalReranker` (on-device ONNX) or
 * implement the `Reranker` interface against any hosted cross-encoder.
 */
export class HeuristicReranker implements Reranker {
  readonly name = "heuristic-reranker";

  async rerank(
    query: string,
    results: SearchResult[],
    topK: number
  ): Promise<SearchResult[]> {
    // Drop stopwords + short tokens from the query side. Without this,
    // single-letter tokens like "i" in "How do I configure …" match every
    // for-loop variable in code chunks and inflate proximity / overlap.
    const qTokens = new Set(
      tokenize(query).filter((t) => t.length >= 3 && !STOPWORDS.has(t))
    );
    if (qTokens.size === 0) return results.slice(0, topK);

    const rescored = results.map((r) => {
      const cTokens = tokenize(r.chunk.content);
      const cTokSet = new Set(cTokens);
      let overlap = 0;
      for (const t of qTokens) if (cTokSet.has(t)) overlap += 1;
      const overlapRatio = overlap / qTokens.size;

      // Proximity: smallest window containing the most query tokens.
      const proximity = computeProximity(cTokens, qTokens);

      // Final = 0.5 * original + 0.3 * overlap + 0.2 * proximity.
      const rerankScore =
        0.5 * normalize(r.score) + 0.3 * overlapRatio + 0.2 * proximity;
      return {
        ...r,
        score: rerankScore,
        rawScores: {
          ...r.rawScores,
          original: r.score,
          overlap: overlapRatio,
          proximity,
        },
      };
    });
    rescored.sort((a, b) => b.score - a.score);
    return rescored.slice(0, topK);
  }
}


/**
 * CascadedReranker — chain N rerankers, each narrowing the candidate set.
 *
 * The standard production pattern: a cheap first-pass narrows from 1000
 * candidates to 100, an expensive cross-encoder narrows from 100 to 10.
 * Each stage trades latency for precision; stacking them gets you both.
 *
 * Usage:
 *   const reranker = new CascadedReranker([
 *     [new HeuristicReranker(), 100],   // cheap, broad
 *     [new LocalReranker(), 10],       // expensive, narrow
 *   ]);
 *
 * Each tuple is [reranker, topK_for_that_stage]. The final stage's topK
 * is what the caller asked for; intermediate stages should pass more.
 */
export class CascadedReranker implements Reranker {
  readonly name: string;
  private stages: Array<[Reranker, number]>;

  constructor(stages: Array<[Reranker, number]>) {
    if (stages.length === 0) {
      throw new Error("CascadedReranker requires at least one stage");
    }
    this.stages = stages;
    this.name = `cascade(${stages.map(([r]) => r.name).join("→")})`;
  }

  async rerank(
    query: string,
    results: SearchResult[],
    topK: number
  ): Promise<SearchResult[]> {
    let current = results;
    for (let i = 0; i < this.stages.length; i++) {
      const [reranker, stageTopK] = this.stages[i]!;
      // Final stage uses the caller's topK; earlier stages use their declared topK.
      const k = i === this.stages.length - 1 ? topK : stageTopK;
      current = await reranker.rerank(query, current, k);
    }
    return current;
  }
}

/** Squash arbitrary scores into [0,1] using a stable sigmoid. */
function normalize(score: number): number {
  return 1 / (1 + Math.exp(-score));
}

/** Heuristic proximity: 1 if all query tokens fit in a small window, → 0 otherwise. */
function computeProximity(chunkTokens: string[], qTokens: Set<string>): number {
  const positions: number[] = [];
  for (let i = 0; i < chunkTokens.length; i++) {
    if (qTokens.has(chunkTokens[i]!)) positions.push(i);
  }
  if (positions.length < 2) return positions.length === 0 ? 0 : 0.5;
  const window = positions[positions.length - 1]! - positions[0]!;
  // Reward windows < 30 tokens, smoothly decay to ~0 at 200.
  return Math.max(0, 1 - window / 200);
}
