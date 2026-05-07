import { tokenize } from "../embeddings/embedder.js";
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
 * For real production use, plug in `CohereReranker`, `JinaReranker`, or
 * a self-hosted cross-encoder by implementing the `Reranker` interface.
 */
export class HeuristicReranker implements Reranker {
  readonly name = "heuristic-reranker";

  async rerank(
    query: string,
    results: SearchResult[],
    topK: number
  ): Promise<SearchResult[]> {
    const qTokens = new Set(tokenize(query));
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
 * CohereReranker — uses Cohere's rerank API.
 *
 * Implemented with fetch directly. Users provide an API key explicitly or
 * via COHERE_API_KEY. Defaults to the v3 model.
 */
export class CohereReranker implements Reranker {
  readonly name: string;
  private apiKey: string;
  private model: string;

  constructor(opts: { apiKey?: string; model?: string } = {}) {
    this.apiKey = opts.apiKey ?? process.env.COHERE_API_KEY ?? "";
    this.model = opts.model ?? "rerank-english-v3.0";
    this.name = `cohere:${this.model}`;
    if (!this.apiKey) {
      throw new Error("CohereReranker: apiKey not provided and COHERE_API_KEY is not set");
    }
  }

  async rerank(query: string, results: SearchResult[], topK: number): Promise<SearchResult[]> {
    if (results.length === 0) return [];
    const res = await fetch("https://api.cohere.com/v1/rerank", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.model,
        query,
        documents: results.map((r) => r.chunk.content),
        top_n: topK,
      }),
    });
    if (!res.ok) {
      throw new Error(`Cohere rerank failed (${res.status}): ${await res.text()}`);
    }
    const json = (await res.json()) as {
      results: Array<{ index: number; relevance_score: number }>;
    };
    return json.results.map((r) => {
      const original = results[r.index]!;
      return {
        ...original,
        score: r.relevance_score,
        rawScores: { ...original.rawScores, original: original.score },
      };
    });
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
