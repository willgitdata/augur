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

/**
 * JinaReranker — uses Jina AI's rerank API.
 *
 * Jina's rerank-v2-base-multilingual handles non-English well, which is the
 * single biggest weakness of the default heuristic + hash-embedder pipeline.
 * Set JINA_API_KEY in the environment or pass `apiKey` directly.
 */
export class JinaReranker implements Reranker {
  readonly name: string;
  private apiKey: string;
  private model: string;
  private endpoint: string;

  constructor(opts: { apiKey?: string; model?: string; endpoint?: string } = {}) {
    this.apiKey = opts.apiKey ?? process.env.JINA_API_KEY ?? "";
    this.model = opts.model ?? "jina-reranker-v2-base-multilingual";
    this.endpoint = opts.endpoint ?? "https://api.jina.ai/v1/rerank";
    this.name = `jina:${this.model}`;
    if (!this.apiKey) {
      throw new Error("JinaReranker: apiKey not provided and JINA_API_KEY is not set");
    }
  }

  async rerank(query: string, results: SearchResult[], topK: number): Promise<SearchResult[]> {
    if (results.length === 0) return [];
    const res = await fetch(this.endpoint, {
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
      throw new Error(`Jina rerank failed (${res.status}): ${await res.text()}`);
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
 *     [new CohereReranker(), 10],       // expensive, narrow
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

/**
 * HttpCrossEncoderReranker — generic adapter for any cross-encoder hosted
 * behind an HTTP endpoint. Bring-your-own request and response shape.
 *
 * Useful for self-hosted BGE / mxbai / mixedbread rerankers, internal
 * scoring services, or anything that doesn't match the Cohere/Jina shape.
 *
 * Default protocol (when `requestBody` and `parseResponse` aren't supplied):
 *   POST { query: string, documents: string[], topK: number }
 *   ←   { scores: number[] }   // one score per document, same order
 *
 * The reranker re-orders the input results by the returned scores, takes the
 * top `topK`, and stamps the new score onto each result.
 */
export class HttpCrossEncoderReranker implements Reranker {
  readonly name: string;
  private endpoint: string;
  private headers: Record<string, string>;
  private requestBody: (query: string, docs: string[], topK: number) => unknown;
  private parseResponse: (raw: unknown, docCount: number) => number[];

  constructor(opts: {
    endpoint: string;
    name?: string;
    headers?: Record<string, string>;
    requestBody?: (query: string, docs: string[], topK: number) => unknown;
    parseResponse?: (raw: unknown, docCount: number) => number[];
  }) {
    this.endpoint = opts.endpoint;
    this.name = opts.name ?? `http-cross-encoder:${new URL(opts.endpoint).hostname}`;
    this.headers = { "Content-Type": "application/json", ...(opts.headers ?? {}) };
    this.requestBody =
      opts.requestBody ?? ((query, docs, topK) => ({ query, documents: docs, topK }));
    this.parseResponse =
      opts.parseResponse ??
      ((raw) => {
        const r = raw as { scores?: number[] };
        if (!Array.isArray(r.scores)) {
          throw new Error("HttpCrossEncoderReranker: response missing 'scores' array");
        }
        return r.scores;
      });
  }

  async rerank(query: string, results: SearchResult[], topK: number): Promise<SearchResult[]> {
    if (results.length === 0) return [];
    const docs = results.map((r) => r.chunk.content);
    const res = await fetch(this.endpoint, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(this.requestBody(query, docs, topK)),
    });
    if (!res.ok) {
      throw new Error(`${this.name} failed (${res.status}): ${await res.text()}`);
    }
    const scores = this.parseResponse(await res.json(), docs.length);
    if (scores.length !== docs.length) {
      throw new Error(
        `${this.name}: expected ${docs.length} scores, got ${scores.length}`
      );
    }
    const rescored = results.map((r, i) => ({
      ...r,
      score: scores[i] ?? 0,
      rawScores: { ...r.rawScores, original: r.score },
    }));
    rescored.sort((a, b) => b.score - a.score);
    return rescored.slice(0, topK);
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
