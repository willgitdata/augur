import { createHash } from "node:crypto";
import { tokenizeAdvanced } from "./text-utils.js";

/**
 * Embedding provider interface.
 *
 * Tradeoff: we keep this very small on purpose. Real-world embedders all share
 * one operation: text-in, vector-out. Anything more complex (caching, batching,
 * rate limiting) is a wrapper concern — see `BatchedEmbedder`.
 *
 * Embedders that benefit from corpus statistics (TF-IDF, learned-IDF, etc) can
 * implement the optional `fit(docs)` method. The orchestrator calls fit() during
 * `index()` so the embedder can update its document-frequency tables before
 * embedding. Embedders that do not need corpus context simply omit fit().
 */
export interface Embedder {
  /** Stable identifier — surfaced in traces. */
  readonly name: string;
  /** Output vector dimension. Must be stable across calls. */
  readonly dimension: number;
  /** Embed a batch of texts. Returns one vector per input, in order. */
  embed(texts: string[]): Promise<number[][]>;
  /**
   * Optional: ingest documents to update embedder state (e.g. document
   * frequencies for TF-IDF). Called by the orchestrator during indexing.
   * No-op by default — embedders without corpus state ignore this.
   */
  fit?(texts: string[]): void;
}

/**
 * HashEmbedder — a deterministic, dependency-free embedder.
 *
 * Why this exists: shipping an SDK that requires an OpenAI key on day one is
 * hostile to developer experience. The HashEmbedder produces stable vectors
 * derived from the hashed feature space of the input. It is *not* semantically
 * meaningful — it's a feature-hashed bag-of-tokens. For real semantic search
 * users plug in `OpenAIEmbedder` or their own implementation.
 *
 * What it's good for:
 * - Local development without API keys
 * - Tests that need deterministic output
 * - The keyword/hybrid pathways still work fine
 *
 * What it's NOT good for:
 * - Real semantic similarity in production
 */
export class HashEmbedder implements Embedder {
  readonly name = "hash-embedder";
  readonly dimension: number;

  constructor(dimension = 384) {
    this.dimension = dimension;
  }

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.embedOne(t));
  }

  private embedOne(text: string): number[] {
    const vec = new Array<number>(this.dimension).fill(0);
    const tokens = tokenize(text);
    for (const token of tokens) {
      // Hash to bucket; sign from second hash byte so values can be ±.
      const h = createHash("sha256").update(token).digest();
      const bucket = h.readUInt32BE(0) % this.dimension;
      const sign = h[4]! & 1 ? 1 : -1;
      vec[bucket] = (vec[bucket] ?? 0) + sign;
    }
    // L2 normalize so cosine == dot product.
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm) || 1;
    return vec.map((v) => v / norm);
  }
}

/**
 * OpenAIEmbedder — uses OpenAI's embeddings API.
 *
 * Implemented with `fetch` directly (no SDK dependency) to keep `@augur/core`
 * dependency-free. Users who want a different provider implement the `Embedder`
 * interface themselves — it's three lines.
 */
export class OpenAIEmbedder implements Embedder {
  readonly name: string;
  readonly dimension: number;
  private apiKey: string;
  private model: string;
  private baseURL: string;

  constructor(opts: {
    apiKey?: string;
    model?: string;
    dimension?: number;
    baseURL?: string;
  } = {}) {
    this.apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    this.model = opts.model ?? "text-embedding-3-small";
    this.dimension = opts.dimension ?? 1536;
    this.baseURL = opts.baseURL ?? "https://api.openai.com/v1";
    this.name = `openai:${this.model}`;
    if (!this.apiKey) {
      throw new Error(
        "OpenAIEmbedder: apiKey not provided and OPENAI_API_KEY is not set"
      );
    }
  }

  async embed(texts: string[]): Promise<number[][]> {
    const res = await fetch(`${this.baseURL}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: texts,
        model: this.model,
        dimensions: this.dimension,
      }),
    });
    if (!res.ok) {
      throw new Error(`OpenAI embeddings failed (${res.status}): ${await res.text()}`);
    }
    const json = (await res.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };
    // Re-sort by index in case API doesn't preserve order.
    return json.data
      .slice()
      .sort((a, b) => a.index - b.index)
      .map((d) => d.embedding);
  }
}

/** Whitespace + punctuation tokenizer, lowercased. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/u)
    .filter((t) => t.length > 0);
}

// Re-export advanced tokenizer + stemmer + stopwords from text-utils for users
// who want to plug them into custom embedders or rerankers.
export { tokenizeAdvanced, stem, STOPWORDS } from "./text-utils.js";

/**
 * TfIdfEmbedder — feature-hashing TF-IDF embedder.
 *
 * Why this is a real upgrade over `HashEmbedder`:
 *   - Stems tokens (running → run, connections → connect) so morphological
 *     variants share a vector dimension.
 *   - Drops English stopwords so common words don't crowd out signal.
 *   - Weights tokens by IDF — rare tokens dominate the vector, common tokens
 *     contribute less. This is what bag-of-words IR has done for 50 years
 *     and remains a strong baseline against modern dense retrievers.
 *   - Sub-linear TF (1 + log tf) so a token appearing 100× doesn't drown
 *     out everything else in the document.
 *
 * Why this is *not* a substitute for a real bi-encoder (BGE, OpenAI, Cohere):
 *   - No semantic understanding — "vehicle" and "car" are unrelated tokens.
 *   - No paraphrase robustness — different phrasings of the same idea miss.
 *   - Quality plateaus around classical TF-IDF; modern dense retrievers add
 *     20-40% NDCG on top of this.
 *
 * Best used as:
 *   - The default embedder for local-only or tests-only deployments.
 *   - A baseline you can A/B against a real embedder via the eval harness.
 *   - The vector half of a hybrid pipeline that leans heavily on BM25 for
 *     exact-match recall and TF-IDF for fuzzy term overlap.
 *
 * Statefulness:
 *   The embedder accumulates document frequencies via `fit(docs)` (called by
 *   `Augur.index()` if available) or implicitly via `embed()` calls (each
 *   batch is treated as new documents and contributes to DFs). For the
 *   cleanest behavior, fit() once on the corpus before embedding queries.
 */
export class TfIdfEmbedder implements Embedder {
  readonly name = "tfidf-embedder";
  readonly dimension: number;

  private dfs = new Map<string, number>();
  private docCount = 0;
  private idfCache: Map<string, number> | null = null;
  private tokenize: (text: string) => string[];

  constructor(opts: { dimension?: number; corpus?: string[]; useStemming?: boolean } = {}) {
    this.dimension = opts.dimension ?? 4096;
    const useStemming = opts.useStemming ?? true;
    this.tokenize = (text) => tokenizeAdvanced(text, { stem: useStemming, dropStopwords: true });
    if (opts.corpus && opts.corpus.length > 0) this.fit(opts.corpus);
  }

  /**
   * Update document-frequency statistics. Call before embedding queries to
   * ensure IDF reflects the indexed corpus.
   */
  fit(texts: string[]): void {
    for (const text of texts) {
      const seen = new Set(this.tokenize(text));
      for (const tok of seen) {
        this.dfs.set(tok, (this.dfs.get(tok) ?? 0) + 1);
      }
      this.docCount += 1;
    }
    this.idfCache = null;
  }

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.embedOne(t));
  }

  private getIdf(): Map<string, number> {
    if (this.idfCache) return this.idfCache;
    const idf = new Map<string, number>();
    const N = Math.max(1, this.docCount);
    for (const [tok, df] of this.dfs) {
      // Smoothed IDF: log((N+1)/(df+1)) + 1 — standard sklearn convention.
      idf.set(tok, Math.log((N + 1) / (df + 1)) + 1);
    }
    this.idfCache = idf;
    return idf;
  }

  private embedOne(text: string): number[] {
    const tokens = this.tokenize(text);
    if (tokens.length === 0) return new Array(this.dimension).fill(0);

    // Term frequencies for this text.
    const tf = new Map<string, number>();
    for (const tok of tokens) tf.set(tok, (tf.get(tok) ?? 0) + 1);

    const idf = this.getIdf();
    const vec = new Array<number>(this.dimension).fill(0);

    for (const [tok, count] of tf) {
      // Sub-linear TF (1 + log count) dampens token-frequency dominance.
      const tfWeight = 1 + Math.log(count);
      // IDF defaults to 1 (the "+1" smoothing baseline) for unseen tokens —
      // unseen-by-the-corpus tokens still contribute, just without IDF boost.
      const idfWeight = idf.get(tok) ?? 1;
      const weight = tfWeight * idfWeight;

      // Feature-hash to a bucket; sign from a separate hash byte so
      // collisions partially cancel (random sign trick).
      const h = createHash("sha256").update(tok).digest();
      const bucket = h.readUInt32BE(0) % this.dimension;
      const sign = h[4]! & 1 ? 1 : -1;
      vec[bucket] = (vec[bucket] ?? 0) + sign * weight;
    }

    // L2 normalize so cosine similarity == dot product.
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm) || 1;
    return vec.map((v) => v / norm);
  }
}

