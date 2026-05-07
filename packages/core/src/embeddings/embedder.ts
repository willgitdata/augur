import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
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
  /**
   * Optional: embed text(s) explicitly tagged as documents-to-be-indexed.
   * Embedders that distinguish doc vs query roles (Gemini's task types,
   * Cohere v3's input_type, BGE's instruct-prefixes) implement this for
   * better retrieval quality. Augur prefers this over `embed()` during
   * `index()` when available.
   */
  embedDocuments?(texts: string[]): Promise<number[][]>;
  /**
   * Optional: embed a single text explicitly tagged as a search query.
   * Augur prefers this over `embed()` during `search()` when available.
   */
  embedQuery?(text: string): Promise<number[]>;
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
 * GeminiEmbedder — Google Gemini embeddings via the Generative Language API.
 *
 * Default model is `gemini-embedding-001` (configurable output dimensionality;
 * we default to 768 for storage/latency, supporting 128/256/512/768/1536/3072).
 * Pass `model: "gemini-embedding-2"` for the newer generation.
 *
 * Why this matters: Gemini's bi-encoder embeddings are real semantic vectors.
 * Pairing with task-type tagging (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)
 * gives noticeably better retrieval than treating both roles symmetrically —
 * the encoder produces different-but-aligned spaces tuned for each role.
 *
 * Auth: pass `apiKey`, or set the `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
 * environment variable. Never check the key into source.
 *
 * Batching: the underlying `:batchEmbedContents` endpoint accepts up to 100
 * inputs per request. Larger batches are split client-side automatically.
 */
export class GeminiEmbedder implements Embedder {
  readonly name: string;
  readonly dimension: number;
  private apiKey: string;
  private model: string;
  private baseURL: string;
  private batchSize: number;

  private maxRetries: number;
  private throttleMs: number;
  private cacheDir: string | null;

  constructor(opts: {
    apiKey?: string;
    model?: string;
    /** Output vector size. Only meaningful for models that accept output_dimensionality. */
    dimension?: number;
    /** Max items per batch request. Defaults to 100 (the API ceiling). */
    batchSize?: number;
    /** Override for proxies / on-prem. Defaults to public Generative Language API. */
    baseURL?: string;
    /** Retries on 429/5xx with exponential backoff. Defaults to 5. */
    maxRetries?: number;
    /** Min delay between requests in ms (free-tier safety). Defaults to 0. */
    throttleMs?: number;
    /**
     * On-disk cache directory keyed by sha256(model|dim|taskType|text). Highly
     * recommended in production — embedding cost is non-trivial and texts
     * rarely change. Defaults to null (no caching).
     */
    cacheDir?: string | null;
  } = {}) {
    this.apiKey =
      opts.apiKey ??
      process.env.GEMINI_API_KEY ??
      process.env.GOOGLE_API_KEY ??
      "";
    this.model = opts.model ?? "gemini-embedding-001";
    this.dimension = opts.dimension ?? 768;
    this.baseURL = opts.baseURL ?? "https://generativelanguage.googleapis.com/v1beta";
    this.batchSize = opts.batchSize ?? 100;
    this.maxRetries = opts.maxRetries ?? 5;
    this.throttleMs = opts.throttleMs ?? 0;
    this.cacheDir = opts.cacheDir ?? null;
    if (this.cacheDir && !existsSync(this.cacheDir)) {
      mkdirSync(this.cacheDir, { recursive: true });
    }
    this.name = `gemini:${this.model}`;
    if (!this.apiKey) {
      throw new Error(
        "GeminiEmbedder: apiKey not provided and GEMINI_API_KEY / GOOGLE_API_KEY not set"
      );
    }
  }

  /** Default to the document task type — typical use site is index-time embedding. */
  async embed(texts: string[]): Promise<number[][]> {
    return this.embedBatch(texts, "RETRIEVAL_DOCUMENT");
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return this.embedBatch(texts, "RETRIEVAL_DOCUMENT");
  }

  async embedQuery(text: string): Promise<number[]> {
    const [v] = await this.embedBatch([text], "RETRIEVAL_QUERY");
    if (!v) throw new Error("GeminiEmbedder: empty embedding response");
    return v;
  }

  private async embedBatch(
    texts: string[],
    taskType: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY"
  ): Promise<number[][]> {
    if (texts.length === 0) return [];

    // Check cache first — only un-cached texts make API calls.
    const out = new Array<number[] | null>(texts.length).fill(null);
    const missing: Array<{ index: number; text: string }> = [];
    if (this.cacheDir) {
      for (let i = 0; i < texts.length; i++) {
        const cached = this.readCache(texts[i]!, taskType);
        if (cached) out[i] = cached;
        else missing.push({ index: i, text: texts[i]! });
      }
    } else {
      for (let i = 0; i < texts.length; i++) missing.push({ index: i, text: texts[i]! });
    }
    if (missing.length === 0) return out as number[][];

    // Embed only the misses, then merge back into the result array.
    const batchTexts = missing.map((m) => m.text);
    const fresh = await this.embedBatchUncached(batchTexts, taskType);
    for (let j = 0; j < missing.length; j++) {
      const vec = fresh[j]!;
      out[missing[j]!.index] = vec;
      if (this.cacheDir) this.writeCache(missing[j]!.text, taskType, vec);
    }
    return out as number[][];
  }

  private async embedBatchUncached(
    texts: string[],
    taskType: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY"
  ): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      // gemini-embedding-* models accept outputDimensionality (128 / 256 / 512
      // / 768 / 1536 / 3072). Older text-embedding-* models ignore it.
      const body = {
        requests: batch.map((text) => {
          const req: Record<string, unknown> = {
            model: `models/${this.model}`,
            content: { parts: [{ text }] },
            taskType,
          };
          if (this.model.startsWith("gemini-embedding-")) {
            req["outputDimensionality"] = this.dimension;
          }
          return req;
        }),
      };
      const url = `${this.baseURL}/models/${this.model}:batchEmbedContents?key=${encodeURIComponent(
        this.apiKey
      )}`;
      const json = await this.fetchWithRetry(url, body);
      for (const e of json.embeddings) out.push(this.normalize(e.values));
      if (this.throttleMs > 0 && i + this.batchSize < texts.length) {
        await sleep(this.throttleMs);
      }
    }
    return out;
  }

  /**
   * Retry on 429 (rate limit) and 5xx with exponential backoff. Honors a
   * `Retry-After` header when the server provides one. Avoids echoing the
   * API key into thrown error messages.
   */
  private async fetchWithRetry(
    url: string,
    body: unknown
  ): Promise<{ embeddings: Array<{ values: number[] }> }> {
    let attempt = 0;
    let lastErr: Error | null = null;
    while (attempt <= this.maxRetries) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        return (await res.json()) as { embeddings: Array<{ values: number[] }> };
      }
      const retriable = res.status === 429 || (res.status >= 500 && res.status < 600);
      if (!retriable || attempt === this.maxRetries) {
        throw new Error(`Gemini embed failed (${res.status} ${res.statusText})`);
      }
      const retryAfter = parseRetryAfter(res.headers.get("retry-after"));
      const backoff = retryAfter ?? Math.min(60_000, 1_000 * Math.pow(2, attempt));
      lastErr = new Error(
        `Gemini ${res.status}; retrying in ${backoff}ms (attempt ${attempt + 1}/${this.maxRetries})`
      );
      // eslint-disable-next-line no-console
      console.warn(`[gemini] ${lastErr.message}`);
      await sleep(backoff + Math.floor(Math.random() * 200));
      attempt += 1;
    }
    throw lastErr ?? new Error("Gemini embed failed after retries");
  }

  /**
   * L2-normalize. Gemini returns normalized vectors only at the native 3072
   * dimension; for any smaller `outputDimensionality` the docs explicitly say
   * to normalize client-side. We always normalize so cosine == dot product
   * downstream.
   */
  private normalize(vec: number[]): number[] {
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm) || 1;
    return vec.map((v) => v / norm);
  }

  private cachePath(text: string, taskType: string): string | null {
    if (!this.cacheDir) return null;
    const key = createHash("sha256")
      .update(`${this.model}|${this.dimension}|${taskType}|${text}`)
      .digest("hex");
    return join(this.cacheDir, `${key}.json`);
  }

  private readCache(text: string, taskType: string): number[] | null {
    const path = this.cachePath(text, taskType);
    if (!path || !existsSync(path)) return null;
    try {
      return JSON.parse(readFileSync(path, "utf8")) as number[];
    } catch {
      return null;
    }
  }

  private writeCache(text: string, taskType: string, vec: number[]): void {
    const path = this.cachePath(text, taskType);
    if (!path) return;
    writeFileSync(path, JSON.stringify(vec));
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Parse a `Retry-After` header value into milliseconds. Supports both
 * delta-seconds and HTTP-date formats. Returns null if absent/unparseable.
 */
function parseRetryAfter(value: string | null): number | null {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
  return null;
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

