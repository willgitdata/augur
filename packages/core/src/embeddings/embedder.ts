import { createHash } from "node:crypto";

/**
 * Embedding provider interface.
 *
 * Tradeoff: we keep this very small on purpose. Real-world embedders all share
 * one operation: text-in, vector-out. Anything more complex (caching, batching,
 * rate limiting) is a wrapper concern — see `BatchedEmbedder`.
 */
export interface Embedder {
  /** Stable identifier — surfaced in traces. */
  readonly name: string;
  /** Output vector dimension. Must be stable across calls. */
  readonly dimension: number;
  /** Embed a batch of texts. Returns one vector per input, in order. */
  embed(texts: string[]): Promise<number[][]>;
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
