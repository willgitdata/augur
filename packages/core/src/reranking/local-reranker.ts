/**
 * LocalReranker — runs a cross-encoder reranker entirely on-device via
 * `@huggingface/transformers`.
 *
 * The default model is `Xenova/ms-marco-MiniLM-L-6-v2` (~22MB), the
 * standard small cross-encoder trained on MS-MARCO. It scores (query,
 * passage) pairs directly — much higher precision than a bi-encoder
 * cosine because the model attends across the boundary.
 *
 * Best practice: pair this with a bi-encoder retrieval pipeline. The
 * bi-encoder (e.g. `LocalEmbedder`) does broad recall over the whole
 * corpus; the cross-encoder reranker scores only the top ~50 candidates.
 * That cascade gives near-cross-encoder precision at near-bi-encoder
 * cost.
 *
 * Heavier alternatives (configurable via the `model` option):
 *
 *   - `Xenova/ms-marco-MiniLM-L-12-v2`   — 33MB,  +small NDCG bump
 *   - `Xenova/bge-reranker-base`         — 280MB, top-tier accuracy
 *   - `Xenova/jina-reranker-v1-tiny-en`  — 33MB,  Jina's tiny variant
 *
 * Dynamic import keeps `@huggingface/transformers` out of the install
 * footprint for consumers that don't use this class.
 */

import { BoundedCache } from "../internal/bounded-cache.js";
import {
  loadTransformers,
  type ProgressCallback,
} from "../internal/transformers-loader.js";
import type { Reranker } from "./reranker.js";
import type { SearchResult } from "../types.js";

export type {
  DownloadProgressEvent,
  ProgressCallback,
} from "../internal/transformers-loader.js";
export { MissingTransformersError } from "../internal/transformers-loader.js";

/**
 * Cross-encoder pipelines are 22-280 MB each (ms-marco-MiniLM-L-6-v2 to
 * bge-reranker-base). Bounded LRU prevents long-running rerank-eval loops
 * from leaking memory. Honors the same env var as the embedder cache.
 */
const PIPELINE_CACHE_DEFAULT_CAPACITY = 4;
const pipelineCache = new BoundedCache<string, Promise<unknown>>(
  parsePositiveInt(process.env.AUGUR_PIPELINE_CACHE_SIZE) ?? PIPELINE_CACHE_DEFAULT_CAPACITY
);

function parsePositiveInt(s: string | undefined): number | null {
  if (!s) return null;
  const n = parseInt(s, 10);
  return Number.isInteger(n) && n > 0 ? n : null;
}

async function getRerankPipeline(
  model: string,
  onProgress: ProgressCallback | undefined
): Promise<unknown> {
  const cached = pipelineCache.get(model);
  if (cached) return cached;
  const p = (async () => {
    const transformers = await loadTransformers();
    // text-classification with a (text, text_pair) input runs the cross-encoder
    // and returns its logit/score for the pair.
    const opts: { progress_callback?: ProgressCallback } = {};
    if (onProgress) opts.progress_callback = onProgress;
    return transformers.pipeline("text-classification", model, opts);
  })();
  pipelineCache.set(model, p);
  return p;
}

export class LocalReranker implements Reranker {
  readonly name: string;
  private model: string;
  private batchSize: number;
  private applySigmoid: boolean;
  private onProgress: ProgressCallback | undefined;

  constructor(opts: {
    model?: string;
    batchSize?: number;
    /**
     * Apply sigmoid to raw cross-encoder logits to get calibrated [0,1]
     * scores. Default true — calibrated scores compose better with MMR,
     * threshold-based filtering, and score-weighted hybrid fusion. Pass
     * `false` if you specifically want raw logits.
     */
    applySigmoid?: boolean;
    /**
     * Called during the one-time model download (and any subsequent
     * cold-start) with `@huggingface/transformers` progress events. Lets
     * you surface a progress line so the first `rerank()` call doesn't
     * look like a 5-10 second hang. Default: undefined (silent).
     * See `LocalEmbedder` for a one-liner stderr example.
     */
    onProgress?: ProgressCallback;
  } = {}) {
    this.model = opts.model ?? "Xenova/ms-marco-MiniLM-L-6-v2";
    this.batchSize = opts.batchSize ?? 16;
    this.applySigmoid = opts.applySigmoid ?? true;
    this.onProgress = opts.onProgress;
    this.name = `local-reranker:${this.model}`;
  }

  async rerank(
    query: string,
    results: SearchResult[],
    topK: number
  ): Promise<SearchResult[]> {
    if (results.length === 0) return [];

    const pipe = (await getRerankPipeline(this.model, this.onProgress)) as (
      input: Array<{ text: string; text_pair: string }>,
      opts: { topk: number; function_to_apply: "none" | "sigmoid" | "softmax" }
    ) => Promise<Array<{ label: string; score: number }>>;

    const scores = new Array<number>(results.length).fill(0);
    for (let i = 0; i < results.length; i += this.batchSize) {
      const batch = results.slice(i, i + this.batchSize);
      const inputs = batch.map((r) => ({ text: query, text_pair: r.chunk.content }));
      // ms-marco-MiniLM and most cross-encoders output a single relevance
      // logit. topk: 1 + function_to_apply: "none" returns the raw score.
      const out = await pipe(inputs, { topk: 1, function_to_apply: "none" });
      for (let b = 0; b < batch.length; b++) {
        const raw = out[b]?.score ?? 0;
        scores[i + b] = this.applySigmoid ? sigmoid(raw) : raw;
      }
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

function sigmoid(x: number): number {
  // Stable for both positive and negative inputs.
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}
