/**
 * LocalEmbedder — runs sentence-transformer models entirely on-device via
 * `@huggingface/transformers` (ONNX Runtime under the hood).
 *
 * No API keys, no rate limits, no network at inference time after the
 * one-time model download. The smallest mainstream model (Xenova/
 * all-MiniLM-L6-v2, ~22MB, 384d) is the default; swap to BGE-small,
 * GTE-small, E5-small, or nomic-embed via the `model` option.
 *
 * Best-practice configuration baked in:
 *
 *   1. Mean pooling + L2 normalization (the convention almost all
 *      sentence-transformer models expect at inference time).
 *
 *   2. Per-task prefixes — many strong embedders (BGE, E5, nomic) require
 *      the caller to prepend short instruction-like prefixes to queries
 *      vs documents. The `queryPrefix` and `docPrefix` options drive this.
 *      `embedQuery()` and `embedDocuments()` apply them automatically;
 *      Augur calls those task-specific methods when available, so the
 *      retrieval/indexing roles get tagged correctly without caller code.
 *
 *   3. Dynamic import — the heavy `@huggingface/transformers` dependency
 *      only resolves when this class is actually used. Consumers who don't
 *      touch LocalEmbedder don't need the package installed.
 *
 *   4. Singleton pipeline per (model, embedder-instance) — the model loads
 *      once and is reused across embed() calls. ONNX session reuse is the
 *      single biggest latency win after model selection.
 *
 * Recommended models and their prefixes:
 *
 *   | Model                              | dim  | size  | queryPrefix                                                  | docPrefix          |
 *   |------------------------------------|------|-------|--------------------------------------------------------------|--------------------|
 *   | Xenova/all-MiniLM-L6-v2 (default)  | 384  | 22MB  | (none)                                                       | (none)             |
 *   | Xenova/bge-small-en-v1.5           | 384  | 33MB  | "Represent this sentence for searching relevant passages: "  | (none)             |
 *   | Xenova/gte-small                   | 384  | 33MB  | (none)                                                       | (none)             |
 *   | Xenova/e5-small-v2                 | 384  | 33MB  | "query: "                                                    | "passage: "        |
 *   | nomic-ai/nomic-embed-text-v1.5     | 768  | 137MB | "search_query: "                                             | "search_document: "|
 */

import type { Embedder } from "./embedder.js";

// Cache the pipeline promise per (model, dtype, device) so repeated
// instantiation is free.
const pipelineCache = new Map<string, Promise<unknown>>();

async function getEmbeddingPipeline(
  model: string,
  dtype: string | undefined,
  device: string | undefined
): Promise<unknown> {
  const key = `${model}|${dtype ?? ""}|${device ?? ""}`;
  let p = pipelineCache.get(key);
  if (p) return p;
  p = (async () => {
    // Dynamic import — keeps consumers who don't use this class free of the
    // transformers.js + onnxruntime-node install footprint (~100MB).
    const transformers = (await import("@huggingface/transformers")) as unknown as {
      pipeline: (
        task: string,
        model: string,
        opts?: { dtype?: string; device?: string }
      ) => Promise<unknown>;
    };
    const opts: { dtype?: string; device?: string } = {};
    if (dtype) opts.dtype = dtype;
    if (device) opts.device = device;
    return transformers.pipeline("feature-extraction", model, opts);
  })();
  pipelineCache.set(key, p);
  return p;
}

export class LocalEmbedder implements Embedder {
  readonly name: string;
  readonly dimension: number;
  private model: string;
  private queryPrefix: string;
  private docPrefix: string;
  private batchSize: number;
  private dtype: string | undefined;
  private device: string | undefined;

  constructor(opts: {
    /**
     * HuggingFace repo id (must be available in ONNX format under that
     * namespace). Defaults to Xenova/all-MiniLM-L6-v2.
     */
    model?: string;
    /** Output vector dimension. Should match the model's hidden size. */
    dimension?: number;
    /**
     * Prepended to every query before embedding. BGE-small needs this:
     * "Represent this sentence for searching relevant passages: ".
     */
    queryPrefix?: string;
    /** Prepended to every document. E5/nomic want "passage: " etc. */
    docPrefix?: string;
    /** Texts per pipeline call. Larger = better GPU/CPU utilization. */
    batchSize?: number;
    /**
     * ONNX weight quantization. "fp32" is the canonical/published config
     * (highest quality, biggest model, slowest). "fp16" cuts size and
     * latency ~2× with near-zero quality loss. "q8" / "q4" trade accuracy
     * for ~3-4× speedup — useful for indexing large corpora when the
     * absolute quality ceiling matters less than throughput. Whether a
     * given model has the variant available is a publish-time choice by
     * the model author; transformers.js downloads the matching ONNX file.
     */
    dtype?: "fp32" | "fp16" | "q8" | "q4";
    /** "wasm" (default), "webgpu" if available. Most setups stay on wasm. */
    device?: "wasm" | "webgpu" | "cpu";
  } = {}) {
    this.model = opts.model ?? "Xenova/all-MiniLM-L6-v2";
    this.dimension = opts.dimension ?? 384;
    this.queryPrefix = opts.queryPrefix ?? "";
    this.docPrefix = opts.docPrefix ?? "";
    this.batchSize = opts.batchSize ?? 32;
    this.dtype = opts.dtype;
    this.device = opts.device;
    const tag = [this.dtype, this.device].filter(Boolean).join(",");
    this.name = `local:${this.model}${tag ? `(${tag})` : ""}`;
  }

  async embed(texts: string[]): Promise<number[][]> {
    // Default semantics for the generic embed() entrypoint: treat as documents.
    return this.embedTagged(texts, this.docPrefix);
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return this.embedTagged(texts, this.docPrefix);
  }

  async embedQuery(text: string): Promise<number[]> {
    const [v] = await this.embedTagged([text], this.queryPrefix);
    if (!v) throw new Error("LocalEmbedder: empty embedding response");
    return v;
  }

  private async embedTagged(texts: string[], prefix: string): Promise<number[][]> {
    if (texts.length === 0) return [];
    const tagged = prefix ? texts.map((t) => prefix + t) : texts;
    const pipe = (await getEmbeddingPipeline(this.model, this.dtype, this.device)) as (
      input: string | string[],
      opts: { pooling: "mean" | "cls" | "none"; normalize: boolean }
    ) => Promise<{ data: Float32Array; dims: number[] }>;

    const out: number[][] = [];
    for (let i = 0; i < tagged.length; i += this.batchSize) {
      const batch = tagged.slice(i, i + this.batchSize);
      // pooling=mean + normalize=true is the canonical sentence-transformer config.
      const result = await pipe(batch, { pooling: "mean", normalize: true });
      const hidden = result.dims[result.dims.length - 1] ?? this.dimension;
      const batchCount = batch.length;
      for (let b = 0; b < batchCount; b++) {
        const start = b * hidden;
        out.push(Array.from(result.data.subarray(start, start + hidden)));
      }
    }
    return out;
  }
}
