/**
 * LocalEmbedder — runs sentence-transformer models on-device via
 * `@huggingface/transformers` (ONNX Runtime). Default is
 * `Xenova/all-MiniLM-L6-v2` (22MB, 384d); swap via the `model` option.
 *
 * Mean pooling + L2 normalization (the canonical sentence-transformer
 * config) is baked in. Per-task prefixes (`queryPrefix` / `docPrefix`)
 * support BGE/E5/nomic-style instruction tuning — see docs/examples.md
 * for the model→prefix matrix.
 *
 * The `@huggingface/transformers` dep is dynamically imported, so
 * consumers who don't use this class don't pay the ~100MB footprint.
 * Pipelines are cached per (model, dtype, device) so the ONNX session
 * loads once.
 */

import { BoundedCache } from "../internal/bounded-cache.js";
import {
  loadTransformers,
  type ProgressCallback,
} from "../internal/transformers-loader.js";
import type { Embedder } from "./embedder.js";

export type {
  DownloadProgressEvent,
  ProgressCallback,
} from "../internal/transformers-loader.js";
export { MissingTransformersError } from "../internal/transformers-loader.js";

interface FeatureExtractionPipeline {
  (
    input: string | string[],
    opts: { pooling: "mean" | "cls" | "none"; normalize: boolean }
  ): Promise<{ data: Float32Array; dims: number[] }>;
}

/**
 * ONNX pipelines hold model weights (22-280 MB each) in RSS. An unbounded
 * Map leaked memory in long-running indexers that rotate models (eval
 * matrices, multi-tenant servers). Cap at 4 — well above typical
 * single-model use, low enough that the worst case is ~1 GB resident.
 * Override via `AUGUR_PIPELINE_CACHE_SIZE` for unusual setups.
 */
const PIPELINE_CACHE_DEFAULT_CAPACITY = 4;
const pipelineCache = new BoundedCache<string, Promise<FeatureExtractionPipeline>>(
  parsePositiveInt(process.env.AUGUR_PIPELINE_CACHE_SIZE) ?? PIPELINE_CACHE_DEFAULT_CAPACITY
);

function parsePositiveInt(s: string | undefined): number | null {
  if (!s) return null;
  const n = parseInt(s, 10);
  return Number.isInteger(n) && n > 0 ? n : null;
}

async function getEmbeddingPipeline(
  model: string,
  dtype: string | undefined,
  device: string | undefined,
  onProgress: ProgressCallback | undefined
): Promise<FeatureExtractionPipeline> {
  // Cache key intentionally excludes onProgress — two callers with
  // different callbacks should share the same warmed-up pipeline. The
  // callback only fires during the initial download/load, not on every
  // subsequent inference, so reusing a cached pipeline silently skips
  // progress events for late arrivals (correct: there's no download).
  const key = `${model}|${dtype ?? ""}|${device ?? ""}`;
  const cached = pipelineCache.get(key);
  if (cached) return cached;
  const p = (async (): Promise<FeatureExtractionPipeline> => {
    const transformers = await loadTransformers();
    const opts: {
      dtype?: string;
      device?: string;
      progress_callback?: ProgressCallback;
    } = {};
    if (dtype) opts.dtype = dtype;
    if (device) opts.device = device;
    if (onProgress) opts.progress_callback = onProgress;
    return (await transformers.pipeline(
      "feature-extraction",
      model,
      opts
    )) as FeatureExtractionPipeline;
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
  private onProgress: ProgressCallback | undefined;

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
    /**
     * Called during the one-time model download (and any subsequent
     * cold-start) with `@huggingface/transformers` progress events. Lets
     * you surface a progress line so the first `embed()` call doesn't
     * look like a 5-10 second hang. Not called on warm pipelines (the
     * cached pipeline emits no events). Default: undefined (silent).
     *
     * Quick one-liner that logs % complete to stderr:
     *
     *   new LocalEmbedder({
     *     onProgress: (e) => {
     *       if (e.status === "progress" && typeof e.progress === "number") {
     *         process.stderr.write(`\r[LocalEmbedder] ${e.file ?? ""} ${e.progress.toFixed(0)}%   `);
     *       } else if (e.status === "done") {
     *         process.stderr.write("\n");
     *       }
     *     },
     *   })
     */
    onProgress?: ProgressCallback;
  } = {}) {
    this.model = opts.model ?? "Xenova/all-MiniLM-L6-v2";
    this.dimension = opts.dimension ?? 384;
    this.queryPrefix = opts.queryPrefix ?? "";
    this.docPrefix = opts.docPrefix ?? "";
    this.batchSize = opts.batchSize ?? 32;
    this.dtype = opts.dtype;
    this.device = opts.device;
    this.onProgress = opts.onProgress;
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
    const pipe = await getEmbeddingPipeline(
      this.model,
      this.dtype,
      this.device,
      this.onProgress
    );

    const out: number[][] = [];
    for (let i = 0; i < tagged.length; i += this.batchSize) {
      const batch = tagged.slice(i, i + this.batchSize);
      const result = await pipe(batch, { pooling: "mean", normalize: true });
      const hidden = result.dims[result.dims.length - 1] ?? this.dimension;
      for (let b = 0; b < batch.length; b++) {
        const start = b * hidden;
        out.push(Array.from(result.data.subarray(start, start + hidden)));
      }
    }
    return out;
  }
}
