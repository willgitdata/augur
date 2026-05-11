/**
 * LocalEmbedder — runs sentence-transformer models on-device via
 * `@huggingface/transformers` (ONNX Runtime). Default is
 * `Xenova/all-MiniLM-L6-v2` (22MB, 384d); swap via the `model` option.
 *
 * Mean pooling + L2 normalization (the canonical sentence-transformer
 * config) is baked in. Per-task prefixes (`queryPrefix` / `docPrefix`)
 * support BGE/E5/nomic-style instruction tuning — see docs/examples.md §5
 * for the model→prefix matrix.
 *
 * The `@huggingface/transformers` dep is dynamically imported, so
 * consumers who don't use this class don't pay the ~100MB footprint.
 * Pipelines are cached per (model, dtype, device) so the ONNX session
 * loads once.
 */

import type { Embedder } from "./embedder.js";

interface FeatureExtractionPipeline {
  (
    input: string | string[],
    opts: { pooling: "mean" | "cls" | "none"; normalize: boolean }
  ): Promise<{ data: Float32Array; dims: number[] }>;
}

interface TransformersModule {
  pipeline: (
    task: string,
    model: string,
    opts?: { dtype?: string; device?: string }
  ) => Promise<FeatureExtractionPipeline>;
}

const pipelineCache = new Map<string, Promise<FeatureExtractionPipeline>>();

async function getEmbeddingPipeline(
  model: string,
  dtype: string | undefined,
  device: string | undefined
): Promise<FeatureExtractionPipeline> {
  const key = `${model}|${dtype ?? ""}|${device ?? ""}`;
  let p = pipelineCache.get(key);
  if (p) return p;
  p = (async (): Promise<FeatureExtractionPipeline> => {
    const transformers = (await import("@huggingface/transformers")) as unknown as TransformersModule;
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
    const pipe = await getEmbeddingPipeline(this.model, this.dtype, this.device);

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
