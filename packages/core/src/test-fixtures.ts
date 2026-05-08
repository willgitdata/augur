/**
 * Shared test fixtures.
 *
 * Excluded from the published package via the tsconfig `exclude`
 * patterns. Imported by tests, never by the runtime. Real ONNX models
 * are too heavy for a unit test; the stub here is enough to exercise
 * routing, fusion, and trace plumbing on hashed bag-of-tokens vectors.
 *
 * If you need a different stub shape for a specific test, prefer adding
 * a new export here over inlining a third copy.
 */

import type { Embedder } from "./embeddings/embedder.js";
import { tokenize } from "./embeddings/embedder.js";

/**
 * Deterministic 64-dimensional bag-of-tokens embedder. Same input →
 * same vector; lexically-overlapping inputs are similar; everything
 * else is noise. Good enough for tests that need a working `embed()`
 * but don't care about retrieval quality.
 */
export class StubEmbedder implements Embedder {
  readonly name = "stub";
  readonly dimension = 64;

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => {
      const v = new Array(this.dimension).fill(0);
      for (const tok of tokenize(t)) {
        let h = 0;
        for (let i = 0; i < tok.length; i++) {
          h = (h * 31 + tok.charCodeAt(i)) >>> 0;
        }
        v[h % this.dimension] += 1;
      }
      const norm = Math.hypot(...v) || 1;
      return v.map((x) => x / norm);
    });
  }
}
