import { test } from "node:test";
import assert from "node:assert/strict";
import { SemanticChunker, chunkDocument } from "./chunker.js";
import type { Embedder } from "../embeddings/embedder.js";

/**
 * SemanticChunker — boundary detection by adjacent-sentence cosine
 * similarity. The chunker is async (it embeds every sentence), so these
 * tests use `chunkAsync` directly and a tiny stub embedder rather than
 * downloading a 22MB ONNX model.
 *
 * The stub assigns a single 8-dim "topic" basis vector per topic word
 * and embeds each sentence as the sum of its topic vectors, normalized.
 * Sentences in the same topic produce nearly-identical vectors (cosine ≈
 * 1); sentences in different topics produce orthogonal vectors (cosine =
 * 0). This is enough to pin the boundary-detection logic without the
 * indeterminism of a real embedder.
 */

const TOPICS = ["alpha", "beta", "gamma", "delta"] as const;

class TopicEmbedder implements Embedder {
  readonly name = "topic-stub";
  readonly dimension = 8;
  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => {
      const v = new Array(this.dimension).fill(0);
      const lower = t.toLowerCase();
      TOPICS.forEach((topic, i) => {
        if (lower.includes(topic)) v[i] += 1;
      });
      const norm = Math.hypot(...v) || 1;
      return v.map((x) => x / norm);
    });
  }
}

const embedder = new TopicEmbedder();

test("SemanticChunker: exposes chunkAsync only — no synchronous .chunk()", () => {
  // SemanticChunker doesn't implement the synchronous Chunker interface
  // by design (it has to embed first). Pinning the absence so a
  // refactor that re-adds .chunk() — even one that throws — has to
  // consciously update this test.
  const c = new SemanticChunker({ embedder });
  assert.equal(typeof (c as { chunk?: unknown }).chunk, "undefined");
  assert.equal(typeof c.chunkAsync, "function");
});

test("SemanticChunker: cuts when adjacent sentences shift topic", async () => {
  const c = new SemanticChunker({ embedder, threshold: 0.5 });
  // Two clear topic blocks, drop in cosine at the boundary should fire a cut.
  const doc = {
    id: "d1",
    content:
      "Alpha alpha alpha. Alpha alpha alpha alpha. " +
      "Beta beta beta. Beta beta beta beta.",
  };
  const chunks = await c.chunkAsync(doc);
  assert.ok(chunks.length >= 2, `expected at least 2 chunks, got ${chunks.length}`);
  // First chunk should be alpha-only; second should contain beta.
  assert.ok(chunks[0]!.content.toLowerCase().includes("alpha"));
  assert.ok(chunks[chunks.length - 1]!.content.toLowerCase().includes("beta"));
});

test("SemanticChunker: stays on one chunk when topic is consistent", async () => {
  const c = new SemanticChunker({ embedder, threshold: 0.5 });
  const doc = {
    id: "d1",
    content: "Alpha alpha alpha. Alpha alpha. Alpha alpha alpha alpha.",
  };
  const chunks = await c.chunkAsync(doc);
  // Same topic throughout → no boundary, single chunk.
  assert.equal(chunks.length, 1);
});

test("SemanticChunker: enforces maxSize cap even on consistent topics", async () => {
  const c = new SemanticChunker({ embedder, threshold: 0.0, maxSize: 50 });
  // Same topic everywhere; threshold=0 disables similarity-cut.
  // maxSize=50 forces splits on size alone.
  const doc = {
    id: "d1",
    content:
      "Alpha alpha alpha. Alpha alpha alpha. Alpha alpha alpha. " +
      "Alpha alpha alpha. Alpha alpha alpha.",
  };
  const chunks = await c.chunkAsync(doc);
  assert.ok(chunks.length >= 2, `maxSize should force splits`);
});

test("SemanticChunker: empty document yields zero chunks", async () => {
  const c = new SemanticChunker({ embedder });
  const chunks = await c.chunkAsync({ id: "d1", content: "" });
  assert.equal(chunks.length, 0);
});

test("SemanticChunker: chunk IDs follow ${docId}:${index} convention", async () => {
  const c = new SemanticChunker({ embedder, threshold: 0.5 });
  const doc = {
    id: "doc-42",
    content: "Alpha alpha. Beta beta. Gamma gamma.",
  };
  const chunks = await c.chunkAsync(doc);
  chunks.forEach((ch, i) => {
    assert.equal(ch.id, `doc-42:${i}`);
    assert.equal(ch.documentId, "doc-42");
    assert.equal(ch.index, i);
  });
});

test("SemanticChunker: propagates document metadata to each chunk", async () => {
  const c = new SemanticChunker({ embedder, threshold: 0.5 });
  const doc = {
    id: "d1",
    content: "Alpha alpha. Beta beta.",
    metadata: { topic: "test-topic", lang: "en" },
  };
  const chunks = await c.chunkAsync(doc);
  for (const ch of chunks) {
    assert.equal(ch.metadata?.topic, "test-topic");
    assert.equal(ch.metadata?.lang, "en");
  }
});

test("chunkDocument: dispatches SemanticChunker to its async path", async () => {
  // Polymorphic helper used by Augur.index — make sure async detection
  // works for SemanticChunker (it doesn't subclass the base Chunker).
  const c = new SemanticChunker({ embedder, threshold: 0.5 });
  const doc = { id: "d1", content: "Alpha alpha. Beta beta." };
  const chunks = await chunkDocument(c, doc);
  assert.ok(chunks.length >= 1);
});
