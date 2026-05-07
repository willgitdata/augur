import { test } from "node:test";
import assert from "node:assert/strict";
import { MetadataChunker } from "./metadata-chunker.js";
import { SentenceChunker, chunkDocument } from "./chunker.js";

test("MetadataChunker: prepends default prefix to each chunk", () => {
  const wrapper = new MetadataChunker({ base: new SentenceChunker() });
  const chunks = wrapper.chunk({
    id: "doc1",
    content: "First sentence here. Second sentence here. Third sentence here.",
    metadata: { topic: "postgres", title: "Pooling" },
  });
  assert.ok(chunks.length > 0);
  for (const c of chunks) {
    assert.ok(c.content.startsWith("[doc1 | title: Pooling | topic: postgres]"));
  }
});

test("MetadataChunker: omits prefix when no metadata fields match", () => {
  const wrapper = new MetadataChunker({ base: new SentenceChunker() });
  const chunks = wrapper.chunk({
    id: "doc-x",
    content: "First. Second. Third.",
  });
  // Default formatter at minimum includes the doc id.
  for (const c of chunks) {
    assert.ok(c.content.startsWith("[doc-x]"));
  }
});

test("MetadataChunker: custom formatPrefix overrides default", () => {
  const wrapper = new MetadataChunker({
    base: new SentenceChunker(),
    formatPrefix: (doc) => `<<${doc.id.toUpperCase()}>>`,
  });
  const chunks = wrapper.chunk({
    id: "abc",
    content: "Only one sentence.",
  });
  assert.ok(chunks[0]!.content.startsWith("<<ABC>>"));
});

test("MetadataChunker: works through chunkDocument helper", async () => {
  const wrapper = new MetadataChunker({ base: new SentenceChunker() });
  const chunks = await chunkDocument(wrapper, {
    id: "doc-y",
    content: "First. Second.",
    metadata: { topic: "redis" },
  });
  assert.ok(chunks.every((c) => c.content.includes("redis")));
});
