import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ContextualChunker,
  MemoryContextCache,
  type ContextProvider,
} from "./contextual-chunker.js";
import { SentenceChunker, chunkDocument } from "./chunker.js";

class CountingProvider implements ContextProvider {
  readonly name = "test";
  calls = 0;
  async contextualize({ chunk }: { chunk: string; document: string }): Promise<string> {
    this.calls++;
    // Echo a stable identifier per chunk so tests can assert order.
    return `ctx:${chunk.slice(0, 12)}`;
  }
}

test("ContextualChunker: prepends LLM context to each base chunk", async () => {
  const provider = new CountingProvider();
  const chunker = new ContextualChunker({
    base: new SentenceChunker({ targetSize: 30, maxSize: 60 }),
    provider,
  });

  const doc = {
    id: "doc-1",
    content:
      "Augur is a retrieval orchestration layer. It picks the strategy per query. " +
      "The trace explains every decision.",
  };
  const chunks = await chunkDocument(chunker, doc);

  assert.ok(chunks.length >= 2, "expected multiple chunks");
  for (const c of chunks) {
    assert.match(c.content, /^ctx:.+\n\n/, `chunk should start with "ctx:..." prefix: ${c.content.slice(0, 40)}`);
  }
  assert.equal(provider.calls, chunks.length);
});

test("ContextualChunker: cache hit avoids re-calling the LLM on re-index", async () => {
  const provider = new CountingProvider();
  const cache = new MemoryContextCache();
  const chunker = new ContextualChunker({
    base: new SentenceChunker({ targetSize: 30, maxSize: 60 }),
    provider,
    cache,
  });

  const doc = {
    id: "doc-1",
    content: "Sentence one is short. Sentence two follows. Sentence three lands.",
  };

  const first = await chunkDocument(chunker, doc);
  const callsAfterFirst = provider.calls;
  assert.ok(callsAfterFirst >= 1);

  // Same doc again — every chunk should hit the cache.
  const second = await chunkDocument(chunker, doc);
  assert.equal(provider.calls, callsAfterFirst, "no new LLM calls on cache hit");
  assert.equal(first.length, second.length);
  for (let i = 0; i < first.length; i++) {
    assert.equal(first[i]!.content, second[i]!.content);
  }
});

test("ContextualChunker: empty context falls through unchanged", async () => {
  const provider: ContextProvider = {
    name: "empty",
    async contextualize() {
      return "   "; // whitespace-only — should be treated as no context
    },
  };
  const chunker = new ContextualChunker({
    base: new SentenceChunker({ targetSize: 30, maxSize: 60 }),
    provider,
  });
  const doc = { id: "d", content: "First sentence. Second one here. Third." };
  const chunks = await chunkDocument(chunker, doc);
  for (const c of chunks) {
    assert.doesNotMatch(c.content, /^\s*\n\n/);
  }
});

test("ContextualChunker: synchronous chunk() throws helpful error", () => {
  const chunker = new ContextualChunker({
    base: new SentenceChunker(),
    provider: new CountingProvider(),
  });
  assert.throws(
    () => chunker.chunk({ id: "x", content: "test" }),
    /async/
  );
});
