import { test } from "node:test";
import assert from "node:assert/strict";
import type { Augur, SearchRequest, SearchResponse } from "../index.js";
import { searchAsLangchainDocs } from "./langchain.js";

/**
 * Stub Augur — captures the SearchRequest and returns a canned
 * SearchResponse so we can exercise the shape conversion without
 * loading the full Augur stack.
 */
function stubAugur(response: SearchResponse): {
  augur: Augur;
  captured: SearchRequest[];
} {
  const captured: SearchRequest[] = [];
  const augur = {
    async search(req: SearchRequest) {
      captured.push(req);
      return response;
    },
  } as unknown as Augur;
  return { augur, captured };
}

const traceStub = {
  id: "t1",
  query: "q",
  startedAt: new Date().toISOString(),
  totalMs: 1,
  decision: {
    strategy: "vector" as const,
    reasons: [],
    reranked: false,
    signals: {
      wordCount: 1,
      avgWordLen: 1,
      hasQuotedPhrase: false,
      hasSpecificTokens: false,
      isQuestion: false,
      ambiguity: 0,
      hasNamedEntity: false,
      hasCodeLike: false,
      hasDateOrVersion: false,
      questionType: null,
      hasNegation: false,
      language: "en",
    },
  },
  spans: [],
  candidates: 0,
  adapter: "stub",
};

test("searchAsLangchainDocs: maps SearchResult → { pageContent, metadata }", async () => {
  const { augur } = stubAugur({
    results: [
      {
        chunk: {
          id: "doc1:0",
          documentId: "doc1",
          content: "hello world",
          index: 0,
          metadata: { topic: "k8s" },
        },
        score: 0.9,
      },
    ],
    trace: traceStub,
  });
  const docs = await searchAsLangchainDocs(augur, "hi");
  assert.equal(docs.length, 1);
  assert.equal(docs[0]!.pageContent, "hello world");
  assert.equal(docs[0]!.metadata.chunkId, "doc1:0");
  assert.equal(docs[0]!.metadata.documentId, "doc1");
  assert.equal(docs[0]!.metadata.score, 0.9);
  assert.equal(docs[0]!.metadata.topic, "k8s", "original chunk metadata is preserved");
});

test("searchAsLangchainDocs: binding-added fields override user metadata if it collides", async () => {
  const { augur } = stubAugur({
    results: [
      {
        chunk: {
          id: "doc1:0",
          documentId: "doc1",
          content: "x",
          index: 0,
          // The chunk metadata sneakily contains a `score` field — the
          // binding's `score` (= retrieval score) must win for
          // downstream consumers to trust it.
          metadata: { score: "stale", chunkId: "spoofed" },
        },
        score: 0.42,
      },
    ],
    trace: traceStub,
  });
  const [doc] = await searchAsLangchainDocs(augur, "q");
  assert.equal(doc!.metadata.score, 0.42);
  assert.equal(doc!.metadata.chunkId, "doc1:0");
});

test("searchAsLangchainDocs: forwards topK / filter / latencyBudgetMs / minScore", async () => {
  const { augur, captured } = stubAugur({ results: [], trace: traceStub });
  await searchAsLangchainDocs(augur, "q", {
    topK: 3,
    filter: { topic: "k8s" },
    latencyBudgetMs: 500,
    minScore: 0.3,
  });
  assert.equal(captured.length, 1);
  assert.equal(captured[0]!.query, "q");
  assert.equal(captured[0]!.topK, 3);
  assert.deepEqual(captured[0]!.filter, { topic: "k8s" });
  assert.equal(captured[0]!.latencyBudgetMs, 500);
  assert.equal(captured[0]!.minScore, 0.3);
});

test("searchAsLangchainDocs: omits optional fields from the SearchRequest when unset", async () => {
  const { augur, captured } = stubAugur({ results: [], trace: traceStub });
  await searchAsLangchainDocs(augur, "q");
  assert.equal(captured[0]!.filter, undefined);
  assert.equal(captured[0]!.latencyBudgetMs, undefined);
  assert.equal(captured[0]!.minScore, undefined);
});

test("searchAsLangchainDocs: empty results → empty docs", async () => {
  const { augur } = stubAugur({ results: [], trace: traceStub });
  const docs = await searchAsLangchainDocs(augur, "no-hits");
  assert.deepEqual(docs, []);
});
