import { test } from "node:test";
import assert from "node:assert/strict";
import type { Augur, SearchRequest, SearchResponse } from "@augur-rag/core";
import { searchAsLlamaIndexNodes } from "./index.js";

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

test("searchAsLlamaIndexNodes: maps SearchResult → NodeWithScore", async () => {
  const { augur } = stubAugur({
    results: [
      {
        chunk: {
          id: "doc1:0",
          documentId: "doc1",
          content: "hello",
          index: 0,
          metadata: { topic: "k8s" },
        },
        score: 0.85,
      },
    ],
    trace: traceStub,
  });
  const nodes = await searchAsLlamaIndexNodes(augur, "hi");
  assert.equal(nodes.length, 1);
  assert.equal(nodes[0]!.node.id_, "doc1:0");
  assert.equal(nodes[0]!.node.text, "hello");
  assert.equal(nodes[0]!.node.metadata.documentId, "doc1");
  assert.equal(nodes[0]!.node.metadata.topic, "k8s");
  assert.equal(nodes[0]!.score, 0.85);
});

test("searchAsLlamaIndexNodes: forwards options", async () => {
  const { augur, captured } = stubAugur({ results: [], trace: traceStub });
  await searchAsLlamaIndexNodes(augur, "q", {
    topK: 7,
    filter: { topic: "k8s" },
    latencyBudgetMs: 300,
    minScore: 0.2,
  });
  assert.equal(captured[0]!.topK, 7);
  assert.deepEqual(captured[0]!.filter, { topic: "k8s" });
  assert.equal(captured[0]!.latencyBudgetMs, 300);
  assert.equal(captured[0]!.minScore, 0.2);
});

test("searchAsLlamaIndexNodes: empty results → empty nodes", async () => {
  const { augur } = stubAugur({ results: [], trace: traceStub });
  const nodes = await searchAsLlamaIndexNodes(augur, "no-hits");
  assert.deepEqual(nodes, []);
});
