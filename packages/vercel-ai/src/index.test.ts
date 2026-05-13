import { test } from "node:test";
import assert from "node:assert/strict";
import type { Augur, SearchRequest, SearchResponse } from "@augur-rag/core";
import { augurToolDescriptor } from "./index.js";

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

test("augurToolDescriptor: shape matches Vercel AI SDK's tool descriptor expectations", () => {
  const { augur } = stubAugur({ results: [], trace: traceStub });
  const desc = augurToolDescriptor(augur);
  assert.equal(typeof desc.description, "string");
  assert.equal(desc.parameters.type, "object");
  assert.deepEqual(desc.parameters.required, ["query"]);
  assert.equal(desc.parameters.properties.query.type, "string");
  assert.equal(typeof desc.execute, "function");
});

test("augurToolDescriptor: execute() runs search and returns trimmed result shape", async () => {
  const { augur, captured } = stubAugur({
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
  const desc = augurToolDescriptor(augur);
  const out = await desc.execute({ query: "hi" });
  assert.equal(captured[0]!.query, "hi");
  assert.equal(out.results.length, 1);
  // Trimmed result shape — only content / documentId / score for the model.
  assert.deepEqual(Object.keys(out.results[0]!).sort(), ["content", "documentId", "score"]);
  assert.equal(out.results[0]!.content, "hello world");
  assert.equal(out.results[0]!.documentId, "doc1");
  assert.equal(out.results[0]!.score, 0.9);
});

test("augurToolDescriptor: default topK is 5", async () => {
  const { augur, captured } = stubAugur({ results: [], trace: traceStub });
  const desc = augurToolDescriptor(augur);
  await desc.execute({ query: "q" });
  assert.equal(captured[0]!.topK, 5);
});

test("augurToolDescriptor: honors topK + latencyBudgetMs overrides", async () => {
  const { augur, captured } = stubAugur({ results: [], trace: traceStub });
  const desc = augurToolDescriptor(augur, { topK: 12, latencyBudgetMs: 250 });
  await desc.execute({ query: "q" });
  assert.equal(captured[0]!.topK, 12);
  assert.equal(captured[0]!.latencyBudgetMs, 250);
});

test("augurToolDescriptor: custom description overrides the default", () => {
  const { augur } = stubAugur({ results: [], trace: traceStub });
  const desc = augurToolDescriptor(augur, {
    description: "Search the company handbook",
  });
  assert.equal(desc.description, "Search the company handbook");
});
