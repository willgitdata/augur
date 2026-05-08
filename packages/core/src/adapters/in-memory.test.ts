import { test } from "node:test";
import assert from "node:assert/strict";
import { InMemoryAdapter } from "./in-memory.js";
import type { Chunk } from "../types.js";

function chunk(id: string, content: string, meta?: Record<string, unknown>): Chunk {
  return { id, documentId: id.split(":")[0]!, content, index: 0, ...(meta ? { metadata: meta } : {}) };
}

test("InMemoryAdapter: keyword search without stemming misses morphological variants", async () => {
  const a = new InMemoryAdapter();
  await a.upsert([
    chunk("1:0", "VACUUM in PostgreSQL reclaims dead tuples and prevents bloat."),
    chunk("2:0", "Redis Cluster shards keys across hash slots."),
  ]);
  // "vacuums" (plural) doesn't match "vacuum" (singular) without stemming.
  const out = await a.searchKeyword({ query: "vacuums", topK: 5 });
  assert.equal(out.length, 0);
});

test("InMemoryAdapter: useStemming=true matches morphological variants", async () => {
  const a = new InMemoryAdapter({ useStemming: true });
  await a.upsert([
    chunk("1:0", "VACUUM in PostgreSQL reclaims dead tuples and prevents bloat."),
    chunk("2:0", "Redis Cluster shards keys across hash slots."),
  ]);
  // Both "vacuums" and "vacuumed" stem to "vacuum" (or close) → hit doc 1.
  const out = await a.searchKeyword({ query: "vacuums", topK: 5 });
  assert.equal(out.length, 1);
  assert.equal(out[0]!.chunk.documentId, "1");
});

test("InMemoryAdapter: useStemming drops English stopwords", async () => {
  const a = new InMemoryAdapter({ useStemming: true });
  await a.upsert([chunk("1:0", "the quick brown fox jumps over the lazy dog")]);
  // "the" is stopworded and ignored entirely; only content tokens drive scoring.
  const out = await a.searchKeyword({ query: "the the the the", topK: 5 });
  assert.equal(out.length, 0);
});

test("InMemoryAdapter: configurable BM25 k1 and b", async () => {
  const a = new InMemoryAdapter({ k1: 2.0, b: 0.5 });
  await a.upsert([chunk("1:0", "redis redis redis cache")]);
  const out = await a.searchKeyword({ query: "redis", topK: 1 });
  assert.equal(out.length, 1);
  assert.ok(out[0]!.score > 0);
});
