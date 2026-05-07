/**
 * Compare chunking strategies on the same document.
 *
 * Run:  pnpm --filter example-chunking start
 *
 * Use this as a starting point for tuning your own chunker — drop a real
 * document in here and see how each strategy slices it.
 */
import {
  FixedSizeChunker,
  SentenceChunker,
  SemanticChunker,
  HashEmbedder,
  chunkDocument,
} from "@augur/core";

const doc = {
  id: "guide",
  content: `
Augur is an adaptive retrieval orchestration layer for AI applications.
It sits on top of existing vector databases and decides, at query time, which
retrieval strategy to use: vector, keyword, hybrid, or vector-plus-rerank.

The product philosophy is "augment, don't replace". You bring your own vector
store, your own embedder, your own data pipeline. Augur decides how to
use them.

Why does adaptive routing matter? Because no single retrieval strategy works
best for every query. Short queries with specific identifiers favor keyword
search. Long natural-language questions favor semantic search. Many real
queries sit in between, and hybrid retrieval reliably outperforms either
alone — but only when paired with sensible rank fusion.

For long-form documents, semantic chunking often beats fixed-size chunking
because it respects topic boundaries. The tradeoff is latency: semantic
chunking requires embedding every sentence at chunk time.
  `.trim(),
};

async function main() {
  const fixed = new FixedSizeChunker({ size: 200, overlap: 30 });
  const sentence = new SentenceChunker({ targetSize: 200, maxSize: 400 });
  const semantic = new SemanticChunker({ embedder: new HashEmbedder(), threshold: 0.5 });

  for (const chunker of [fixed, sentence, semantic] as const) {
    const chunks = await chunkDocument(chunker, doc);
    console.log(`\n--- ${chunker.name} (${chunks.length} chunks) ---`);
    chunks.forEach((c, i) => {
      const preview = c.content.length > 100 ? c.content.slice(0, 100) + "…" : c.content;
      console.log(`  ${i}. (${c.content.length} chars) ${preview}`);
    });
  }
}
main();
