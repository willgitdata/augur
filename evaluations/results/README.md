# Cached eval results

Canonical metric snapshots regenerated from the bundled 504-query eval.
Use these as the diff target when measuring whether a change improves
or regresses retrieval quality:

```bash
# Diff your local run against the default-config baseline
pnpm eval -- --compare evaluations/results/baseline-default.json

# Diff against the best-known config (LocalEmbedder + LocalReranker + MetadataChunker + BM25-stem)
pnpm eval -- --reranker local --metadata-chunker --bm25-stem \
  --compare evaluations/results/baseline-best.json
```

## Files

| File                      | Config                                                                | NDCG@10 |
| ------------------------- | --------------------------------------------------------------------- | ------: |
| `baseline-default.json`   | `LocalEmbedder` + `SentenceChunker` + `HeuristicReranker`             |   0.845 |
| `baseline-best.json`      | `LocalEmbedder` + `LocalReranker` + `MetadataChunker` + stemmed BM25  |   0.910 |

## Refreshing

The results drift only when you change either:
- The corpus (`evaluations/corpus.json`)
- The queries (`evaluations/queries.json`)
- The retrieval pipeline (router signals, scoring functions, embedder defaults)

When that happens, regenerate:

```bash
pnpm eval -- --save evaluations/results/baseline-default.json
pnpm eval -- --reranker local --metadata-chunker --bm25-stem \
  --save evaluations/results/baseline-best.json
```

Commit the updated JSON in the same PR that changed the underlying
behavior. Reviewers can then see the full delta from one CI artifact.
