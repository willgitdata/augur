# Migrating

Augur is pre-1.0. Breaking changes can land on `0.x` minors and we call them
out here so upgrades aren't archeology against `git log`. Each section is
self-contained: the old shape, the new shape, the smallest diff that gets
you across.

For the full list of additions and non-breaking changes per release, see
[CHANGELOG.md](CHANGELOG.md).

---

## Upgrading to `0.2.x` (from `0.1.x`)

### `Augur` requires an explicit `embedder`

The placeholder `HashEmbedder` / `TfIdfEmbedder` are gone — they produced
near-random vectors that surfaced as product bugs in user pipelines.

```diff
- const augr = new Augur();
+ import { LocalEmbedder } from "@augur/core";
+ const augr = new Augur({ embedder: new LocalEmbedder() });
```

For hosted providers, implement the three-method `Embedder` interface;
see [EXAMPLES.md §5](EXAMPLES.md) for OpenAI / Cohere / Voyage / Gemini /
Jina snippets.

### `Augur` no longer defaults to `HeuristicReranker`

The previous default reranker was a token-overlap + proximity heuristic.
It gave the trace a "yes I rerank" line while doing close to nothing —
users on the auto path got bare retrieval-shaped output that pretended
it was reranked. The default is now `null`: no reranker fires unless you
ask for one.

To keep the headline accuracy mode (the cross-encoder voting on every
query that produces NDCG@10 = 0.920 on the bundled eval), pass
`LocalReranker` explicitly:

```diff
- const augr = new Augur({ embedder: new LocalEmbedder() });
+ import { LocalEmbedder, LocalReranker } from "@augur/core";
+ const augr = new Augur({
+   embedder: new LocalEmbedder(),
+   reranker: new LocalReranker(),
+ });
```

If you want the old behavior back (you almost certainly do not — it was
silently broken), pass `new HeuristicReranker()` explicitly. Hosted
rerankers (Cohere, Voyage, Jina) plug in via the one-method `Reranker`
interface — see [EXAMPLES.md §5](EXAMPLES.md).

The `/health` endpoint's `reranker` field is now nullable — `null` when
no reranker is configured.

### `signals.tokens` → `signals.wordCount`

The previous names were misleading: these are whitespace-split *words*,
not embedding subword tokens. The router uses them for rule decisions
(`≥6 words AND isQuestion → vector`); the embedder has its own
WordPiece/BPE tokenization that's independent.

```diff
- if (signals.tokens >= 6 && signals.isQuestion) ...
+ if (signals.wordCount >= 6 && signals.isQuestion) ...

- console.log(signals.avgTokenLen);
+ console.log(signals.avgWordLen);
```

### `signals.language` widened from `"en" | "non-en"` → BCP-47 code

The router now distinguishes `ja`, `zh`, `ko`, `ru`, `ar`, `hi`, `th`,
`he`, `el` from each other (and from `en`) via Unicode-script analysis,
which feeds the new `autoLanguageFilter` feature. If you were
pattern-matching on the binary value:

```diff
- if (signals.language === "non-en") ...
+ if (signals.language !== "en") ...
```

### `HeuristicRouter` defaults `alwaysRerank: true`

The cross-encoder votes on every query out of the box. This is what the
README's headline numbers measure. Latency-sensitive deployments that
were relying on the conditional-rerank fast path need to opt back in:

```diff
- const router = new HeuristicRouter();
+ const router = new HeuristicRouter({ alwaysRerank: false });
```

`alwaysRerank: false` is the right choice when:
- p95 < 25 ms is a hard SLO (your retrieval is BM25-only and a
  cross-encoder forward pass is your dominant cost), or
- you've measured that your specific corpus + queries don't benefit
  from rerank (rare — the gain is real on most workloads).

### Hosted-provider embedder/reranker classes removed from core

`GeminiEmbedder`, `OpenAIEmbedder`, `CohereEmbedder`, `VoyageEmbedder`,
`JinaEmbedder`, `JinaReranker`, `CohereReranker`, `VoyageReranker` are
no longer exported from `@augur/core`. Implement the `Embedder` /
`Reranker` interfaces against your provider's SDK in your own codebase
— see [EXAMPLES.md §5](EXAMPLES.md) for ten-line snippets per provider.

This keeps `@augur/core` dependency-free and forces secrets management
into the user's own code, where it belongs.

### `apps/dashboard` and `evaluations/` removed from the repo

These were development tools, not runtime dependencies. The repo now
ships only the SDK (`@augur/core`) and the optional Fastify wrapper
(`@augur/server`). Both removed trees are preserved in git history and
may resurface as standalone sister repos.

If you were running the bundled BEIR / 504-query eval harness, you can
restore it from commit `feffc73^` or earlier. The published packages
were not affected — performance numbers in the README and CHANGELOG
were measured against `@augur/core` exactly as it ships today.

The `AUGUR_SEED_DEMO` env var is gone with the bundled corpus.

---

## Adopting new features (non-breaking)

These don't require code changes — they're additive — but they're worth
turning on:

### `autoLanguageFilter`

Off by default. Turn it on if your corpus has language-localized
canonical answers (one Japanese doc and one English doc per topic);
leave it off for primarily-English knowledge bases queried in many
languages (see the JSDoc on `AugurOptions.autoLanguageFilter` for
the full reasoning).

```ts
const augr = new Augur({
  embedder: new LocalEmbedder(),
  autoLanguageFilter: true,
});
```

### `SearchRequest.minScore`

Drops results below a confidence floor — useful when "no answer" is a
better signal to your downstream LLM than a noisy low-relevance answer.
Cross-encoder scores are typically calibrated `[0, 1]`; `0.4` is a
reasonable starting point.

```ts
await augr.search({ query, minScore: 0.4 });
```

### `LocalEmbedder` quantization (`dtype`, `device`)

```ts
new LocalEmbedder({ dtype: "q8" });   // ~4× smaller, marginal quality loss
new LocalEmbedder({ device: "wasm" }); // explicit ONNX runtime
```

### `ContextualChunker`

Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)
pattern (~67% reduction in chunk-failure rate per their published
numbers). Wraps any base chunker; uses your LLM provider via a
one-method `ContextProvider` interface. See [EXAMPLES.md](EXAMPLES.md)
for the wiring.
