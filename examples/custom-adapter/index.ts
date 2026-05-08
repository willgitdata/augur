/**
 * Custom adapter — shows how to plug in your own backend.
 *
 * Anything implementing `VectorAdapter` works. Here we wrap a JSON file —
 * not a real production backend, but it demonstrates that the surface area
 * is small enough to write in an afternoon.
 *
 * Run:  pnpm --filter example-custom-adapter start
 */
import { Augur, BaseAdapter, LocalEmbedder } from "@augur/core";
import type {
  AdapterCapabilities,
  Chunk,
  KeywordSearchOpts,
  SearchResult,
  VectorSearchOpts,
} from "@augur/core";
import { readFile, writeFile, access } from "node:fs/promises";

class JsonFileAdapter extends BaseAdapter {
  readonly name = "json-file";
  readonly capabilities: AdapterCapabilities = {
    vector: true,
    keyword: true,
    hybrid: true,
    computesEmbeddings: false,
    filtering: false,
  };

  private chunks = new Map<string, Chunk>();
  private path: string;

  constructor(path: string) {
    super();
    this.path = path;
  }

  async load() {
    try {
      await access(this.path);
      const data = JSON.parse(await readFile(this.path, "utf-8"));
      for (const c of data.chunks) this.chunks.set(c.id, c);
    } catch {
      // file doesn't exist yet — that's fine
    }
  }
  private async persist() {
    await writeFile(this.path, JSON.stringify({ chunks: [...this.chunks.values()] }, null, 2));
  }

  async upsert(chunks: Chunk[]) {
    for (const c of chunks) this.chunks.set(c.id, c);
    await this.persist();
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    const out: SearchResult[] = [];
    for (const c of this.chunks.values()) {
      if (!c.embedding) continue;
      out.push({ chunk: c, score: cosine(opts.embedding, c.embedding) });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, opts.topK);
  }

  async searchKeyword(opts: KeywordSearchOpts): Promise<SearchResult[]> {
    const tokens = opts.query.toLowerCase().split(/\s+/).filter(Boolean);
    const scored: SearchResult[] = [];
    for (const c of this.chunks.values()) {
      const lc = c.content.toLowerCase();
      let s = 0;
      for (const t of tokens) if (lc.includes(t)) s += 1;
      if (s > 0) scored.push({ chunk: c, score: s / tokens.length });
    }
    return scored.sort((a, b) => b.score - a.score).slice(0, opts.topK);
  }

  async delete(ids: string[]) {
    for (const id of ids) this.chunks.delete(id);
    await this.persist();
  }
  async count() { return this.chunks.size; }
  async clear() { this.chunks.clear(); await this.persist(); }
}

function cosine(a: number[], b: number[]) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]! * b[i]!; na += a[i]! ** 2; nb += b[i]! ** 2; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function main() {
  const adapter = new JsonFileAdapter("./store.json");
  await adapter.load();
  const augr = new Augur({ adapter, embedder: new LocalEmbedder() });

  await augr.index([
    { id: "1", content: "Augur is an adaptive retrieval orchestration layer." },
    { id: "2", content: "Custom adapters are easy: implement the VectorAdapter interface." },
  ]);

  const { results, trace } = await augr.search({ query: "how to write an adapter" });
  console.log("strategy:", trace.decision.strategy);
  for (const r of results) console.log(r.chunk.id, r.score.toFixed(3), r.chunk.content);
}
main();
