"use client";

import { useState } from "react";
import { TraceView } from "@/components/TraceView";
import { ResultList } from "@/components/ResultList";

const AUGR = process.env.AUGUR_URL ?? "http://localhost:3001";

interface SearchResp {
  results: Array<{
    chunk: { id: string; documentId: string; content: string; index: number };
    score: number;
    rawScores?: Record<string, number>;
  }>;
  trace: any;
}

export default function PlaygroundPage() {
  const [query, setQuery] = useState("How do I configure connection pooling in Postgres?");
  const [topK, setTopK] = useState(5);
  const [forced, setForced] = useState<string>("");
  const [budget, setBudget] = useState<number | "">("");
  const [resp, setResp] = useState<SearchResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch() {
    setLoading(true);
    setError(null);
    try {
      const body: any = { query, topK };
      if (forced) body.forceStrategy = forced;
      if (budget !== "") body.latencyBudgetMs = Number(budget);
      const r = await fetch(`${AUGR}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(await r.text());
      setResp(await r.json());
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <section>
        <h1 className="text-lg font-semibold mb-4">Query Playground</h1>
        <div className="space-y-3">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            className="w-full bg-ink-800 border border-ink-700 rounded px-3 py-2 outline-none focus:border-accent-500"
            placeholder="Enter a query..."
          />
          <div className="grid grid-cols-3 gap-3">
            <label className="text-xs text-ink-500">
              Top-K
              <input
                type="number"
                value={topK}
                min={1}
                max={50}
                onChange={(e) => setTopK(parseInt(e.target.value || "5", 10))}
                className="mt-1 w-full bg-ink-800 border border-ink-700 rounded px-2 py-1.5"
              />
            </label>
            <label className="text-xs text-ink-500">
              Force strategy
              <select
                value={forced}
                onChange={(e) => setForced(e.target.value)}
                className="mt-1 w-full bg-ink-800 border border-ink-700 rounded px-2 py-1.5"
              >
                <option value="">auto</option>
                <option value="vector">vector</option>
                <option value="keyword">keyword</option>
                <option value="hybrid">hybrid</option>
                <option value="rerank">rerank</option>
              </select>
            </label>
            <label className="text-xs text-ink-500">
              Latency budget (ms)
              <input
                type="number"
                min={0}
                value={budget}
                onChange={(e) =>
                  setBudget(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))
                }
                className="mt-1 w-full bg-ink-800 border border-ink-700 rounded px-2 py-1.5"
              />
            </label>
          </div>
          <button
            onClick={runSearch}
            disabled={loading || query.trim().length === 0}
            className="bg-accent-600 hover:bg-accent-500 text-white text-sm font-medium px-4 py-2 rounded disabled:opacity-50"
          >
            {loading ? "Searching…" : "Search"}
          </button>
          {error && (
            <div className="text-red-400 text-xs border border-red-900 bg-red-950 rounded p-3 mt-2">
              {error}
            </div>
          )}
        </div>

        {resp && (
          <div className="mt-6">
            <h2 className="text-sm font-semibold text-ink-300 mb-2">
              Results ({resp.results.length})
            </h2>
            <ResultList results={resp.results} />
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-4">Trace</h2>
        {resp ? (
          <TraceView trace={resp.trace} />
        ) : (
          <div className="text-ink-500 text-xs border border-dashed border-ink-700 rounded p-6 text-center">
            Run a query to see the routing decision and execution timeline.
          </div>
        )}
      </section>
    </div>
  );
}
