"use client";

import { useEffect, useState } from "react";
import { TraceView } from "@/components/TraceView";

const QB = process.env.QUERYBRAIN_URL ?? "http://localhost:3001";

export default function TracesPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [selected, setSelected] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    try {
      const r = await fetch(`${QB}/traces?limit=200`);
      const json = await r.json();
      setTraces(json.traces);
      if (!selected && json.traces[0]) setSelected(json.traces[0]);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    load();
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
      <aside>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold">Recent traces</h2>
          <span className="text-xs text-ink-500">{traces.length}</span>
        </div>
        {loading ? (
          <div className="text-xs text-ink-500">Loading…</div>
        ) : traces.length === 0 ? (
          <div className="text-xs text-ink-500 border border-dashed border-ink-700 rounded p-4">
            No traces yet. Try the playground.
          </div>
        ) : (
          <ul className="space-y-1.5">
            {traces.map((t) => (
              <li key={t.id}>
                <button
                  onClick={() => setSelected(t)}
                  className={`w-full text-left text-xs px-3 py-2 rounded border transition ${
                    selected?.id === t.id
                      ? "bg-ink-800 border-accent-500"
                      : "border-ink-700 hover:border-ink-500"
                  }`}
                >
                  <div className="flex justify-between">
                    <span className="truncate max-w-[180px]">{t.query}</span>
                    <span className="text-ink-500">{t.totalMs.toFixed(0)}ms</span>
                  </div>
                  <div className="flex gap-2 mt-1 text-[10px] text-ink-500">
                    <span>{t.decision.strategy}</span>
                    {t.decision.reranked && <span>+rerank</span>}
                    <span className="ml-auto">{t.candidates} cand.</span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>
      <section>
        {selected ? (
          <>
            <div className="mb-3">
              <div className="text-xs text-ink-500">query</div>
              <div className="text-sm">{selected.query}</div>
            </div>
            <TraceView trace={selected} />
          </>
        ) : (
          <div className="text-ink-500 text-xs border border-dashed border-ink-700 rounded p-6 text-center">
            Select a trace.
          </div>
        )}
      </section>
    </div>
  );
}
