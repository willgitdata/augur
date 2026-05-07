"use client";

interface Span {
  name: string;
  startMs: number;
  endMs: number;
  durationMs: number;
  attributes?: Record<string, unknown>;
}

interface Trace {
  id: string;
  query: string;
  totalMs: number;
  adapter: string;
  embeddingModel?: string;
  candidates: number;
  decision: {
    strategy: string;
    reasons: string[];
    reranked: boolean;
    signals: Record<string, unknown>;
  };
  spans: Span[];
}

const strategyColor: Record<string, string> = {
  vector: "bg-blue-900 text-blue-300 border-blue-800",
  keyword: "bg-amber-900 text-amber-300 border-amber-800",
  hybrid: "bg-purple-900 text-purple-300 border-purple-800",
  rerank: "bg-emerald-900 text-emerald-300 border-emerald-800",
};

export function TraceView({ trace }: { trace: Trace }) {
  const total = Math.max(trace.totalMs, 0.01);

  return (
    <div className="space-y-4">
      {/* Decision panel */}
      <div className="border border-ink-700 rounded p-4 bg-ink-800/50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span
              className={`text-xs px-2 py-0.5 rounded border ${
                strategyColor[trace.decision.strategy] ?? "bg-ink-700 text-ink-300 border-ink-700"
              }`}
            >
              {trace.decision.strategy}
            </span>
            {trace.decision.reranked && (
              <span className="text-xs px-2 py-0.5 rounded border bg-ink-700 text-ink-300 border-ink-700">
                + rerank
              </span>
            )}
          </div>
          <div className="text-xs text-ink-500">{trace.totalMs.toFixed(1)} ms</div>
        </div>
        <ul className="space-y-1 text-xs text-ink-300">
          {trace.decision.reasons.map((r, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-accent-500">→</span>
              <span>{r}</span>
            </li>
          ))}
        </ul>
        <div className="mt-3 pt-3 border-t border-ink-700 grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-ink-500">adapter</span>
            <div>{trace.adapter}</div>
          </div>
          <div>
            <span className="text-ink-500">embedder</span>
            <div>{trace.embeddingModel ?? "n/a"}</div>
          </div>
          <div>
            <span className="text-ink-500">candidates</span>
            <div>{trace.candidates}</div>
          </div>
          <div>
            <span className="text-ink-500">trace id</span>
            <div className="truncate">{trace.id}</div>
          </div>
        </div>
      </div>

      {/* Signals */}
      <div className="border border-ink-700 rounded p-4">
        <div className="text-xs font-semibold text-ink-300 mb-2">Query signals</div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {Object.entries(trace.decision.signals).map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-ink-700 py-1">
              <span className="text-ink-500">{k}</span>
              <span>{typeof v === "number" ? v.toFixed(2) : String(v)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Span timeline */}
      <div className="border border-ink-700 rounded p-4">
        <div className="text-xs font-semibold text-ink-300 mb-3">Execution timeline</div>
        <div className="space-y-1">
          {trace.spans.map((s, i) => (
            <div key={i} className="text-xs">
              <div className="flex justify-between mb-0.5">
                <span>{s.name}</span>
                <span className="text-ink-500">{s.durationMs.toFixed(2)} ms</span>
              </div>
              <div className="h-1.5 bg-ink-800 rounded relative">
                <div
                  className="absolute h-full bg-accent-500 rounded"
                  style={{
                    left: `${(s.startMs / total) * 100}%`,
                    width: `${Math.max((s.durationMs / total) * 100, 0.5)}%`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
