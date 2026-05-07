"use client";

interface Result {
  chunk: { id: string; documentId: string; content: string; index: number };
  score: number;
  rawScores?: Record<string, number>;
}

export function ResultList({ results }: { results: Result[] }) {
  if (results.length === 0) {
    return (
      <div className="text-ink-500 text-xs border border-dashed border-ink-700 rounded p-6 text-center">
        No results.
      </div>
    );
  }
  return (
    <ol className="space-y-3">
      {results.map((r, i) => (
        <li key={r.chunk.id} className="border border-ink-700 rounded p-3 bg-ink-800/40">
          <div className="flex justify-between items-start mb-2">
            <div className="text-xs text-ink-500">
              #{i + 1} · doc <span className="text-ink-300">{r.chunk.documentId}</span> · chunk{" "}
              {r.chunk.index}
            </div>
            <div className="text-xs">
              <span className="text-ink-500 mr-1">score</span>
              <span className="text-accent-500">{r.score.toFixed(4)}</span>
            </div>
          </div>
          <div className="text-xs leading-relaxed whitespace-pre-wrap">
            {r.chunk.content.length > 400
              ? r.chunk.content.slice(0, 400) + "…"
              : r.chunk.content}
          </div>
          {r.rawScores && Object.keys(r.rawScores).length > 0 && (
            <div className="mt-2 pt-2 border-t border-ink-700 flex gap-4 text-[10px] text-ink-500">
              {Object.entries(r.rawScores).map(([k, v]) => (
                <span key={k}>
                  {k}: <span className="text-ink-300">{v.toFixed(3)}</span>
                </span>
              ))}
            </div>
          )}
        </li>
      ))}
    </ol>
  );
}
