import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Augur Dashboard",
  description: "Inspect and debug retrieval traces",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-mono text-sm antialiased">
        <header className="border-b border-ink-700 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-2 w-2 rounded-full bg-accent-500" />
            <span className="font-semibold tracking-tight">Augur</span>
            <span className="text-ink-500 text-xs">/ dashboard</span>
          </div>
          <nav className="flex gap-6 text-ink-300 text-xs">
            <a href="/" className="hover:text-white">Playground</a>
            <a href="/traces" className="hover:text-white">Traces</a>
            <a href={process.env.QUERYBRAIN_URL + "/docs"} target="_blank" className="hover:text-white">API Docs ↗</a>
          </nav>
        </header>
        <main className="px-6 py-6 max-w-6xl mx-auto">{children}</main>
      </body>
    </html>
  );
}
