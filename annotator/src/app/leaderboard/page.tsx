"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import Link from "next/link";

type Entry = {
  rank: number;
  id: number;
  name: string;
  annotated: number;
  approved: number;
  avgTime: number;
};

function formatTime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

export default function LeaderboardPage() {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [loading, setLoading] = useState(true);

  const userId = useSyncExternalStore(
    () => () => {},
    () => localStorage.getItem("strata_user_id"),
    () => null,
  );

  async function fetchLeaderboard() {
    try {
      const res = await fetch("/api/leaderboard");
      const data = await res.json();
      setEntries(data.entries ?? []);
    } catch (err) {
      console.error("Failed to fetch leaderboard:", err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchLeaderboard();
    const interval = setInterval(fetchLeaderboard, 30000);
    return () => clearInterval(interval);
  }, []);

  const currentUserId = userId ? Number(userId) : null;

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
        <h1 className="text-sm font-semibold text-zinc-50">Leaderboard</h1>
        <Link
          href="/annotate"
          className="rounded-lg px-3 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
        >
          Back to annotating
        </Link>
      </header>

      <div className="flex-1 overflow-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <span className="text-sm text-zinc-500">Loading...</span>
          </div>
        ) : entries.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <span className="text-sm text-zinc-500">
              No annotations yet. Be the first!
            </span>
          </div>
        ) : (
          <table className="mx-auto w-full max-w-xl">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-xs text-zinc-500">
                <th className="px-3 py-2 font-medium">#</th>
                <th className="px-3 py-2 font-medium">Name</th>
                <th className="px-3 py-2 text-right font-medium">Annotated</th>
                <th className="px-3 py-2 text-right font-medium">Approved</th>
                <th className="px-3 py-2 text-right font-medium">Avg Time</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => {
                const isMe = e.id === currentUserId;
                return (
                  <tr
                    key={e.id}
                    className={`border-b border-zinc-800/50 text-sm ${
                      isMe
                        ? "bg-blue-600/10 text-zinc-100"
                        : "text-zinc-300"
                    }`}
                  >
                    <td className="px-3 py-2 font-mono text-xs text-zinc-500">
                      {e.rank}
                    </td>
                    <td className="px-3 py-2">
                      {e.name}
                      {isMe && (
                        <span className="ml-2 text-xs text-blue-400">you</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {e.annotated}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {e.approved}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-zinc-400">
                      {formatTime(e.avgTime)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
