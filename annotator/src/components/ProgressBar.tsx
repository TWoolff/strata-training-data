"use client";

type Props = {
  totalImages: number;
  pendingImages: number;
  personalCount: number;
};

export function ProgressBar({ totalImages, pendingImages, personalCount }: Props) {
  const done = totalImages - pendingImages;
  const pct = totalImages > 0 ? Math.round((done / totalImages) * 100) : 0;

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <div className="h-1.5 w-24 overflow-hidden rounded-full bg-zinc-700">
          <div
            className="h-full rounded-full bg-blue-500 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-xs text-zinc-500">
          {pendingImages} of {totalImages} need help
        </span>
      </div>
      <div className="h-3 w-px bg-zinc-700" />
      <span className="text-xs text-zinc-400">
        You: <span className="font-mono font-semibold text-zinc-200">{personalCount}</span>
      </span>
    </div>
  );
}
