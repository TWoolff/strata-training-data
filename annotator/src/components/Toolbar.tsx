"use client";

type Props = {
  brushSize: number;
  onBrushSizeChange: (size: number) => void;
  overlayOpacity: number;
  onOverlayOpacityChange: (opacity: number) => void;
  erasing: boolean;
  onToggleEraser: () => void;
  zoom: number;
  onResetZoom: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onSkip: () => void;
  onSubmit: () => void;
  annotationCount: number;
};

export function Toolbar({
  brushSize,
  onBrushSizeChange,
  overlayOpacity,
  onOverlayOpacityChange,
  erasing,
  onToggleEraser,
  zoom,
  onResetZoom,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onSkip,
  onSubmit,
  annotationCount,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800 bg-zinc-900 px-3 py-2 md:gap-4 md:px-4">
      {/* Brush size */}
      <div className="flex items-center gap-1.5 md:gap-2">
        <label className="hidden text-xs text-zinc-500 md:inline">Brush</label>
        <input
          type="range"
          min={2}
          max={50}
          value={brushSize}
          onChange={(e) => onBrushSizeChange(Number(e.target.value))}
          className="w-16 accent-zinc-400 md:w-20"
        />
        <span className="w-6 text-right font-mono text-xs text-zinc-400">
          {brushSize}
        </span>
      </div>

      {/* Overlay opacity — hidden on mobile */}
      <div className="hidden items-center gap-2 md:flex">
        <label className="text-xs text-zinc-500">Overlay</label>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(overlayOpacity * 100)}
          onChange={(e) => onOverlayOpacityChange(Number(e.target.value) / 100)}
          className="w-20 accent-zinc-400"
        />
        <span className="w-8 text-right font-mono text-xs text-zinc-400">
          {Math.round(overlayOpacity * 100)}%
        </span>
      </div>

      {/* Eraser */}
      <button
        onClick={onToggleEraser}
        className={`rounded px-2 py-1 text-xs transition-colors ${
          erasing
            ? "bg-zinc-600 text-white"
            : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        }`}
        title="Eraser (x)"
      >
        Eraser
      </button>

      <div className="hidden h-4 w-px bg-zinc-700 md:block" />

      {/* Undo/Redo */}
      <button
        onClick={onUndo}
        disabled={!canUndo}
        className="rounded px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
        title="Undo"
      >
        Undo
      </button>
      <button
        onClick={onRedo}
        disabled={!canRedo}
        className="rounded px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
        title="Redo"
      >
        Redo
      </button>

      {/* Zoom — hidden on mobile (use pinch instead) */}
      <div className="hidden h-4 w-px bg-zinc-700 md:block" />
      <div className="hidden items-center gap-2 md:flex">
        <span className="font-mono text-xs text-zinc-500">
          {Math.round(zoom * 100)}%
        </span>
        <button
          onClick={onResetZoom}
          className="rounded px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
        >
          Reset
        </button>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Annotation count — hidden on mobile */}
      <span className="hidden text-xs text-zinc-500 md:inline">
        {annotationCount} annotated
      </span>
      <div className="hidden h-4 w-px bg-zinc-700 md:block" />

      {/* Skip / Submit */}
      <button
        onClick={onSkip}
        className="rounded px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
      >
        Skip
      </button>
      <button
        onClick={onSubmit}
        className="rounded bg-blue-600 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-500"
      >
        Submit
      </button>
    </div>
  );
}
