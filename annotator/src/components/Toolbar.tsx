"use client";

type Props = {
  brushSize: number;
  onBrushSizeChange: (size: number) => void;
  overlayOpacity: number;
  onOverlayOpacityChange: (opacity: number) => void;
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
    <div className="flex items-center gap-4 border-t border-zinc-800 bg-zinc-900 px-4 py-2">
      {/* Brush size */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-zinc-500">Brush</label>
        <input
          type="range"
          min={2}
          max={50}
          value={brushSize}
          onChange={(e) => onBrushSizeChange(Number(e.target.value))}
          className="w-20 accent-zinc-400"
        />
        <span className="w-6 text-right font-mono text-xs text-zinc-400">
          {brushSize}
        </span>
      </div>

      {/* Overlay opacity */}
      <div className="flex items-center gap-2">
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

      <div className="h-4 w-px bg-zinc-700" />

      {/* Undo/Redo */}
      <button
        onClick={onUndo}
        disabled={!canUndo}
        className="rounded px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
        title="Undo (Ctrl+Z)"
      >
        Undo
      </button>
      <button
        onClick={onRedo}
        disabled={!canRedo}
        className="rounded px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:hover:bg-transparent"
        title="Redo (Ctrl+Shift+Z)"
      >
        Redo
      </button>

      <div className="h-4 w-px bg-zinc-700" />

      {/* Zoom */}
      <div className="flex items-center gap-2">
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

      {/* Annotation count */}
      <span className="text-xs text-zinc-500">
        {annotationCount} annotated
      </span>

      <div className="h-4 w-px bg-zinc-700" />

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
