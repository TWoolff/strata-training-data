"use client";

import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from "react";
import { REGIONS, type Region } from "@/lib/regions";

const MAX_UNDO = 20;

export type CanvasHandle = {
  getMaskAsGrayscalePng: () => string | null;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  resetZoom: () => void;
  getZoom: () => number;
};

type Props = {
  imageUrl: string | null;
  segUrl: string | null;
  width: number;
  height: number;
  activeRegion: Region;
  brushSize: number;
  overlayOpacity: number;
  onZoomChange?: (zoom: number) => void;
  onUndoRedoChange?: (canUndo: boolean, canRedo: boolean) => void;
};

/** Build a lookup: grayscale pixel value -> RGBA color for overlay. */
function buildGrayscaleToColor(): Uint8Array {
  const lut = new Uint8Array(256 * 4);
  for (const region of REGIONS) {
    const offset = region.id * 4;
    lut[offset] = region.color[0];
    lut[offset + 1] = region.color[1];
    lut[offset + 2] = region.color[2];
    lut[offset + 3] = 255;
  }
  return lut;
}

/** Build a lookup: RGBA color -> region ID (using R,G,B as key). */
function buildColorToId(): Map<number, number> {
  const map = new Map<number, number>();
  for (const region of REGIONS) {
    const key = (region.color[0] << 16) | (region.color[1] << 8) | region.color[2];
    map.set(key, region.id);
  }
  return map;
}

const GRAYSCALE_TO_COLOR = buildGrayscaleToColor();
const COLOR_TO_ID = buildColorToId();

export const Canvas = forwardRef<CanvasHandle, Props>(function Canvas(
  {
    imageUrl,
    segUrl,
    width,
    height,
    activeRegion,
    brushSize,
    overlayOpacity,
    onZoomChange,
    onUndoRedoChange,
  },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const cursorCanvasRef = useRef<HTMLCanvasElement>(null);

  const undoStack = useRef<ImageData[]>([]);
  const redoStack = useRef<ImageData[]>([]);
  const isPainting = useRef(false);
  const isPanning = useRef(false);
  const spaceDown = useRef(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  const zoom = useRef(1);
  const panOffset = useRef({ x: 0, y: 0 });

  // Stable refs for callbacks so they never trigger effect re-runs
  const onUndoRedoChangeRef = useRef(onUndoRedoChange);
  onUndoRedoChangeRef.current = onUndoRedoChange;
  const onZoomChangeRef = useRef(onZoomChange);
  onZoomChangeRef.current = onZoomChange;

  const notifyUndoRedo = useCallback(() => {
    onUndoRedoChangeRef.current?.(undoStack.current.length > 0, redoStack.current.length > 0);
  }, []);

  const pushUndo = useCallback(() => {
    const ctx = maskCanvasRef.current?.getContext("2d");
    if (!ctx) return;
    const snap = ctx.getImageData(0, 0, width, height);
    undoStack.current.push(snap);
    if (undoStack.current.length > MAX_UNDO) undoStack.current.shift();
    redoStack.current = [];
    notifyUndoRedo();
  }, [width, height, notifyUndoRedo]);

  const undo = useCallback(() => {
    const ctx = maskCanvasRef.current?.getContext("2d");
    if (!ctx || undoStack.current.length === 0) return;
    const current = ctx.getImageData(0, 0, width, height);
    redoStack.current.push(current);
    const prev = undoStack.current.pop()!;
    ctx.putImageData(prev, 0, 0);
    notifyUndoRedo();
  }, [width, height, notifyUndoRedo]);

  const redo = useCallback(() => {
    const ctx = maskCanvasRef.current?.getContext("2d");
    if (!ctx || redoStack.current.length === 0) return;
    const current = ctx.getImageData(0, 0, width, height);
    undoStack.current.push(current);
    const next = redoStack.current.pop()!;
    ctx.putImageData(next, 0, 0);
    notifyUndoRedo();
  }, [width, height, notifyUndoRedo]);

  const applyTransform = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const inner = container.querySelector("[data-canvas-inner]") as HTMLElement;
    if (!inner) return;
    inner.style.transform = `translate(${panOffset.current.x}px, ${panOffset.current.y}px) scale(${zoom.current})`;
  }, []);

  const fitToView = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const padding = 32;
    const scaleX = (rect.width - padding) / width;
    const scaleY = (rect.height - padding) / height;
    const fit = Math.min(scaleX, scaleY, 4);
    zoom.current = fit;
    panOffset.current = { x: 0, y: 0 };
    applyTransform();
    onZoomChangeRef.current?.(fit);
  }, [width, height, applyTransform]);

  const resetZoom = useCallback(() => {
    fitToView();
  }, [fitToView]);

  // Fit canvas to viewport on mount and when image changes
  useEffect(() => {
    // Small delay to ensure container has layout dimensions
    const id = requestAnimationFrame(() => fitToView());
    return () => cancelAnimationFrame(id);
  }, [fitToView, imageUrl]);

  // Load character image onto background canvas
  useEffect(() => {
    if (!imageUrl) return;
    const canvas = bgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
    };
    img.src = imageUrl;
  }, [imageUrl, width, height]);

  // Load segmentation mask and convert to colored overlay
  useEffect(() => {
    if (!segUrl) return;
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Draw grayscale seg to a temp canvas to read pixel values
      const tmp = document.createElement("canvas");
      tmp.width = width;
      tmp.height = height;
      const tmpCtx = tmp.getContext("2d")!;
      tmpCtx.drawImage(img, 0, 0, width, height);
      const srcData = tmpCtx.getImageData(0, 0, width, height);

      // Convert grayscale region IDs to colored overlay
      const dstData = ctx.createImageData(width, height);
      for (let i = 0; i < srcData.data.length; i += 4) {
        const regionId = srcData.data[i]; // grayscale value = region ID
        const lutOffset = regionId * 4;
        dstData.data[i] = GRAYSCALE_TO_COLOR[lutOffset];
        dstData.data[i + 1] = GRAYSCALE_TO_COLOR[lutOffset + 1];
        dstData.data[i + 2] = GRAYSCALE_TO_COLOR[lutOffset + 2];
        dstData.data[i + 3] = 255;
      }
      ctx.putImageData(dstData, 0, 0);

      // Reset undo/redo
      undoStack.current = [];
      redoStack.current = [];
      notifyUndoRedo();
    };
    img.src = segUrl;
  }, [segUrl, width, height, notifyUndoRedo]);

  // Convert client coordinates to canvas coordinates
  const canvasCoords = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } | null => {
      const canvas = maskCanvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = (clientX - rect.left) / (rect.width / width);
      const y = (clientY - rect.top) / (rect.height / height);
      return { x, y };
    },
    [width, height],
  );

  // Paint a circle at position
  const paintAt = useCallback(
    (x: number, y: number) => {
      const ctx = maskCanvasRef.current?.getContext("2d");
      if (!ctx) return;
      const [r, g, b] = activeRegion.color;
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.beginPath();
      ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
      ctx.fill();
    },
    [activeRegion, brushSize],
  );

  // Paint a line between two points (for smooth strokes)
  const paintLine = useCallback(
    (x0: number, y0: number, x1: number, y1: number) => {
      const dx = x1 - x0;
      const dy = y1 - y0;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const steps = Math.max(1, Math.ceil(dist / (brushSize / 4)));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        paintAt(x0 + dx * t, y0 + dy * t);
      }
    },
    [paintAt, brushSize],
  );

  // Draw cursor preview
  const drawCursor = useCallback(
    (x: number, y: number) => {
      const canvas = cursorCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);
      const [r, g, b] = activeRegion.color;
      ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.lineWidth = 1.5 / zoom.current;
      ctx.beginPath();
      ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
      ctx.stroke();
    },
    [activeRegion, brushSize, width, height],
  );

  // Mouse/pointer event handlers
  useEffect(() => {
    const cursor = cursorCanvasRef.current;
    if (!cursor) return;

    // --- Mouse handlers ---

    function handleMouseDown(e: MouseEvent) {
      if (e.button === 1 || (e.button === 0 && spaceDown.current)) {
        isPanning.current = true;
        lastPos.current = { x: e.clientX, y: e.clientY };
        e.preventDefault();
        return;
      }
      if (e.button === 2) {
        e.preventDefault();
        return;
      }
      if (e.button === 0) {
        const pos = canvasCoords(e.clientX, e.clientY);
        if (!pos) return;
        pushUndo();
        isPainting.current = true;
        paintAt(pos.x, pos.y);
        lastPos.current = pos;
      }
    }

    function handleMouseMove(e: MouseEvent) {
      const pos = canvasCoords(e.clientX, e.clientY);

      if (isPanning.current && lastPos.current) {
        panOffset.current.x += e.clientX - lastPos.current.x;
        panOffset.current.y += e.clientY - lastPos.current.y;
        lastPos.current = { x: e.clientX, y: e.clientY };
        applyTransform();
        return;
      }

      if (pos) drawCursor(pos.x, pos.y);

      if (isPainting.current && pos) {
        if (lastPos.current) {
          paintLine(lastPos.current.x, lastPos.current.y, pos.x, pos.y);
        } else {
          paintAt(pos.x, pos.y);
        }
        lastPos.current = pos;
      }
    }

    function handleMouseUp(e: MouseEvent) {
      if (e.button === 2) {
        e.preventDefault();
        return;
      }
      isPainting.current = false;
      isPanning.current = false;
      lastPos.current = null;
    }

    function handleContextMenu(e: MouseEvent) {
      e.preventDefault();
      const pos = canvasCoords(e.clientX, e.clientY);
      if (!pos) return;
      const ctx = maskCanvasRef.current?.getContext("2d");
      if (!ctx) return;
      const pixel = ctx.getImageData(Math.floor(pos.x), Math.floor(pos.y), 1, 1).data;
      const key = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];
      const regionId = COLOR_TO_ID.get(key);
      if (regionId !== undefined) {
        cursor!.dispatchEvent(
          new CustomEvent("regionpick", { detail: regionId, bubbles: true }),
        );
      }
    }

    function handleWheel(e: WheelEvent) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.25, Math.min(10, zoom.current * delta));

      const container = containerRef.current;
      if (container) {
        const rect = container.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const scale = newZoom / zoom.current;
        panOffset.current.x = cx - scale * (cx - panOffset.current.x);
        panOffset.current.y = cy - scale * (cy - panOffset.current.y);
      }

      zoom.current = newZoom;
      applyTransform();
      onZoomChangeRef.current?.(newZoom);
    }

    // --- Touch handlers ---

    const touchStartDist = { current: 0 };
    const touchStartZoom = { current: 1 };

    function getTouchDist(a: Touch, b: Touch) {
      const dx = a.clientX - b.clientX;
      const dy = a.clientY - b.clientY;
      return Math.sqrt(dx * dx + dy * dy);
    }

    function getTouchMid(a: Touch, b: Touch) {
      return {
        x: (a.clientX + b.clientX) / 2,
        y: (a.clientY + b.clientY) / 2,
      };
    }

    function handleTouchStart(e: TouchEvent) {
      e.preventDefault();
      if (e.touches.length === 1) {
        const t = e.touches[0];
        const pos = canvasCoords(t.clientX, t.clientY);
        if (!pos) return;
        pushUndo();
        isPainting.current = true;
        paintAt(pos.x, pos.y);
        lastPos.current = pos;
      } else if (e.touches.length === 2) {
        isPainting.current = false;
        isPanning.current = true;
        touchStartDist.current = getTouchDist(e.touches[0], e.touches[1]);
        touchStartZoom.current = zoom.current;
        const mid = getTouchMid(e.touches[0], e.touches[1]);
        lastPos.current = { x: mid.x, y: mid.y };
      }
    }

    function handleTouchMove(e: TouchEvent) {
      e.preventDefault();
      if (e.touches.length === 1 && isPainting.current) {
        const t = e.touches[0];
        const pos = canvasCoords(t.clientX, t.clientY);
        if (!pos) return;
        if (lastPos.current) {
          paintLine(lastPos.current.x, lastPos.current.y, pos.x, pos.y);
        } else {
          paintAt(pos.x, pos.y);
        }
        lastPos.current = pos;
      } else if (e.touches.length === 2 && isPanning.current) {
        const dist = getTouchDist(e.touches[0], e.touches[1]);
        const mid = getTouchMid(e.touches[0], e.touches[1]);

        // Pinch zoom
        const newZoom = Math.max(
          0.25,
          Math.min(10, touchStartZoom.current * (dist / touchStartDist.current)),
        );

        // Pan
        if (lastPos.current) {
          const container = containerRef.current;
          if (container) {
            const rect = container.getBoundingClientRect();
            const cx = mid.x - rect.left;
            const cy = mid.y - rect.top;
            const scale = newZoom / zoom.current;
            panOffset.current.x = cx - scale * (cx - panOffset.current.x);
            panOffset.current.y = cy - scale * (cy - panOffset.current.y);
          }
          panOffset.current.x += mid.x - lastPos.current.x;
          panOffset.current.y += mid.y - lastPos.current.y;
        }

        zoom.current = newZoom;
        lastPos.current = { x: mid.x, y: mid.y };
        applyTransform();
        onZoomChangeRef.current?.(newZoom);
      }
    }

    function handleTouchEnd(e: TouchEvent) {
      if (e.touches.length === 0) {
        isPainting.current = false;
        isPanning.current = false;
        lastPos.current = null;
      } else if (e.touches.length === 1 && isPanning.current) {
        // Went from 2 fingers to 1 — start painting with remaining finger
        isPanning.current = false;
        const t = e.touches[0];
        const pos = canvasCoords(t.clientX, t.clientY);
        if (pos) {
          pushUndo();
          isPainting.current = true;
          paintAt(pos.x, pos.y);
          lastPos.current = pos;
        }
      }
    }

    // --- Attach listeners ---

    cursor.addEventListener("mousedown", handleMouseDown);
    cursor.addEventListener("mousemove", handleMouseMove);
    cursor.addEventListener("mouseup", handleMouseUp);
    cursor.addEventListener("mouseleave", handleMouseUp);
    cursor.addEventListener("contextmenu", handleContextMenu);
    cursor.addEventListener("wheel", handleWheel, { passive: false });
    cursor.addEventListener("touchstart", handleTouchStart, { passive: false });
    cursor.addEventListener("touchmove", handleTouchMove, { passive: false });
    cursor.addEventListener("touchend", handleTouchEnd);
    cursor.addEventListener("touchcancel", handleTouchEnd);

    return () => {
      cursor.removeEventListener("mousedown", handleMouseDown);
      cursor.removeEventListener("mousemove", handleMouseMove);
      cursor.removeEventListener("mouseup", handleMouseUp);
      cursor.removeEventListener("mouseleave", handleMouseUp);
      cursor.removeEventListener("contextmenu", handleContextMenu);
      cursor.removeEventListener("wheel", handleWheel);
      cursor.removeEventListener("touchstart", handleTouchStart);
      cursor.removeEventListener("touchmove", handleTouchMove);
      cursor.removeEventListener("touchend", handleTouchEnd);
      cursor.removeEventListener("touchcancel", handleTouchEnd);
    };
  }, [
    canvasCoords,
    paintAt,
    paintLine,
    drawCursor,
    pushUndo,
    applyTransform,
  ]);

  // Keyboard: space for pan mode
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.code === "Space") {
        e.preventDefault();
        spaceDown.current = true;
      }
    }
    function handleKeyUp(e: KeyboardEvent) {
      if (e.code === "Space") {
        spaceDown.current = false;
        isPanning.current = false;
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // Export mask as grayscale PNG (base64)
  const getMaskAsGrayscalePng = useCallback((): string | null => {
    const ctx = maskCanvasRef.current?.getContext("2d");
    if (!ctx) return null;
    const data = ctx.getImageData(0, 0, width, height);
    const out = ctx.createImageData(width, height);

    for (let i = 0; i < data.data.length; i += 4) {
      const key = (data.data[i] << 16) | (data.data[i + 1] << 8) | data.data[i + 2];
      const regionId = COLOR_TO_ID.get(key) ?? 0;
      out.data[i] = regionId;
      out.data[i + 1] = regionId;
      out.data[i + 2] = regionId;
      out.data[i + 3] = 255;
    }

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = width;
    tmpCanvas.height = height;
    tmpCanvas.getContext("2d")!.putImageData(out, 0, 0);
    return tmpCanvas.toDataURL("image/png");
  }, [width, height]);

  useImperativeHandle(
    ref,
    () => ({
      getMaskAsGrayscalePng,
      undo,
      redo,
      canUndo: () => undoStack.current.length > 0,
      canRedo: () => redoStack.current.length > 0,
      resetZoom,
      getZoom: () => zoom.current,
    }),
    [getMaskAsGrayscalePng, undo, redo, resetZoom, fitToView],
  );

  return (
    <div
      ref={containerRef}
      className="relative flex-1 overflow-hidden bg-zinc-950 cursor-none md:cursor-none"
    >
      <div
        data-canvas-inner=""
        className="absolute left-1/2 top-1/2 origin-center"
        style={{
          width,
          height,
          marginLeft: -width / 2,
          marginTop: -height / 2,
        }}
      >
        {/* Background: character image */}
        <canvas
          ref={bgCanvasRef}
          width={width}
          height={height}
          className="absolute inset-0"
        />
        {/* Mask: segmentation overlay */}
        <canvas
          ref={maskCanvasRef}
          width={width}
          height={height}
          className="absolute inset-0"
          style={{ opacity: overlayOpacity }}
        />
        {/* Cursor: brush preview + touch/mouse interaction target */}
        <canvas
          ref={cursorCanvasRef}
          width={width}
          height={height}
          className="absolute inset-0"
          style={{ touchAction: "none" }}
        />
      </div>
    </div>
  );
});
