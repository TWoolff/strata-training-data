"use client";

import { Suspense, useEffect, useState, useCallback, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { REGIONS, type Region } from "@/lib/regions";
import { Canvas, type CanvasHandle } from "@/components/Canvas";
import { RegionPalette } from "@/components/RegionPalette";

type AnnotationSummary = {
  annotation_id: number;
  image_id: number;
  user_id: number;
  time_spent: number | null;
  annotated_at: string;
  dataset: string;
  example_id: string;
  image_url: string;
  seg_url: string;
  width: number;
  height: number;
  annotator_name: string;
};

type AnnotationDetail = AnnotationSummary & {
  mask_data: string;
};

type Filters = {
  datasets: string[];
  annotators: { id: number; name: string }[];
};

/** Decode a grayscale PNG (base64 data URL or URL) to pixel array on a 512x512 canvas. */
function decodeGrayscalePng(
  src: string,
  width: number,
  height: number,
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, width, height);
      const data = ctx.getImageData(0, 0, width, height).data;
      // Extract R channel (grayscale = R = G = B)
      const gray = new Uint8Array(width * height);
      for (let i = 0; i < gray.length; i++) {
        gray[i] = data[i * 4];
      }
      resolve(gray);
    };
    img.onerror = reject;
    img.src = src;
  });
}

/** Render a grayscale region mask to a colored data URL for display. */
function renderColoredMask(
  gray: Uint8Array,
  width: number,
  height: number,
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const imgData = ctx.createImageData(width, height);
  for (let i = 0; i < gray.length; i++) {
    const regionId = gray[i];
    const region = REGIONS[regionId] ?? REGIONS[0];
    const [r, g, b] = region.color;
    imgData.data[i * 4] = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = regionId === 0 ? 0 : 180;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL("image/png");
}

/** Render a diff image highlighting pixels that changed between two masks. */
function renderDiffMask(
  original: Uint8Array,
  corrected: Uint8Array,
  width: number,
  height: number,
): { dataUrl: string; changedCount: number } {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const imgData = ctx.createImageData(width, height);
  let changedCount = 0;
  for (let i = 0; i < original.length; i++) {
    if (original[i] !== corrected[i]) {
      changedCount++;
      // Changed pixel: show corrected region color at full opacity
      const region = REGIONS[corrected[i]] ?? REGIONS[0];
      const [r, g, b] = region.color;
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    } else {
      // Unchanged: dim gray
      imgData.data[i * 4] = 40;
      imgData.data[i * 4 + 1] = 40;
      imgData.data[i * 4 + 2] = 40;
      imgData.data[i * 4 + 3] = corrected[i] === 0 ? 0 : 60;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return { dataUrl: canvas.toDataURL("image/png"), changedCount };
}

export default function ReviewPage() {
  return (
    <Suspense
      fallback={
        <div className="flex flex-1 items-center justify-center">
          <span className="text-sm text-zinc-500">Loading...</span>
        </div>
      }
    >
      <ReviewPageInner />
    </Suspense>
  );
}

function ReviewPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Auth
  const [authed, setAuthed] = useState(false);
  const [authInput, setAuthInput] = useState("");
  const [authError, setAuthError] = useState("");
  const reviewKey = useRef<string>("");

  // Data
  const [annotations, setAnnotations] = useState<AnnotationSummary[]>([]);
  const [filters, setFilters] = useState<Filters>({ datasets: [], annotators: [] });
  const [loading, setLoading] = useState(true);

  // Filters
  const [filterDataset, setFilterDataset] = useState("");
  const [filterAnnotator, setFilterAnnotator] = useState("");

  // Detail view
  const [detailIndex, setDetailIndex] = useState<number | null>(null);
  const [detail, setDetail] = useState<AnnotationDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [originalOverlay, setOriginalOverlay] = useState<string | null>(null);
  const [correctedOverlay, setCorrectedOverlay] = useState<string | null>(null);
  const [diffOverlay, setDiffOverlay] = useState<string | null>(null);
  const [changedPixels, setChangedPixels] = useState(0);

  // Edit mode
  const [editing, setEditing] = useState(false);
  const [activeRegion, setActiveRegion] = useState<Region>(REGIONS[1]);
  const [brushSize, setBrushSize] = useState(10);
  const [saving, setSaving] = useState(false);
  const editCanvasRef = useRef<CanvasHandle>(null);

  // Batch selection
  const [selected, setSelected] = useState<Set<number>>(new Set());

  // Check auth on mount
  useEffect(() => {
    const storedKey = localStorage.getItem("strata_review_key");
    const urlKey = searchParams.get("key");
    const key = urlKey || storedKey || "";
    if (key) {
      reviewKey.current = key;
      if (urlKey) {
        localStorage.setItem("strata_review_key", urlKey);
      }
      setAuthed(true);
    }
  }, [searchParams]);

  // Fetch annotations list
  const fetchAnnotations = useCallback(async () => {
    if (!reviewKey.current && !authed) return;
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (reviewKey.current) params.set("key", reviewKey.current);
      if (filterDataset) params.set("dataset", filterDataset);
      if (filterAnnotator) params.set("annotator", filterAnnotator);

      const res = await fetch(`/api/review?${params}`);
      if (res.status === 401) {
        setAuthed(false);
        localStorage.removeItem("strata_review_key");
        return;
      }
      const data = await res.json();
      setAnnotations(data.annotations ?? []);
      setFilters(data.filters ?? { datasets: [], annotators: [] });
    } catch (err) {
      console.error("Failed to fetch review list:", err);
    } finally {
      setLoading(false);
    }
  }, [authed, filterDataset, filterAnnotator]);

  useEffect(() => {
    if (authed) fetchAnnotations();
  }, [authed, fetchAnnotations]);

  // Auth form submit
  async function handleAuth(e: React.FormEvent) {
    e.preventDefault();
    const key = authInput.trim();
    if (!key) {
      setAuthError("Enter the review key");
      return;
    }
    // Test the key
    const res = await fetch(`/api/review?key=${encodeURIComponent(key)}`);
    if (res.status === 401) {
      setAuthError("Invalid key");
      return;
    }
    reviewKey.current = key;
    localStorage.setItem("strata_review_key", key);
    setAuthed(true);
  }

  // Open detail view
  const openDetail = useCallback(
    async (index: number) => {
      const ann = annotations[index];
      if (!ann) return;
      setDetailIndex(index);
      setDetailLoading(true);
      setOriginalOverlay(null);
      setCorrectedOverlay(null);
      setDiffOverlay(null);

      try {
        const params = new URLSearchParams({
          annotation_id: String(ann.annotation_id),
        });
        if (reviewKey.current) params.set("key", reviewKey.current);

        const res = await fetch(`/api/review?${params}`);
        const data = await res.json();
        const d = data.annotation as AnnotationDetail;
        setDetail(d);

        // Decode masks and compute overlays
        const w = d.width || 512;
        const h = d.height || 512;
        const [origGray, corrGray] = await Promise.all([
          decodeGrayscalePng(d.seg_url, w, h),
          decodeGrayscalePng(d.mask_data, w, h),
        ]);

        setOriginalOverlay(renderColoredMask(origGray, w, h));
        setCorrectedOverlay(renderColoredMask(corrGray, w, h));
        const diff = renderDiffMask(origGray, corrGray, w, h);
        setDiffOverlay(diff.dataUrl);
        setChangedPixels(diff.changedCount);
      } catch (err) {
        console.error("Failed to load annotation detail:", err);
      } finally {
        setDetailLoading(false);
      }
    },
    [annotations],
  );

  const closeDetail = useCallback(() => {
    setDetailIndex(null);
    setDetail(null);
    setEditing(false);
  }, []);

  // Save corrected mask from edit mode
  const saveCorrection = useCallback(async () => {
    if (!detail || !editCanvasRef.current) return;
    const maskData = editCanvasRef.current.getMaskAsGrayscalePng();
    if (!maskData) return;

    setSaving(true);
    try {
      const params = reviewKey.current ? `?key=${encodeURIComponent(reviewKey.current)}` : "";
      const res = await fetch(`/api/review${params}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          annotation_id: detail.annotation_id,
          mask_data: maskData,
        }),
      });
      const data = await res.json();
      if (data.success) {
        // Update local detail with new mask and refresh overlays
        setDetail((prev) => prev ? { ...prev, mask_data: maskData } : prev);
        setEditing(false);

        // Recompute overlays with the corrected mask
        const w = detail.width || 512;
        const h = detail.height || 512;
        const [origGray, corrGray] = await Promise.all([
          decodeGrayscalePng(detail.seg_url, w, h),
          decodeGrayscalePng(maskData, w, h),
        ]);
        setCorrectedOverlay(renderColoredMask(corrGray, w, h));
        const diff = renderDiffMask(origGray, corrGray, w, h);
        setDiffOverlay(diff.dataUrl);
        setChangedPixels(diff.changedCount);
      }
    } catch (err) {
      console.error("Failed to save correction:", err);
    } finally {
      setSaving(false);
    }
  }, [detail]);

  // Review action (approve/reject)
  const reviewAction = useCallback(
    async (approved: boolean) => {
      if (detailIndex === null || !detail) return;
      try {
        const res = await fetch(
          `/api/review${reviewKey.current ? `?key=${encodeURIComponent(reviewKey.current)}` : ""}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              annotation_id: detail.annotation_id,
              approved,
            }),
          },
        );
        const data = await res.json();
        if (!data.success) return;

        // Remove from list and advance
        setAnnotations((prev) => {
          const next = prev.filter(
            (a) => a.annotation_id !== detail.annotation_id,
          );
          if (detailIndex < next.length) {
            // Stay at same index (next item slides in)
            setTimeout(() => openDetail(detailIndex), 0);
          } else if (next.length > 0) {
            setTimeout(() => openDetail(next.length - 1), 0);
          } else {
            closeDetail();
          }
          return next;
        });
      } catch (err) {
        console.error("Failed to submit review:", err);
      }
    },
    [detail, detailIndex, openDetail, closeDetail],
  );

  // Skip to next/prev
  const navigateDetail = useCallback(
    (direction: 1 | -1) => {
      if (detailIndex === null) return;
      const next = detailIndex + direction;
      if (next >= 0 && next < annotations.length) {
        openDetail(next);
      }
    },
    [detailIndex, annotations.length, openDetail],
  );

  // Batch review
  const batchReview = useCallback(
    async (approved: boolean) => {
      if (selected.size === 0) return;
      const ids = Array.from(selected);
      try {
        await fetch(
          `/api/review${reviewKey.current ? `?key=${encodeURIComponent(reviewKey.current)}` : ""}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ annotation_ids: ids, approved }),
          },
        );
        setAnnotations((prev) =>
          prev.filter((a) => !selected.has(a.annotation_id)),
        );
        setSelected(new Set());
      } catch (err) {
        console.error("Batch review failed:", err);
      }
    },
    [selected],
  );

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (detailIndex !== null && !editing) {
        if (e.key === "a") {
          e.preventDefault();
          reviewAction(true);
        } else if (e.key === "r") {
          e.preventDefault();
          reviewAction(false);
        } else if (e.key === "e") {
          e.preventDefault();
          setEditing(true);
        } else if (e.key === "s" || e.key === "Escape") {
          e.preventDefault();
          closeDetail();
        } else if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          e.preventDefault();
          navigateDetail(1);
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          navigateDetail(-1);
        }
      } else if (editing && e.key === "Escape") {
        e.preventDefault();
        setEditing(false);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [detailIndex, editing, reviewAction, closeDetail, navigateDetail]);

  // Toggle selection
  function toggleSelect(annotationId: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(annotationId)) {
        next.delete(annotationId);
      } else {
        next.add(annotationId);
      }
      return next;
    });
  }

  function selectAll() {
    setSelected(new Set(annotations.map((a) => a.annotation_id)));
  }

  function selectNone() {
    setSelected(new Set());
  }

  // Auth gate
  if (!authed) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-4">
        <form
          onSubmit={handleAuth}
          className="flex w-full max-w-sm flex-col gap-4"
        >
          <h1 className="text-lg font-semibold text-zinc-50">
            Review Access
          </h1>
          <p className="text-sm text-zinc-500">
            Enter the review key to access QC review.
          </p>
          <input
            type="password"
            value={authInput}
            onChange={(e) => setAuthInput(e.target.value)}
            placeholder="Review key"
            autoFocus
            className="h-10 rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-50 placeholder-zinc-500 outline-none focus:border-blue-500"
          />
          {authError && (
            <p className="text-sm text-red-400">{authError}</p>
          )}
          <button
            type="submit"
            className="h-10 rounded-lg bg-blue-600 text-sm font-medium text-white hover:bg-blue-500"
          >
            Enter
          </button>
        </form>
      </div>
    );
  }

  // Detail view modal
  if (detailIndex !== null) {
    return (
      <div className="flex h-full flex-col">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
          <div className="flex items-center gap-3">
            <button
              onClick={editing ? () => setEditing(false) : closeDetail}
              className="rounded-lg px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              {editing ? "Back to review" : "Back to grid"}
            </button>
            {detail && (
              <span className="text-xs text-zinc-500">
                {detail.dataset}/{detail.example_id} by{" "}
                <span className="text-zinc-300">{detail.annotator_name}</span>
                {detail.time_spent !== null && (
                  <span className="ml-2 text-zinc-600">
                    {detail.time_spent}s
                  </span>
                )}
              </span>
            )}
            <span className="text-xs text-zinc-600">
              {detailIndex + 1} / {annotations.length}
            </span>
          </div>
          {!editing && (
            <div className="flex items-center gap-1 text-xs text-zinc-600">
              <kbd className="rounded border border-zinc-700 px-1.5 py-0.5">a</kbd>
              <span>approve</span>
              <kbd className="ml-2 rounded border border-zinc-700 px-1.5 py-0.5">r</kbd>
              <span>reject</span>
              <kbd className="ml-2 rounded border border-zinc-700 px-1.5 py-0.5">e</kbd>
              <span>edit</span>
              <kbd className="ml-2 rounded border border-zinc-700 px-1.5 py-0.5">&larr;&rarr;</kbd>
              <span>navigate</span>
            </div>
          )}
        </header>

        {editing && detail ? (
          <>
            {/* Edit mode: Canvas + RegionPalette */}
            <div className="flex flex-1 flex-col min-h-0 overflow-hidden md:flex-row">
              <Canvas
                ref={editCanvasRef}
                imageUrl={detail.image_url}
                segUrl={detail.mask_data}
                width={detail.width || 512}
                height={detail.height || 512}
                activeRegion={activeRegion}
                brushSize={brushSize}
                overlayOpacity={0.5}
              />
              <RegionPalette
                activeRegion={activeRegion}
                onRegionChange={setActiveRegion}
              />
            </div>

            {/* Edit toolbar */}
            <div className="flex items-center justify-center gap-3 border-t border-zinc-800 px-4 py-2">
              <div className="flex items-center gap-1.5">
                <label className="text-xs text-zinc-500">Brush</label>
                <input
                  type="range"
                  min={2}
                  max={50}
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="w-20 accent-zinc-400"
                />
                <span className="w-6 text-right font-mono text-xs text-zinc-400">
                  {brushSize}
                </span>
              </div>
              <div className="flex-1" />
              <button
                onClick={() => setEditing(false)}
                className="rounded-lg px-3 py-1.5 text-sm text-zinc-400 transition-colors hover:bg-zinc-800"
              >
                Cancel
              </button>
              <button
                onClick={saveCorrection}
                disabled={saving}
                className="rounded-lg bg-blue-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-50"
              >
                {saving ? "Saving..." : "Save correction"}
              </button>
            </div>
          </>
        ) : (
          <>
            {/* Three-panel comparison */}
            <div className="flex flex-1 items-center justify-center gap-4 overflow-auto p-4">
              {detailLoading ? (
                <span className="text-sm text-zinc-500">Loading...</span>
              ) : detail ? (
                <>
                  {/* Panel 1: Original + pseudo-label overlay */}
                  <div className="flex flex-col items-center gap-2">
                    <span className="text-xs font-medium text-zinc-400">
                      Original (pseudo-label)
                    </span>
                    <div
                      className="relative border border-zinc-800"
                      style={{ width: 320, height: 320 }}
                    >
                      <img
                        src={detail.image_url}
                        alt="Character"
                        className="absolute inset-0 h-full w-full object-contain"
                        crossOrigin="anonymous"
                      />
                      {originalOverlay && (
                        <img
                          src={originalOverlay}
                          alt="Pseudo-label overlay"
                          className="absolute inset-0 h-full w-full object-contain opacity-50"
                        />
                      )}
                    </div>
                  </div>

                  {/* Panel 2: Original + corrected overlay */}
                  <div className="flex flex-col items-center gap-2">
                    <span className="text-xs font-medium text-zinc-400">
                      Corrected (annotator)
                    </span>
                    <div
                      className="relative border border-zinc-800"
                      style={{ width: 320, height: 320 }}
                    >
                      <img
                        src={detail.image_url}
                        alt="Character"
                        className="absolute inset-0 h-full w-full object-contain"
                        crossOrigin="anonymous"
                      />
                      {correctedOverlay && (
                        <img
                          src={correctedOverlay}
                          alt="Corrected overlay"
                          className="absolute inset-0 h-full w-full object-contain opacity-50"
                        />
                      )}
                    </div>
                  </div>

                  {/* Panel 3: Diff */}
                  <div className="flex flex-col items-center gap-2">
                    <span className="text-xs font-medium text-zinc-400">
                      Diff ({changedPixels.toLocaleString()} px changed)
                    </span>
                    <div
                      className="relative border border-zinc-800"
                      style={{ width: 320, height: 320 }}
                    >
                      <img
                        src={detail.image_url}
                        alt="Character"
                        className="absolute inset-0 h-full w-full object-contain opacity-30"
                        crossOrigin="anonymous"
                      />
                      {diffOverlay && (
                        <img
                          src={diffOverlay}
                          alt="Diff overlay"
                          className="absolute inset-0 h-full w-full object-contain"
                        />
                      )}
                    </div>
                  </div>
                </>
              ) : null}
            </div>

            {/* Action buttons */}
            <div className="flex items-center justify-center gap-3 border-t border-zinc-800 px-4 py-3">
              <button
                onClick={() => navigateDetail(-1)}
                disabled={detailIndex === 0}
                className="rounded-lg px-3 py-1.5 text-sm text-zinc-400 transition-colors hover:bg-zinc-800 disabled:opacity-30"
              >
                Prev
              </button>
              <button
                onClick={() => reviewAction(false)}
                className="rounded-lg bg-red-600/80 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-500"
              >
                Reject
              </button>
              <button
                onClick={() => setEditing(true)}
                className="rounded-lg bg-zinc-700 px-4 py-1.5 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-600"
              >
                Edit
              </button>
              <button
                onClick={() => reviewAction(true)}
                className="rounded-lg bg-green-600/80 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-green-500"
              >
                Approve
              </button>
              <button
                onClick={() => navigateDetail(1)}
                disabled={detailIndex === annotations.length - 1}
                className="rounded-lg px-3 py-1.5 text-sm text-zinc-400 transition-colors hover:bg-zinc-800 disabled:opacity-30"
              >
                Next
              </button>
            </div>
          </>
        )}
      </div>
    );
  }

  // Grid view
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-semibold text-zinc-50">QC Review</h1>
          <span className="text-xs text-zinc-500">
            {annotations.length} pending review
          </span>
        </div>
        <div className="flex items-center gap-2">
          {/* Filters */}
          <select
            value={filterDataset}
            onChange={(e) => setFilterDataset(e.target.value)}
            className="h-7 rounded border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-300"
          >
            <option value="">All datasets</option>
            {filters.datasets.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
          <select
            value={filterAnnotator}
            onChange={(e) => setFilterAnnotator(e.target.value)}
            className="h-7 rounded border border-zinc-700 bg-zinc-900 px-2 text-xs text-zinc-300"
          >
            <option value="">All annotators</option>
            {filters.annotators.map((a) => (
              <option key={a.id} value={String(a.id)}>
                {a.name}
              </option>
            ))}
          </select>
          <Link
            href="/annotate"
            className="rounded-lg px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            Annotate
          </Link>
          <Link
            href="/leaderboard"
            className="rounded-lg px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            Leaderboard
          </Link>
        </div>
      </header>

      {/* Batch actions bar */}
      {selected.size > 0 && (
        <div className="flex items-center gap-3 border-b border-zinc-800 bg-zinc-900/50 px-4 py-2">
          <span className="text-xs text-zinc-400">
            {selected.size} selected
          </span>
          <button
            onClick={() => batchReview(true)}
            className="rounded-lg bg-green-600/80 px-3 py-1 text-xs font-medium text-white hover:bg-green-500"
          >
            Approve all
          </button>
          <button
            onClick={() => batchReview(false)}
            className="rounded-lg bg-red-600/80 px-3 py-1 text-xs font-medium text-white hover:bg-red-500"
          >
            Reject all
          </button>
          <button
            onClick={selectNone}
            className="rounded-lg px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            Clear
          </button>
          <button
            onClick={selectAll}
            className="rounded-lg px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            Select all
          </button>
        </div>
      )}

      {/* Grid */}
      <div className="flex-1 overflow-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <span className="text-sm text-zinc-500">Loading...</span>
          </div>
        ) : annotations.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <div className="flex flex-col items-center gap-2 text-center">
              <div className="text-3xl">&#10003;</div>
              <h2 className="text-lg font-semibold text-zinc-200">
                All reviewed!
              </h2>
              <p className="text-sm text-zinc-500">
                No annotations waiting for review.
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
            {annotations.map((ann, i) => {
              const isSelected = selected.has(ann.annotation_id);
              return (
                <div
                  key={ann.annotation_id}
                  className={`group relative cursor-pointer overflow-hidden rounded-lg border transition-colors ${
                    isSelected
                      ? "border-blue-500 bg-blue-600/10"
                      : "border-zinc-800 hover:border-zinc-600"
                  }`}
                >
                  {/* Checkbox */}
                  <div
                    className="absolute top-2 left-2 z-10"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleSelect(ann.annotation_id);
                    }}
                  >
                    <div
                      className={`flex h-5 w-5 items-center justify-center rounded border text-xs ${
                        isSelected
                          ? "border-blue-500 bg-blue-600 text-white"
                          : "border-zinc-600 bg-zinc-900/80 text-transparent group-hover:border-zinc-400"
                      }`}
                    >
                      {isSelected && "✓"}
                    </div>
                  </div>

                  {/* Thumbnail */}
                  <div
                    className="relative aspect-square"
                    onClick={() => openDetail(i)}
                  >
                    <img
                      src={ann.image_url}
                      alt={ann.example_id}
                      className="h-full w-full object-contain"
                      crossOrigin="anonymous"
                      loading="lazy"
                    />
                  </div>

                  {/* Info badge */}
                  <div
                    className="flex items-center justify-between px-2 py-1.5"
                    onClick={() => openDetail(i)}
                  >
                    <span className="truncate text-xs text-zinc-400">
                      {ann.annotator_name}
                    </span>
                    <span className="text-xs text-zinc-600">
                      {ann.dataset}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
