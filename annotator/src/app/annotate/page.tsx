"use client";

import { useEffect, useState, useRef, useCallback, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { OnboardingGuide } from "@/components/OnboardingGuide";
import { Canvas, type CanvasHandle } from "@/components/Canvas";
import { RegionPalette } from "@/components/RegionPalette";
import { Toolbar } from "@/components/Toolbar";
import { ProgressBar } from "@/components/ProgressBar";
import { ToastContainer, showToast } from "@/components/Toast";
import { REGIONS, regionByShortcut, type Region } from "@/lib/regions";

type ImageData = {
  id: number;
  dataset: string;
  example_id: string;
  image_url: string;
  seg_url: string;
  width: number;
  height: number;
};

export default function AnnotatePage() {
  const router = useRouter();
  const canvasRef = useRef<CanvasHandle>(null);
  const loadTimeRef = useRef(Date.now());
  const imageIdRef = useRef<number | null>(null);

  const [showOnboarding, setShowOnboarding] = useState(false);
  const [activeRegion, setActiveRegion] = useState<Region>(REGIONS[1]); // head
  const [erasing, setErasing] = useState(false);
  const prevRegionRef = useRef<Region>(REGIONS[1]);
  const [brushSize, setBrushSize] = useState(15);
  const [overlayOpacity, setOverlayOpacity] = useState(0.4);
  const [zoom, setZoom] = useState(1);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [currentImage, setCurrentImage] = useState<ImageData | null>(null);
  const [loading, setLoading] = useState(true);
  const [annotationCount, setAnnotationCount] = useState(0);
  const [streak, setStreak] = useState(0);
  const [totalImages, setTotalImages] = useState(0);
  const [pendingImages, setPendingImages] = useState(0);
  const [totalAnnotations, setTotalAnnotations] = useState(0);

  const userName = useSyncExternalStore(
    () => () => {},
    () => localStorage.getItem("strata_user_name"),
    () => null,
  );

  const userId = useSyncExternalStore(
    () => () => {},
    () => localStorage.getItem("strata_user_id"),
    () => null,
  );

  const hasReviewKey = useSyncExternalStore(
    () => () => {},
    () => !!localStorage.getItem("strata_review_key"),
    () => false,
  );

  useEffect(() => {
    if (!userId) {
      router.replace("/");
    }
  }, [userId, router]);

  // Fetch next pending image
  const fetchNextImage = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/images?status=pending");
      const data = await res.json();
      const img = data.image ?? null;
      setCurrentImage(img);
      imageIdRef.current = img?.id ?? null;
      loadTimeRef.current = Date.now();
    } catch (err) {
      console.error("Failed to fetch image:", err);
      setCurrentImage(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (userId) fetchNextImage();
  }, [userId, fetchNextImage]);

  // Fetch global stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      setTotalImages(data.total ?? 0);
      setPendingImages(data.pending ?? 0);
    } catch {
      // silent
    }
  }, []);

  // Fetch total annotation count across all users
  const fetchTotalAnnotations = useCallback(async () => {
    try {
      const res = await fetch("/api/leaderboard");
      const data = await res.json();
      const sum = (data.entries ?? []).reduce(
        (acc: number, e: { annotated: number }) => acc + e.annotated,
        0,
      );
      setTotalAnnotations(sum);
    } catch {
      // silent
    }
  }, []);

  useEffect(() => {
    if (userId) {
      fetchStats();
      fetchTotalAnnotations();
    }
  }, [userId, fetchStats, fetchTotalAnnotations]);

  // Submit annotation
  const handleSubmit = useCallback(async () => {
    const submitImageId = imageIdRef.current;
    if (!submitImageId || !userId || !canvasRef.current) return;
    const maskData = canvasRef.current.getMaskAsGrayscalePng();
    if (!maskData) return;

    const timeSpent = Math.round((Date.now() - loadTimeRef.current) / 1000);

    try {
      const res = await fetch("/api/annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: submitImageId,
          user_id: Number(userId),
          mask_data: maskData,
          time_spent: timeSpent,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setAnnotationCount(data.count);
        const newStreak = streak + 1;
        setStreak(newStreak);

        // Streak toasts
        const MILESTONES = [50, 25, 10, 5];
        const milestone = MILESTONES.find((m) => newStreak === m);
        if (milestone) {
          showToast(`${milestone} streak! You're on fire`);
        } else if (newStreak === 1) {
          showToast("Nice! 1 done");
        } else {
          showToast(`${newStreak} in a row!`);
        }

        // Speed toast
        if (timeSpent <= 30 && timeSpent > 0) {
          setTimeout(() => showToast(`Speed demon! Only ${timeSpent}s`), 300);
        }

        // Refresh stats
        fetchStats();
        fetchTotalAnnotations();
      }
    } catch (err) {
      console.error("Failed to submit annotation:", err);
    }

    fetchNextImage();
  }, [userId, streak, fetchNextImage, fetchStats, fetchTotalAnnotations]);

  // Skip image
  const handleSkip = useCallback(() => {
    fetchNextImage();
  }, [fetchNextImage]);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Don't handle if typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Undo: Ctrl+Z / Cmd+Z
      if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        canvasRef.current?.undo();
        return;
      }
      // Redo: Ctrl+Shift+Z / Cmd+Shift+Z
      if ((e.ctrlKey || e.metaKey) && e.key === "z" && e.shiftKey) {
        e.preventDefault();
        canvasRef.current?.redo();
        return;
      }

      // Eraser toggle: x
      if (e.key === "x") {
        setErasing((prev) => {
          if (!prev) {
            prevRegionRef.current = activeRegion;
            setActiveRegion(REGIONS[0]); // background
          } else {
            setActiveRegion(prevRegionRef.current);
          }
          return !prev;
        });
        return;
      }

      // Brush size: [ and ]
      if (e.key === "[") {
        setBrushSize((s) => Math.max(2, s - 3));
        return;
      }
      if (e.key === "]") {
        setBrushSize((s) => Math.min(50, s + 3));
        return;
      }

      // Region shortcuts
      const region = regionByShortcut(e.key);
      if (region) {
        setActiveRegion(region);
        setErasing(false);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Listen for eyedropper picks from canvas
  useEffect(() => {
    function handleRegionPick(e: Event) {
      const regionId = (e as CustomEvent).detail;
      if (typeof regionId === "number" && REGIONS[regionId]) {
        setActiveRegion(REGIONS[regionId]);
      }
    }
    window.addEventListener("regionpick", handleRegionPick);
    return () => window.removeEventListener("regionpick", handleRegionPick);
  }, []);

  function handleLogout() {
    localStorage.removeItem("strata_user_id");
    localStorage.removeItem("strata_user_name");
    router.replace("/");
  }

  function handleShowOnboarding() {
    setShowOnboarding(true);
  }

  function handleDismissOnboarding() {
    localStorage.setItem("strata_onboarding_done", "1");
    setShowOnboarding(false);
  }

  if (!userId || !userName) return null;

  return (
    <div className="grid" style={{ height: "100dvh", gridTemplateRows: "auto 1fr auto" }}>
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-semibold text-zinc-50">Strata Label</h1>
          {currentImage && (
            <span className="text-xs text-zinc-500">
              {currentImage.dataset}/{currentImage.example_id}
            </span>
          )}
          {totalImages > 0 && (
            <ProgressBar
              totalImages={totalImages}
              pendingImages={pendingImages}
              personalCount={annotationCount}
            />
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">
            {userName}
          </span>
          <span className="text-xs text-zinc-600">
            {totalAnnotations} total
          </span>
          {hasReviewKey && (
            <Link
              href="/review"
              className="rounded-lg px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
            >
              Review
            </Link>
          )}
          <Link
            href="/leaderboard"
            className="rounded-lg px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            Leaderboard
          </Link>
          <button
            onClick={handleShowOnboarding}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
            title="Show help guide"
          >
            ?
          </button>
          <button
            onClick={handleLogout}
            className="rounded-lg px-2 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* Main content: Canvas + Palette */}
      <div className="flex min-h-0 overflow-hidden">
        {loading ? (
          <div className="flex flex-1 items-center justify-center">
            <span className="text-sm text-zinc-500">Loading...</span>
          </div>
        ) : !currentImage ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="text-3xl">&#10003;</div>
              <h2 className="text-lg font-semibold text-zinc-200">
                All caught up!
              </h2>
              <p className="max-w-sm text-sm text-zinc-500">
                No pending images to annotate. Check back later or ask an admin
                to add more images.
              </p>
            </div>
          </div>
        ) : (
          <>
            <Canvas
              ref={canvasRef}
              imageUrl={currentImage.image_url}
              segUrl={currentImage.seg_url}
              width={currentImage.width ?? 512}
              height={currentImage.height ?? 512}
              activeRegion={activeRegion}
              brushSize={brushSize}
              overlayOpacity={overlayOpacity}
              onZoomChange={setZoom}
              onUndoRedoChange={(u, r) => {
                setCanUndo(u);
                setCanRedo(r);
              }}
            />
            <RegionPalette
              activeRegion={activeRegion}
              onRegionChange={(r) => {
                setActiveRegion(r);
                setErasing(false);
              }}
            />
          </>
        )}
      </div>

      {/* Toolbar */}
      {currentImage && (
        <Toolbar
          brushSize={brushSize}
          onBrushSizeChange={setBrushSize}
          overlayOpacity={overlayOpacity}
          onOverlayOpacityChange={setOverlayOpacity}
          erasing={erasing}
          onToggleEraser={() => {
            setErasing((prev) => {
              if (!prev) {
                prevRegionRef.current = activeRegion;
                setActiveRegion(REGIONS[0]);
              } else {
                setActiveRegion(prevRegionRef.current);
              }
              return !prev;
            });
          }}
          zoom={zoom}
          onResetZoom={() => canvasRef.current?.resetZoom()}
          canUndo={canUndo}
          canRedo={canRedo}
          onUndo={() => canvasRef.current?.undo()}
          onRedo={() => canvasRef.current?.redo()}
          onSkip={handleSkip}
          onSubmit={handleSubmit}
          annotationCount={annotationCount}
        />
      )}

      {showOnboarding && (
        <OnboardingGuide onDismiss={handleDismissOnboarding} />
      )}

      <ToastContainer />
    </div>
  );
}
