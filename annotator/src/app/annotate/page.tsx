"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";
import { OnboardingGuide } from "@/components/OnboardingGuide";

export default function AnnotatePage() {
  const router = useRouter();
  const [showOnboarding, setShowOnboarding] = useState(false);

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

  useEffect(() => {
    if (!userId) {
      router.replace("/");
    }
  }, [userId, router]);

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
    <div className="flex flex-1 flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold text-zinc-50">Strata Label</h1>
          <span className="text-sm text-zinc-500">
            Signed in as{" "}
            <span className="text-zinc-300">{userName}</span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleShowOnboarding}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
            title="Show help guide"
          >
            ?
          </button>
          <button
            onClick={handleLogout}
            className="rounded-lg px-3 py-1.5 text-sm text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* Placeholder content */}
      <div className="flex flex-1 items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-zinc-800 text-2xl">
            🎨
          </div>
          <h2 className="text-xl font-semibold text-zinc-200">
            Annotation Canvas
          </h2>
          <p className="max-w-sm text-sm text-zinc-500">
            The annotation interface will be built here. You&apos;ll be able to
            view character images and paint over segmentation masks to correct
            body region labels.
          </p>
        </div>
      </div>

      {showOnboarding && (
        <OnboardingGuide onDismiss={handleDismissOnboarding} />
      )}
    </div>
  );
}
