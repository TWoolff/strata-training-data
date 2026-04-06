"use client";

import { BodyMap } from "./BodyMap";

type OnboardingGuideProps = {
  onDismiss: () => void;
};

export function OnboardingGuide({ onDismiss }: OnboardingGuideProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="flex max-h-[90vh] w-full max-w-lg flex-col gap-6 overflow-y-auto rounded-xl border border-zinc-700 bg-zinc-900 p-6 shadow-2xl">
        {/* Header */}
        <div className="flex flex-col gap-2 text-center">
          <h2 className="text-2xl font-bold text-zinc-50">
            How Annotation Works
          </h2>
          <p className="text-sm text-zinc-400">
            Paint over the colored regions to correct mistakes. Each color = one
            body part.
          </p>
        </div>

        {/* Body map */}
        <BodyMap />

        {/* Instructions */}
        <div className="flex flex-col gap-2 text-sm text-zinc-300">
          <p>
            <strong className="text-zinc-100">Your task:</strong> review
            AI-generated segmentation masks and fix any mislabeled regions.
          </p>
          <p>
            Select a body region from the palette (or press its keyboard
            shortcut), then paint over pixels that should belong to that region.
          </p>
        </div>

        {/* Dismiss button */}
        <button
          onClick={onDismiss}
          className="h-11 rounded-lg bg-blue-600 text-base font-medium text-white transition-colors hover:bg-blue-500"
        >
          Got it!
        </button>
      </div>
    </div>
  );
}
