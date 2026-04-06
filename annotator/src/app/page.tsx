"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { OnboardingGuide } from "@/components/OnboardingGuide";

export default function Home() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Check if user is already logged in
  useEffect(() => {
    const userId = localStorage.getItem("strata_user_id");
    if (userId) {
      router.replace("/annotate");
    }
  }, [router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Please enter your name");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed }),
      });

      if (!res.ok) {
        const data = await res.json();
        setError(data.error || "Something went wrong");
        return;
      }

      const user = await res.json();
      localStorage.setItem("strata_user_id", String(user.id));
      localStorage.setItem("strata_user_name", user.name);

      // Show onboarding on first visit
      if (!localStorage.getItem("strata_onboarding_done")) {
        setShowOnboarding(true);
      } else {
        router.push("/annotate");
      }
    } catch {
      setError("Failed to connect. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function handleOnboardingDismiss() {
    localStorage.setItem("strata_onboarding_done", "1");
    setShowOnboarding(false);
    router.push("/annotate");
  }

  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4">
      <main className="flex w-full max-w-md flex-col items-center gap-8">
        {/* Logo / Title */}
        <div className="flex flex-col items-center gap-3 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-zinc-50">
            Strata Label
          </h1>
          <p className="max-w-sm text-lg text-zinc-400">
            Help train our AI to understand character anatomy
          </p>
        </div>

        {/* Name Input Form */}
        <form onSubmit={handleSubmit} className="flex w-full flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="name" className="text-sm font-medium text-zinc-300">
              Your name
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your name"
              maxLength={100}
              autoFocus
              className="h-12 rounded-lg border border-zinc-700 bg-zinc-900 px-4 text-base text-zinc-50 placeholder-zinc-500 outline-none transition-colors focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
            {error && <p className="text-sm text-red-400">{error}</p>}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="h-12 rounded-lg bg-blue-600 text-base font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Signing in..." : "Start Annotating"}
          </button>
        </form>

        {/* Brief description */}
        <div className="flex flex-col gap-2 rounded-lg border border-zinc-800 p-4 text-sm text-zinc-500">
          <p>
            Paint over colored regions to correct AI-generated segmentation
            masks. Each color represents one of 22 body parts used to rig
            animated characters.
          </p>
          <p>No account needed — just enter your name to start.</p>
        </div>
      </main>

      {showOnboarding && (
        <OnboardingGuide onDismiss={handleOnboardingDismiss} />
      )}
    </div>
  );
}
