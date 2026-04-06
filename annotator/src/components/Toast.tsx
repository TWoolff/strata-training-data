"use client";

import { useEffect, useState } from "react";

type ToastItem = {
  id: number;
  message: string;
};

let nextId = 0;
let listeners: Array<() => void> = [];
let toasts: ToastItem[] = [];

function emit() {
  for (const l of listeners) l();
}

export function showToast(message: string, durationMs = 2000) {
  const id = nextId++;
  toasts = [...toasts, { id, message }];
  emit();
  setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== id);
    emit();
  }, durationMs);
}

export function ToastContainer() {
  const [items, setItems] = useState<ToastItem[]>([]);

  useEffect(() => {
    const update = () => setItems([...toasts]);
    listeners.push(update);
    return () => {
      listeners = listeners.filter((l) => l !== update);
    };
  }, []);

  if (items.length === 0) return null;

  return (
    <div className="pointer-events-none fixed bottom-20 left-1/2 z-50 flex -translate-x-1/2 flex-col items-center gap-2">
      {items.map((t) => (
        <div
          key={t.id}
          className="animate-fade-in rounded-lg bg-zinc-700 px-4 py-2 text-sm font-medium text-zinc-100 shadow-lg"
        >
          {t.message}
        </div>
      ))}
    </div>
  );
}
