"use client";

import { REGIONS, regionCss, type Region } from "@/lib/regions";

type Props = {
  activeRegion: Region;
  onRegionChange: (region: Region) => void;
};

const GROUPS: { label: string; ids: number[] }[] = [
  { label: "Head", ids: [1, 2] },
  { label: "Torso", ids: [3, 4, 5] },
  { label: "Left Arm", ids: [6, 7, 8, 9] },
  { label: "Right Arm", ids: [10, 11, 12, 13] },
  { label: "Left Leg", ids: [14, 15, 16] },
  { label: "Right Leg", ids: [17, 18, 19] },
  { label: "Other", ids: [20, 21] },
];

export function RegionPalette({ activeRegion, onRegionChange }: Props) {
  return (
    <div className="flex w-48 flex-col gap-3 overflow-y-auto border-l border-zinc-800 bg-zinc-900 p-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Regions
      </h3>
      {GROUPS.map((group) => (
        <div key={group.label}>
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
            {group.label}
          </div>
          <div className="flex flex-col gap-0.5">
            {group.ids.map((id) => {
              const region = REGIONS[id];
              const isActive = activeRegion.id === id;
              return (
                <button
                  key={id}
                  onClick={() => onRegionChange(region)}
                  className={`flex items-center gap-2 rounded px-2 py-1 text-left text-xs transition-colors ${
                    isActive
                      ? "bg-zinc-700 text-zinc-100"
                      : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                  }`}
                >
                  <span
                    className="inline-block h-3 w-3 shrink-0 rounded-sm border border-zinc-600"
                    style={{ backgroundColor: regionCss(region) }}
                  />
                  <span className="flex-1 truncate">{region.name}</span>
                  <kbd className="rounded bg-zinc-800 px-1 font-mono text-[10px] text-zinc-500">
                    {region.shortcut}
                  </kbd>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
