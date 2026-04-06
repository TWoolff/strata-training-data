"use client";

import { REGIONS, regionCss } from "@/lib/regions";

/**
 * SVG body silhouette with 22 colored anatomy regions.
 * Each region is a simplified polygon matching the Strata skeleton schema.
 */
export function BodyMap() {
  // Simplified front-facing body region paths (viewBox: 0 0 200 400)
  const regionPaths: Record<string, string> = {
    head: "M88,10 Q100,0 112,10 Q124,20 120,40 L118,55 Q110,65 100,65 Q90,65 82,55 L80,40 Q76,20 88,10Z",
    neck: "M90,65 L110,65 L108,80 L92,80Z",
    chest: "M70,80 L130,80 L132,130 L68,130Z",
    spine: "M78,130 L122,130 L120,160 L80,160Z",
    hips: "M78,160 L122,160 L124,185 L76,185Z",
    shoulder_l: "M130,80 L150,85 L148,100 L132,95Z",
    upper_arm_l: "M148,100 L155,105 L150,145 L142,140Z",
    forearm_l: "M150,145 L155,148 L152,195 L145,192Z",
    hand_l: "M145,192 L155,195 L158,220 L143,218Z",
    shoulder_r: "M70,80 L50,85 L52,100 L68,95Z",
    upper_arm_r: "M52,100 L45,105 L50,145 L58,140Z",
    forearm_r: "M50,145 L45,148 L48,195 L55,192Z",
    hand_r: "M55,192 L45,195 L42,220 L57,218Z",
    upper_leg_l: "M102,185 L124,185 L120,260 L105,260Z",
    lower_leg_l: "M105,260 L120,260 L118,340 L107,340Z",
    foot_l: "M107,340 L118,340 L125,360 L105,360Z",
    upper_leg_r: "M76,185 L98,185 L95,260 L80,260Z",
    lower_leg_r: "M80,260 L95,260 L93,340 L82,340Z",
    foot_r: "M82,340 L93,340 L95,360 L75,360Z",
    hair_back: "M82,5 Q100,-5 118,5 Q128,12 125,30 L75,30 Q72,12 82,5Z",
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <svg viewBox="0 0 200 370" className="h-80 w-auto">
        {/* Background */}
        <rect width="200" height="370" fill="transparent" />

        {/* Render each region */}
        {Object.entries(regionPaths).map(([name, path]) => {
          const region = REGIONS.find((r) => r.name === name);
          if (!region) return null;
          return (
            <path
              key={name}
              d={path}
              fill={regionCss(region)}
              fillOpacity={0.85}
              stroke="rgba(255,255,255,0.3)"
              strokeWidth={0.5}
            />
          );
        })}
      </svg>

      {/* Legend grid */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
        {REGIONS.filter((r) => r.id !== 0).map((region) => (
          <div key={region.id} className="flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-3 shrink-0 rounded-sm"
              style={{ backgroundColor: regionCss(region) }}
            />
            <span className="text-zinc-300">
              {region.name.replace(/_/g, " ")}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
