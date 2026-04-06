export type Region = {
  id: number;
  name: string;
  color: [number, number, number];
  shortcut: string;
};

/** 22-class anatomy regions matching pipeline/config.py REGION_NAMES + REGION_COLORS. */
export const REGIONS: Region[] = [
  { id: 0, name: "background", color: [0, 0, 0], shortcut: "0" },
  { id: 1, name: "head", color: [255, 0, 0], shortcut: "1" },
  { id: 2, name: "neck", color: [0, 255, 0], shortcut: "2" },
  { id: 3, name: "chest", color: [0, 0, 255], shortcut: "3" },
  { id: 4, name: "spine", color: [255, 255, 0], shortcut: "4" },
  { id: 5, name: "hips", color: [255, 0, 255], shortcut: "5" },
  { id: 6, name: "shoulder_l", color: [192, 64, 0], shortcut: "6" },
  { id: 7, name: "upper_arm_l", color: [128, 0, 0], shortcut: "7" },
  { id: 8, name: "forearm_l", color: [0, 128, 0], shortcut: "8" },
  { id: 9, name: "hand_l", color: [0, 0, 128], shortcut: "9" },
  { id: 10, name: "shoulder_r", color: [0, 64, 192], shortcut: "q" },
  { id: 11, name: "upper_arm_r", color: [128, 128, 0], shortcut: "w" },
  { id: 12, name: "forearm_r", color: [128, 0, 128], shortcut: "e" },
  { id: 13, name: "hand_r", color: [0, 128, 128], shortcut: "r" },
  { id: 14, name: "upper_leg_l", color: [64, 0, 0], shortcut: "t" },
  { id: 15, name: "lower_leg_l", color: [0, 64, 0], shortcut: "y" },
  { id: 16, name: "foot_l", color: [0, 0, 64], shortcut: "u" },
  { id: 17, name: "upper_leg_r", color: [64, 64, 0], shortcut: "i" },
  { id: 18, name: "lower_leg_r", color: [64, 0, 64], shortcut: "o" },
  { id: 19, name: "foot_r", color: [0, 64, 64], shortcut: "p" },
  { id: 20, name: "accessory", color: [128, 128, 128], shortcut: "a" },
  { id: 21, name: "hair_back", color: [200, 200, 50], shortcut: "s" },
];

export const NUM_REGIONS = 22;

/** Look up a region by keyboard shortcut key. */
export function regionByShortcut(key: string): Region | undefined {
  return REGIONS.find((r) => r.shortcut === key);
}

/** Convert region color to CSS rgb string. */
export function regionCss(region: Region): string {
  const [r, g, b] = region.color;
  return `rgb(${r}, ${g}, ${b})`;
}
