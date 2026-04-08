import { getClient } from "./db";

const SCHEMA_SQL = [
  `CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT (datetime('now'))
  )`,
  `CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    example_id TEXT NOT NULL UNIQUE,
    image_url TEXT NOT NULL,
    seg_url TEXT NOT NULL,
    width INTEGER DEFAULT 512,
    height INTEGER DEFAULT 512,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now'))
  )`,
  `CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    mask_data TEXT NOT NULL,
    time_spent INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
  )`,
  `CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id),
    approved INTEGER NOT NULL,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
  )`,
  `CREATE TABLE IF NOT EXISTS user_streaks (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    current_streak INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    last_active_date TEXT
  )`,
];

/** Create all tables if they don't exist. */
export async function initSchema(): Promise<void> {
  const client = getClient();
  for (const sql of SCHEMA_SQL) {
    await client.execute(sql);
  }
}
