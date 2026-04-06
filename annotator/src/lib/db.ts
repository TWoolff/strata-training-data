import { createClient, type Client } from "@libsql/client";

let client: Client | null = null;

export function getClient(): Client {
  if (!client) {
    const url = process.env.TURSO_DATABASE_URL;
    const authToken = process.env.TURSO_AUTH_TOKEN;

    if (!url) {
      throw new Error("TURSO_DATABASE_URL is not set");
    }

    client = createClient({ url, authToken });
  }
  return client;
}

export async function execute(sql: string, args?: Record<string, string | number | null>) {
  return getClient().execute(args ? { sql, args } : sql);
}
