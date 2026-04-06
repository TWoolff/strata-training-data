import { execute } from "@/lib/db";
import { initSchema } from "@/lib/schema";

let schemaReady = false;

async function ensureSchema() {
  if (!schemaReady) {
    await initSchema();
    schemaReady = true;
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const name = typeof body.name === "string" ? body.name.trim() : "";

    if (!name || name.length > 100) {
      return Response.json(
        { error: "Name is required (max 100 characters)" },
        { status: 400 },
      );
    }

    await ensureSchema();

    // Upsert: insert if new, return existing if name already taken
    await execute(
      "INSERT OR IGNORE INTO users (name) VALUES (:name)",
      { name },
    );

    const result = await execute(
      "SELECT id, name, created_at FROM users WHERE name = :name",
      { name },
    );

    const user = result.rows[0];
    return Response.json({ id: user.id, name: user.name });
  } catch (error) {
    console.error("POST /api/users error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const name = searchParams.get("name")?.trim();

    if (!name) {
      return Response.json({ error: "name query parameter required" }, { status: 400 });
    }

    await ensureSchema();

    const result = await execute(
      "SELECT id, name, created_at FROM users WHERE name = :name",
      { name },
    );

    if (result.rows.length === 0) {
      return Response.json({ error: "User not found" }, { status: 404 });
    }

    const user = result.rows[0];
    return Response.json({ id: user.id, name: user.name });
  } catch (error) {
    console.error("GET /api/users error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
