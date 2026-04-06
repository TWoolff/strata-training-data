import { execute } from "@/lib/db";
import { initSchema } from "@/lib/schema";

let schemaReady = false;

async function ensureSchema() {
  if (!schemaReady) {
    await initSchema();
    schemaReady = true;
  }
}

export async function GET() {
  try {
    await ensureSchema();

    const result = await execute(
      `SELECT
         COUNT(*) as total,
         SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
       FROM images`,
    );

    const row = result.rows[0];

    return Response.json({
      total: Number(row.total),
      pending: Number(row.pending),
    });
  } catch (error) {
    console.error("GET /api/stats error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
