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
         u.id,
         u.name,
         COUNT(a.id) as annotated,
         SUM(CASE WHEN r.approved = 1 THEN 1 ELSE 0 END) as approved,
         CASE WHEN COUNT(a.id) > 0
           THEN CAST(SUM(COALESCE(a.time_spent, 0)) AS REAL) / COUNT(a.id)
           ELSE 0
         END as avg_time
       FROM users u
       LEFT JOIN annotations a ON a.user_id = u.id
       LEFT JOIN reviews r ON r.annotation_id = a.id
       GROUP BY u.id
       HAVING COUNT(a.id) > 0
       ORDER BY annotated DESC`,
    );

    const entries = result.rows.map((row, i) => ({
      rank: i + 1,
      id: Number(row.id),
      name: String(row.name),
      annotated: Number(row.annotated),
      approved: Number(row.approved),
      avgTime: Math.round(Number(row.avg_time)),
    }));

    return Response.json({ entries });
  } catch (error) {
    console.error("GET /api/leaderboard error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
