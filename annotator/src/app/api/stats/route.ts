import { execute } from "@/lib/db";
import { initSchema } from "@/lib/schema";

let schemaReady = false;

async function ensureSchema() {
  if (!schemaReady) {
    await initSchema();
    schemaReady = true;
  }
}

export async function GET(request: Request) {
  try {
    await ensureSchema();

    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("user_id");

    const result = await execute(
      `SELECT
         COUNT(*) as total,
         SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
       FROM images`,
    );

    const row = result.rows[0];
    const response: Record<string, unknown> = {
      total: Number(row.total),
      pending: Number(row.pending),
    };

    // Personal stats if user_id provided
    if (userId) {
      const uid = Number(userId);

      const personalResult = await execute(
        `SELECT
           COUNT(a.id) as annotated,
           SUM(CASE WHEN r.approved = 1 THEN 1 ELSE 0 END) as approved,
           SUM(CASE WHEN r.approved = 0 THEN 1 ELSE 0 END) as rejected
         FROM annotations a
         LEFT JOIN reviews r ON r.annotation_id = a.id
         WHERE a.user_id = :uid`,
        { uid },
      );

      const todayResult = await execute(
        `SELECT COUNT(*) as count FROM annotations
         WHERE user_id = :uid AND date(created_at) = date('now')`,
        { uid },
      );

      const streakResult = await execute(
        `SELECT current_streak, longest_streak FROM user_streaks WHERE user_id = :uid`,
        { uid },
      );

      const p = personalResult.rows[0];
      const annotated = Number(p.annotated);
      const approved = Number(p.approved);
      const rejected = Number(p.rejected);
      const reviewed = approved + rejected;

      response.personal = {
        annotated,
        approved,
        rejected,
        pending_review: annotated - reviewed,
        approval_rate: reviewed > 0 ? Math.round((approved / reviewed) * 100) : null,
        today: Number(todayResult.rows[0].count),
        streak: streakResult.rows.length > 0 ? Number(streakResult.rows[0].current_streak) : 0,
        longest_streak: streakResult.rows.length > 0 ? Number(streakResult.rows[0].longest_streak) : 0,
      };
    }

    return Response.json(response);
  } catch (error) {
    console.error("GET /api/stats error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
