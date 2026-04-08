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
    const { image_id, user_id, mask_data, time_spent } = body;

    if (!image_id || !user_id || !mask_data) {
      return Response.json(
        { error: "image_id, user_id, and mask_data are required" },
        { status: 400 },
      );
    }

    await ensureSchema();

    await execute(
      `INSERT INTO annotations (image_id, user_id, mask_data, time_spent)
       VALUES (:image_id, :user_id, :mask_data, :time_spent)`,
      {
        image_id: Number(image_id),
        user_id: Number(user_id),
        mask_data: String(mask_data),
        time_spent: time_spent ? Number(time_spent) : null,
      },
    );

    await execute(
      `UPDATE images SET status = 'annotated' WHERE id = :image_id`,
      { image_id: Number(image_id) },
    );

    // Update daily streak
    const uid = Number(user_id);
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    const streakRow = await execute(
      `SELECT current_streak, longest_streak, last_active_date FROM user_streaks WHERE user_id = :uid`,
      { uid },
    );

    let currentStreak = 1;
    let longestStreak = 1;

    if (streakRow.rows.length === 0) {
      await execute(
        `INSERT INTO user_streaks (user_id, current_streak, longest_streak, last_active_date)
         VALUES (:uid, 1, 1, :today)`,
        { uid, today },
      );
    } else {
      const row = streakRow.rows[0];
      const lastDate = String(row.last_active_date);
      const prevStreak = Number(row.current_streak);
      longestStreak = Number(row.longest_streak);

      if (lastDate === today) {
        // Same day — no streak change
        currentStreak = prevStreak;
      } else {
        // Check if yesterday
        const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
        currentStreak = lastDate === yesterday ? prevStreak + 1 : 1;
        longestStreak = Math.max(longestStreak, currentStreak);

        await execute(
          `UPDATE user_streaks
           SET current_streak = :currentStreak, longest_streak = :longestStreak, last_active_date = :today
           WHERE user_id = :uid`,
          { currentStreak, longestStreak, today, uid },
        );
      }
    }

    const countResult = await execute(
      `SELECT COUNT(*) as count FROM annotations WHERE user_id = :user_id`,
      { user_id: uid },
    );

    const todayCount = await execute(
      `SELECT COUNT(*) as count FROM annotations
       WHERE user_id = :uid AND date(created_at) = date('now')`,
      { uid },
    );

    return Response.json({
      success: true,
      count: Number(countResult.rows[0].count),
      today: Number(todayCount.rows[0].count),
      streak: currentStreak,
      longestStreak,
    });
  } catch (error) {
    console.error("POST /api/annotations error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
