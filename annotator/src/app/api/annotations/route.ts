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

    const countResult = await execute(
      `SELECT COUNT(*) as count FROM annotations WHERE user_id = :user_id`,
      { user_id: Number(user_id) },
    );

    return Response.json({
      success: true,
      count: Number(countResult.rows[0].count),
    });
  } catch (error) {
    console.error("POST /api/annotations error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
