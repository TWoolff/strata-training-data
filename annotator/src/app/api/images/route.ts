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
    const { searchParams } = new URL(request.url);
    const status = searchParams.get("status") || "pending";

    await ensureSchema();

    const result = await execute(
      `SELECT id, dataset, example_id, image_url, seg_url, width, height
       FROM images
       WHERE status = :status
       ORDER BY priority DESC, RANDOM()
       LIMIT 1`,
      { status },
    );

    if (result.rows.length === 0) {
      return Response.json({ image: null });
    }

    const row = result.rows[0];
    return Response.json({
      image: {
        id: row.id,
        dataset: row.dataset,
        example_id: row.example_id,
        image_url: row.image_url,
        seg_url: row.seg_url,
        width: row.width,
        height: row.height,
      },
    });
  } catch (error) {
    console.error("GET /api/images error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
