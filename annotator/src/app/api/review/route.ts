import { execute } from "@/lib/db";
import { initSchema } from "@/lib/schema";

let schemaReady = false;

async function ensureSchema() {
  if (!schemaReady) {
    await initSchema();
    schemaReady = true;
  }
}

const REVIEW_SECRET = process.env.REVIEW_SECRET || "";

function checkAuth(request: Request): boolean {
  if (!REVIEW_SECRET) return true; // no secret configured = open
  const { searchParams } = new URL(request.url);
  const key = searchParams.get("key");
  return key === REVIEW_SECRET;
}

/**
 * GET /api/review — list annotations ready for review.
 * Query params:
 *   key — review secret for auth
 *   dataset — filter by dataset name
 *   annotator — filter by annotator user id
 *   include_mask — if "1", include mask_data (for detail view)
 *   annotation_id — fetch a single annotation by id (for detail view)
 */
export async function GET(request: Request) {
  try {
    if (!checkAuth(request)) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    await ensureSchema();

    const { searchParams } = new URL(request.url);
    const dataset = searchParams.get("dataset");
    const annotator = searchParams.get("annotator");
    const includeMask = searchParams.get("include_mask") === "1";
    const annotationId = searchParams.get("annotation_id");

    // Single annotation detail view
    if (annotationId) {
      const result = await execute(
        `SELECT
           a.id as annotation_id,
           a.image_id,
           a.user_id,
           a.mask_data,
           a.time_spent,
           a.created_at as annotated_at,
           i.dataset,
           i.example_id,
           i.image_url,
           i.seg_url,
           i.width,
           i.height,
           u.name as annotator_name
         FROM annotations a
         JOIN images i ON i.id = a.image_id
         JOIN users u ON u.id = a.user_id
         WHERE a.id = :annotation_id`,
        { annotation_id: Number(annotationId) },
      );

      if (result.rows.length === 0) {
        return Response.json({ error: "Annotation not found" }, { status: 404 });
      }

      const row = result.rows[0];
      return Response.json({
        annotation: {
          annotation_id: Number(row.annotation_id),
          image_id: Number(row.image_id),
          user_id: Number(row.user_id),
          mask_data: String(row.mask_data),
          time_spent: row.time_spent ? Number(row.time_spent) : null,
          annotated_at: String(row.annotated_at),
          dataset: String(row.dataset),
          example_id: String(row.example_id),
          image_url: String(row.image_url),
          seg_url: String(row.seg_url),
          width: Number(row.width),
          height: Number(row.height),
          annotator_name: String(row.annotator_name),
        },
      });
    }

    // List view: annotations without a review yet
    let sql = `
      SELECT
        a.id as annotation_id,
        a.image_id,
        a.user_id,
        a.time_spent,
        a.created_at as annotated_at,
        ${includeMask ? "a.mask_data," : ""}
        i.dataset,
        i.example_id,
        i.image_url,
        i.seg_url,
        i.width,
        i.height,
        u.name as annotator_name
      FROM annotations a
      JOIN images i ON i.id = a.image_id
      JOIN users u ON u.id = a.user_id
      LEFT JOIN reviews r ON r.annotation_id = a.id
      WHERE r.id IS NULL
    `;

    const args: Record<string, string | number | null> = {};

    if (dataset) {
      sql += ` AND i.dataset = :dataset`;
      args.dataset = dataset;
    }
    if (annotator) {
      sql += ` AND a.user_id = :annotator`;
      args.annotator = Number(annotator);
    }

    sql += ` ORDER BY a.created_at DESC`;

    const result = await execute(sql, Object.keys(args).length > 0 ? args : undefined);

    const annotations = result.rows.map((row) => {
      const item: Record<string, unknown> = {
        annotation_id: Number(row.annotation_id),
        image_id: Number(row.image_id),
        user_id: Number(row.user_id),
        time_spent: row.time_spent ? Number(row.time_spent) : null,
        annotated_at: String(row.annotated_at),
        dataset: String(row.dataset),
        example_id: String(row.example_id),
        image_url: String(row.image_url),
        seg_url: String(row.seg_url),
        width: Number(row.width),
        height: Number(row.height),
        annotator_name: String(row.annotator_name),
      };
      if (includeMask) {
        item.mask_data = String(row.mask_data);
      }
      return item;
    });

    // Also fetch filter options
    const datasetsResult = await execute(
      `SELECT DISTINCT i.dataset
       FROM annotations a
       JOIN images i ON i.id = a.image_id
       LEFT JOIN reviews r ON r.annotation_id = a.id
       WHERE r.id IS NULL
       ORDER BY i.dataset`,
    );
    const annotatorsResult = await execute(
      `SELECT DISTINCT u.id, u.name
       FROM annotations a
       JOIN users u ON u.id = a.user_id
       LEFT JOIN reviews r ON r.annotation_id = a.id
       WHERE r.id IS NULL
       ORDER BY u.name`,
    );

    return Response.json({
      annotations,
      filters: {
        datasets: datasetsResult.rows.map((r) => String(r.dataset)),
        annotators: annotatorsResult.rows.map((r) => ({
          id: Number(r.id),
          name: String(r.name),
        })),
      },
    });
  } catch (error) {
    console.error("GET /api/review error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}

/**
 * POST /api/review — save a review decision.
 * Body: { annotation_id, approved (boolean), notes? (string) }
 * Or batch: { annotation_ids: number[], approved (boolean), notes? }
 */
export async function POST(request: Request) {
  try {
    if (!checkAuth(request)) {
      return Response.json({ error: "Unauthorized" }, { status: 401 });
    }

    await ensureSchema();

    const body = await request.json();
    const { annotation_id, annotation_ids, approved, notes } = body;

    if (approved === undefined || approved === null) {
      return Response.json(
        { error: "approved is required" },
        { status: 400 },
      );
    }

    const ids: number[] = annotation_ids
      ? (annotation_ids as number[])
      : annotation_id
        ? [Number(annotation_id)]
        : [];

    if (ids.length === 0) {
      return Response.json(
        { error: "annotation_id or annotation_ids is required" },
        { status: 400 },
      );
    }

    const approvedInt = approved ? 1 : 0;
    let reviewed = 0;

    for (const id of ids) {
      // Check if already reviewed
      const existing = await execute(
        `SELECT id FROM reviews WHERE annotation_id = :annotation_id`,
        { annotation_id: id },
      );
      if (existing.rows.length > 0) continue;

      await execute(
        `INSERT INTO reviews (annotation_id, approved, notes)
         VALUES (:annotation_id, :approved, :notes)`,
        {
          annotation_id: id,
          approved: approvedInt,
          notes: notes ? String(notes) : null,
        },
      );

      // Update image status
      const status = approved ? "approved" : "rejected";
      const annResult = await execute(
        `SELECT image_id FROM annotations WHERE id = :id`,
        { id },
      );
      if (annResult.rows.length > 0) {
        await execute(
          `UPDATE images SET status = :status WHERE id = :image_id`,
          { status, image_id: Number(annResult.rows[0].image_id) },
        );
      }

      reviewed++;
    }

    return Response.json({ success: true, reviewed });
  } catch (error) {
    console.error("POST /api/review error:", error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
