"""Tests for the .moc3 binary parser module."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from pipeline.moc3_parser import (
    _ID_FIELD_SIZE,
    _MAGIC,
    _SOT_COUNT,
    parse_moc3,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic .moc3 binaries
# ---------------------------------------------------------------------------

# Minimum SOT index requirements for parsing:
#   SOT[0]  = CIT offset
#   SOT[3]  = Part IDs offset
#   SOT[33] = ArtMesh IDs offset
#   SOT[34] = parentPartIndex offset
#   SOT[35] = uvBeginIndex offset
#   SOT[36] = vertexCount offset
#   SOT[40] = drawOrder offset
#   SOT[41] = textureNo offset
#   SOT[45] = posIndexBegin offset
#   SOT[46] = posIndexCount offset
#   SOT[78] = UV float32 pairs offset
#   SOT[79] = triangle uint16 indices offset


def _make_id_field(name: str) -> bytes:
    """Create a 64-byte null-padded ID field."""
    encoded = name.encode("utf-8")[:_ID_FIELD_SIZE]
    return encoded + b"\x00" * (_ID_FIELD_SIZE - len(encoded))


def _build_synthetic_moc3(
    *,
    version: int = 3,
    parts: list[str] | None = None,
    meshes: list[dict] | None = None,
) -> bytes:
    """Build a minimal synthetic .moc3 binary for testing.

    Args:
        version: .moc3 version byte (1–4).
        parts: List of Part ID strings.
        meshes: List of dicts with keys: id, parent_part_idx, vertex_count,
                uvs (list of (u,v) floats), triangles (list of int),
                draw_order, texture_no.

    Returns:
        Bytes representing the synthetic .moc3 file.
    """
    if parts is None:
        parts = ["PartBody"]
    if meshes is None:
        meshes = [
            {
                "id": "ArtMesh1",
                "parent_part_idx": 0,
                "vertex_count": 4,
                "uvs": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                "triangles": [0, 1, 2, 0, 2, 3],
                "draw_order": 100,
                "texture_no": 0,
            }
        ]

    parts_count = len(parts)
    artmesh_count = len(meshes)

    # We'll build the file in sections, tracking offsets as we go.
    # Layout: header (64) + SOT (160*4=640) + CIT + PartIDs + ArtMeshIDs
    #         + per-mesh arrays + UV array + triangle array

    # --- Header (64 bytes) ---
    header = bytearray(64)
    header[0:4] = _MAGIC
    header[4] = version

    # --- SOT placeholder (160 × uint32) ---
    sot_data = bytearray(_SOT_COUNT * 4)

    # Current write position
    pos = 64 + len(sot_data)

    # --- CIT (20 × int32) ---
    cit_offset = pos
    cit = [0] * 20
    cit[0] = parts_count
    cit[4] = artmesh_count
    cit_bytes = struct.pack(f"<{len(cit)}i", *cit)
    pos += len(cit_bytes)

    # --- Part IDs ---
    part_ids_offset = pos
    part_id_bytes = b"".join(_make_id_field(p) for p in parts)
    pos += len(part_id_bytes)

    # --- ArtMesh IDs ---
    artmesh_ids_offset = pos
    artmesh_id_bytes = b"".join(_make_id_field(m["id"]) for m in meshes)
    pos += len(artmesh_id_bytes)

    # --- Per-mesh int32 arrays ---
    parent_part_idx_offset = pos
    parent_bytes = struct.pack("<" + "i" * artmesh_count, *[m["parent_part_idx"] for m in meshes])
    pos += len(parent_bytes)

    uv_begin_idx_offset = pos
    # Compute cumulative UV begins
    uv_begins = []
    cumulative = 0
    for m in meshes:
        uv_begins.append(cumulative)
        cumulative += m["vertex_count"]
    uv_begin_bytes = struct.pack("<" + "i" * artmesh_count, *uv_begins)
    pos += len(uv_begin_bytes)

    vertex_count_offset = pos
    vc_bytes = struct.pack("<" + "i" * artmesh_count, *[m["vertex_count"] for m in meshes])
    pos += len(vc_bytes)

    draw_order_offset = pos
    do_bytes = struct.pack("<" + "i" * artmesh_count, *[m["draw_order"] for m in meshes])
    pos += len(do_bytes)

    texture_no_offset = pos
    tn_bytes = struct.pack("<" + "i" * artmesh_count, *[m["texture_no"] for m in meshes])
    pos += len(tn_bytes)

    # Compute cumulative triangle index begins
    pos_index_begin_offset = pos
    tri_begins = []
    tri_cumulative = 0
    for m in meshes:
        tri_begins.append(tri_cumulative)
        tri_cumulative += len(m["triangles"])
    pib_bytes = struct.pack("<" + "i" * artmesh_count, *tri_begins)
    pos += len(pib_bytes)

    pos_index_count_offset = pos
    pic_bytes = struct.pack("<" + "i" * artmesh_count, *[len(m["triangles"]) for m in meshes])
    pos += len(pic_bytes)

    # --- Global UV array (float32 pairs) ---
    uv_array_offset = pos
    all_uvs: list[float] = []
    for m in meshes:
        for u, v in m["uvs"]:
            all_uvs.extend([u, v])
    uv_array_bytes = struct.pack(f"<{len(all_uvs)}f", *all_uvs)
    pos += len(uv_array_bytes)

    # --- Global triangle index array (uint16) ---
    tri_array_offset = pos
    all_tris: list[int] = []
    for m in meshes:
        all_tris.extend(m["triangles"])
    tri_array_bytes = struct.pack(f"<{len(all_tris)}H", *all_tris)
    pos += len(tri_array_bytes)

    # --- Fill SOT ---
    def _set_sot(idx: int, val: int) -> None:
        struct.pack_into("<I", sot_data, idx * 4, val)

    _set_sot(0, cit_offset)
    _set_sot(3, part_ids_offset)
    _set_sot(33, artmesh_ids_offset)
    _set_sot(34, parent_part_idx_offset)
    _set_sot(35, uv_begin_idx_offset)
    _set_sot(36, vertex_count_offset)
    _set_sot(40, draw_order_offset)
    _set_sot(41, texture_no_offset)
    _set_sot(45, pos_index_begin_offset)
    _set_sot(46, pos_index_count_offset)
    _set_sot(78, uv_array_offset)
    _set_sot(79, tri_array_offset)

    # --- Assemble ---
    result = bytes(header) + bytes(sot_data)
    result += cit_bytes + part_id_bytes + artmesh_id_bytes
    result += parent_bytes + uv_begin_bytes + vc_bytes
    result += do_bytes + tn_bytes + pib_bytes + pic_bytes
    result += uv_array_bytes + tri_array_bytes

    return result


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_moc3_path(tmp_path: Path) -> Path:
    """Create a synthetic .moc3 file with one mesh."""
    data = _build_synthetic_moc3(
        version=3,
        parts=["PartBody", "PartHead"],
        meshes=[
            {
                "id": "ArtMesh_body",
                "parent_part_idx": 0,
                "vertex_count": 4,
                "uvs": [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)],
                "triangles": [0, 1, 2, 0, 2, 3],
                "draw_order": 50,
                "texture_no": 0,
            },
            {
                "id": "ArtMesh_head",
                "parent_part_idx": 1,
                "vertex_count": 3,
                "uvs": [(0.6, 0.1), (0.9, 0.1), (0.75, 0.4)],
                "triangles": [0, 1, 2],
                "draw_order": 100,
                "texture_no": 0,
            },
        ],
    )
    path = tmp_path / "test.moc3"
    path.write_bytes(data)
    return path


@pytest.fixture()
def multi_mesh_moc3_path(tmp_path: Path) -> Path:
    """Create a synthetic .moc3 with multiple meshes including edge cases."""
    data = _build_synthetic_moc3(
        version=4,
        parts=["PartBody", "PartHead", "PartEffect"],
        meshes=[
            {
                "id": "ArtMesh_body",
                "parent_part_idx": 0,
                "vertex_count": 4,
                "uvs": [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
                "triangles": [0, 1, 2, 0, 2, 3],
                "draw_order": 10,
                "texture_no": 0,
            },
            {
                "id": "ArtMesh_head",
                "parent_part_idx": 1,
                "vertex_count": 3,
                "uvs": [(0.5, 0.0), (1.0, 0.0), (0.75, 0.5)],
                "triangles": [0, 1, 2],
                "draw_order": 20,
                "texture_no": 0,
            },
            {
                "id": "ArtMesh_effect",
                "parent_part_idx": 2,
                "vertex_count": 3,
                "uvs": [(0.0, 0.5), (0.5, 0.5), (0.25, 1.0)],
                "triangles": [0, 1, 2],
                "draw_order": 5,
                "texture_no": 0,
            },
        ],
    )
    path = tmp_path / "multi.moc3"
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Tests: Magic and header validation
# ---------------------------------------------------------------------------


class TestParseMagicValidation:
    """Test that the parser rejects non-MOC3 files."""

    def test_reject_non_moc3_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.moc3"
        bad_file.write_bytes(b"NOT_MOC3" + b"\x00" * 1000)
        result = parse_moc3(bad_file)
        assert result is None

    def test_reject_too_small_file(self, tmp_path: Path) -> None:
        tiny_file = tmp_path / "tiny.moc3"
        tiny_file.write_bytes(b"MOC3\x03")
        result = parse_moc3(tiny_file)
        assert result is None

    def test_reject_nonexistent_file(self, tmp_path: Path) -> None:
        result = parse_moc3(tmp_path / "nonexistent.moc3")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Version extraction
# ---------------------------------------------------------------------------


class TestParseVersionExtraction:
    """Test that version bytes 1–4 are correctly extracted."""

    @pytest.mark.parametrize("version", [1, 2, 3, 4])
    def test_version_extraction(self, tmp_path: Path, version: int) -> None:
        data = _build_synthetic_moc3(version=version)
        path = tmp_path / f"v{version}.moc3"
        path.write_bytes(data)

        model = parse_moc3(path)
        assert model is not None
        assert model.version == version


# ---------------------------------------------------------------------------
# Tests: Basic parsing
# ---------------------------------------------------------------------------


class TestParseBasic:
    """Test basic parsing of synthetic .moc3 files."""

    def test_parse_synthetic(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        assert model.version == 3
        assert model.parts_count == 2
        assert model.artmesh_count == 2
        assert len(model.meshes) == 2

    def test_mesh_ids(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        ids = [m.mesh_id for m in model.meshes]
        assert "ArtMesh_body" in ids
        assert "ArtMesh_head" in ids

    def test_parent_part_ids(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        body_mesh = next(m for m in model.meshes if m.mesh_id == "ArtMesh_body")
        head_mesh = next(m for m in model.meshes if m.mesh_id == "ArtMesh_head")
        assert body_mesh.parent_part_id == "PartBody"
        assert head_mesh.parent_part_id == "PartHead"

    def test_draw_orders(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        body_mesh = next(m for m in model.meshes if m.mesh_id == "ArtMesh_body")
        head_mesh = next(m for m in model.meshes if m.mesh_id == "ArtMesh_head")
        assert body_mesh.draw_order == 50
        assert head_mesh.draw_order == 100

    def test_texture_numbers(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        for mesh in model.meshes:
            assert mesh.texture_no == 0

    def test_part_ids_list(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        assert model.part_ids == ["PartBody", "PartHead"]


# ---------------------------------------------------------------------------
# Tests: UV coordinates
# ---------------------------------------------------------------------------


class TestUVCoordinates:
    """Test that UV coordinates are correctly extracted and in [0, 1]."""

    def test_uv_count_matches_vertex_count(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        for mesh in model.meshes:
            assert len(mesh.uvs) == mesh.vertex_count

    def test_uv_range(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        for mesh in model.meshes:
            for u, v in mesh.uvs:
                assert 0.0 <= u <= 1.0, f"UV u={u} out of range in {mesh.mesh_id}"
                assert 0.0 <= v <= 1.0, f"UV v={v} out of range in {mesh.mesh_id}"

    def test_specific_uv_values(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        body = next(m for m in model.meshes if m.mesh_id == "ArtMesh_body")
        assert len(body.uvs) == 4
        assert body.uvs[0] == pytest.approx((0.1, 0.1), abs=1e-5)
        assert body.uvs[1] == pytest.approx((0.5, 0.1), abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: Triangle index localization
# ---------------------------------------------------------------------------


class TestTriangleIndexLocalization:
    """Test that triangle indices are correctly localized to vertex count."""

    def test_indices_within_vertex_count(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        for mesh in model.meshes:
            for idx in mesh.triangle_indices:
                assert 0 <= idx < mesh.vertex_count, (
                    f"Index {idx} >= vertex_count {mesh.vertex_count} in {mesh.mesh_id}"
                )

    def test_triangle_count_divisible_by_three(self, synthetic_moc3_path: Path) -> None:
        model = parse_moc3(synthetic_moc3_path)
        assert model is not None
        for mesh in model.meshes:
            assert len(mesh.triangle_indices) % 3 == 0

    def test_index_localization_with_overflow(self, tmp_path: Path) -> None:
        """Verify that indices exceeding vertex_count are localized via modulo."""
        data = _build_synthetic_moc3(
            meshes=[
                {
                    "id": "ArtMesh_overflow",
                    "parent_part_idx": 0,
                    "vertex_count": 3,
                    "uvs": [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
                    # Indices 3, 4, 5 should become 0, 1, 2 via modulo
                    "triangles": [3, 4, 5],
                    "draw_order": 10,
                    "texture_no": 0,
                }
            ],
        )
        path = tmp_path / "overflow.moc3"
        path.write_bytes(data)

        model = parse_moc3(path)
        assert model is not None
        assert len(model.meshes) == 1
        mesh = model.meshes[0]
        assert mesh.triangle_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Tests: Zero-mesh / empty models
# ---------------------------------------------------------------------------


class TestParseZeroMesh:
    """Test handling of models with no ArtMeshes."""

    def test_zero_artmesh_count(self, tmp_path: Path) -> None:
        """A .moc3 with artmesh_count=0 should return a model with empty meshes."""
        data = _build_synthetic_moc3(meshes=[])
        # Patch artmesh count to 0 in the CIT
        path = tmp_path / "empty.moc3"
        path.write_bytes(data)

        model = parse_moc3(path)
        assert model is not None
        assert model.artmesh_count == 0
        assert model.meshes == []


# ---------------------------------------------------------------------------
# Tests: Multiple meshes
# ---------------------------------------------------------------------------


class TestMultipleMeshes:
    """Test parsing files with multiple ArtMeshes."""

    def test_mesh_count(self, multi_mesh_moc3_path: Path) -> None:
        model = parse_moc3(multi_mesh_moc3_path)
        assert model is not None
        assert len(model.meshes) == 3

    def test_mesh_ordering(self, multi_mesh_moc3_path: Path) -> None:
        model = parse_moc3(multi_mesh_moc3_path)
        assert model is not None
        ids = [m.mesh_id for m in model.meshes]
        assert ids == ["ArtMesh_body", "ArtMesh_head", "ArtMesh_effect"]

    def test_draw_order_values(self, multi_mesh_moc3_path: Path) -> None:
        model = parse_moc3(multi_mesh_moc3_path)
        assert model is not None
        orders = {m.mesh_id: m.draw_order for m in model.meshes}
        assert orders["ArtMesh_body"] == 10
        assert orders["ArtMesh_head"] == 20
        assert orders["ArtMesh_effect"] == 5


# ---------------------------------------------------------------------------
# Integration tests (require real data — skip if not available)
# ---------------------------------------------------------------------------

_REAL_MOC3 = Path("data/live2d/001/xinnong_5.moc3")


@pytest.mark.skipif(not _REAL_MOC3.exists(), reason="Real .moc3 test data not available")
class TestIntegrationRealFile:
    """Integration tests against a real .moc3 file."""

    def test_parse_real_file(self) -> None:
        model = parse_moc3(_REAL_MOC3)
        assert model is not None
        assert model.version >= 1
        assert model.artmesh_count > 0
        assert len(model.meshes) > 0

    def test_real_uv_ranges(self) -> None:
        model = parse_moc3(_REAL_MOC3)
        assert model is not None
        for mesh in model.meshes:
            for u, v in mesh.uvs:
                assert 0.0 <= u <= 1.0
                assert 0.0 <= v <= 1.0

    def test_real_triangle_indices_valid(self) -> None:
        model = parse_moc3(_REAL_MOC3)
        assert model is not None
        for mesh in model.meshes:
            for idx in mesh.triangle_indices:
                assert 0 <= idx < mesh.vertex_count

    def test_real_artmesh_count(self) -> None:
        model = parse_moc3(_REAL_MOC3)
        assert model is not None
        # xinnong_5 has 1056 artmeshes
        assert model.artmesh_count == 1056

    def test_real_parts_count(self) -> None:
        model = parse_moc3(_REAL_MOC3)
        assert model is not None
        assert model.parts_count == 117
