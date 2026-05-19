from __future__ import annotations

import math

import polars as pl

from danbooru_graph.etl import prepare_vocab
from danbooru_graph.scoring import score_edges
from danbooru_graph.sparse_edges import build_edges


def _write_fixture(path) -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "rating": ["g", "g", "s", "g"],
            "is_deleted": [False, False, False, False],
            "is_banned": [False, False, False, False],
            "tag_string_artist": ["", "", "", ""],
            "tag_string_character": [
                "asuna_(blue_archive) karin_(blue_archive)",
                "asuna_(blue_archive)",
                "karin_(blue_archive)",
                "hina_(blue_archive)",
            ],
            "tag_string_copyright": ["blue_archive", "blue_archive", "blue_archive", ""],
            "tag_string_general": [
                "blue_eyes smile",
                "blue_eyes",
                "dark-skinned_female smile",
                "smile",
            ],
            "tag_string_meta": ["highres", "", "", ""],
        }
    )
    df.write_parquet(path)


def test_prepare_build_score_cross_category(tmp_path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_dir = tmp_path / "processed"
    _write_fixture(raw_path)

    prepare_vocab(str(raw_path), out_dir, min_tag_count=1)
    build_edges(out_dir, "character-general", min_pair_count=2)
    score_edges(out_dir, "character-general")

    vocab = pl.read_parquet(out_dir / "tag_vocab.parquet")
    assert "" not in vocab["tag"].to_list()

    edges = pl.read_parquet(out_dir / "edges_character_general.parquet")
    row = edges.filter(
        (pl.col("tag_a") == "asuna_(blue_archive)") & (pl.col("tag_b") == "blue_eyes")
    ).row(0, named=True)

    assert row["co_count"] == 2
    assert row["count_a"] == 2
    assert row["count_b"] == 2
    assert math.isclose(row["lift"], 2.0, rel_tol=1e-9)
    assert math.isclose(row["pmi"], 1.0, rel_tol=1e-9)
    assert math.isclose(row["ppmi"], 1.0, rel_tol=1e-9)
    assert math.isclose(row["npmi"], 1.0, rel_tol=1e-9)
    assert math.isclose(row["confidence_a_to_b"], 1.0, rel_tol=1e-9)
    assert math.isclose(row["confidence_b_to_a"], 1.0, rel_tol=1e-9)
    assert math.isclose(row["discounted_ppmi"], 1.0 / 6.0, rel_tol=1e-9)


def test_same_category_edges_have_no_self_or_reverse_duplicates(tmp_path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_dir = tmp_path / "processed"
    _write_fixture(raw_path)

    prepare_vocab(str(raw_path), out_dir, min_tag_count=1)
    build_edges(out_dir, "character-character", min_pair_count=1)

    edges = pl.read_parquet(out_dir / "edges_character_character_counts.parquet")
    assert edges.filter(pl.col("tag_a_id") == pl.col("tag_b_id")).height == 0

    pairs = {(row["tag_a_id"], row["tag_b_id"]) for row in edges.iter_rows(named=True)}
    assert not any((b, a) in pairs for a, b in pairs)


def test_npmi_is_finite_for_full_dataset_binding(tmp_path) -> None:
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    pl.DataFrame({"post_idx": [0, 1], "post_id": [1, 2]}).write_parquet(out_dir / "posts.parquet")
    pl.DataFrame(
        {
            "tag_id": [0, 1],
            "category": ["character", "general"],
            "tag": ["always_character", "always_general"],
            "count": [2, 2],
        }
    ).write_parquet(out_dir / "tag_vocab.parquet")
    pl.DataFrame(
        {"tag_a_id": [0], "tag_b_id": [1], "co_count": [2]},
        schema={"tag_a_id": pl.UInt32, "tag_b_id": pl.UInt32, "co_count": pl.UInt32},
    ).write_parquet(out_dir / "edges_character_general_counts.parquet")

    score_edges(out_dir, "character-general")
    row = pl.read_parquet(out_dir / "edges_character_general.parquet").row(0, named=True)

    assert math.isfinite(row["npmi"])
    assert row["npmi"] == 1.0
