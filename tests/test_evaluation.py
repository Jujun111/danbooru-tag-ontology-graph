from __future__ import annotations

import json
import math

import polars as pl
from typer.testing import CliRunner

from danbooru_graph.cli import app
from danbooru_graph.evaluation import (
    build_character_copyright_profile,
    evaluate_community_purity,
    summarize_community_general_tags,
)


def _write_communities(path) -> None:
    communities = [
        {
            "community_id": 0,
            "size": 3,
            "core_members": ["a", "b"],
            "all_members": ["a", "b", "c"],
        },
        {
            "community_id": 1,
            "size": 3,
            "core_members": ["x", "y"],
            "all_members": ["x", "y", "unknown"],
        },
    ]
    path.write_text(json.dumps(communities), encoding="utf-8")


def test_evaluate_community_purity_exact_values(tmp_path) -> None:
    communities = tmp_path / "communities.json"
    profile = tmp_path / "profile.parquet"
    out_dir = tmp_path / "evaluation"
    _write_communities(communities)
    pl.DataFrame(
        {
            "character": ["a", "b", "c", "x", "y"],
            "dominant_copyright": ["ip_a", "ip_a", "ip_b", "ip_x", "ip_x"],
            "dominant_copyright_count": [10, 9, 7, 11, 10],
            "character_count": [10, 10, 10, 11, 10],
            "dominant_copyright_share": [1.0, 0.9, 0.7, 1.0, 1.0],
        }
    ).write_parquet(profile)

    out_path = evaluate_community_purity(communities, profile, out_dir)
    result = pl.read_parquet(out_path).sort("community_id")
    row0 = result.row(0, named=True)
    row1 = result.row(1, named=True)

    assert row0["dominant_copyright"] == "ip_a"
    assert math.isclose(row0["purity"], 2 / 3)
    assert math.isclose(row0["coverage"], 1.0)
    assert row1["dominant_copyright"] == "ip_x"
    assert math.isclose(row1["purity"], 1.0)
    assert math.isclose(row1["coverage"], 2 / 3)


def test_summarize_community_general_tags_uses_core_members(tmp_path) -> None:
    communities = tmp_path / "communities.json"
    edges = tmp_path / "edges_character_general.parquet"
    out_dir = tmp_path / "evaluation"
    _write_communities(communities)
    pl.DataFrame(
        {
            "tag_a": ["a", "b", "c", "x", "y"],
            "tag_b": ["halo", "halo", "tail", "animal_ears", "animal_ears"],
            "co_count": [20, 15, 99, 12, 11],
            "npmi": [0.5, 0.4, 0.9, 0.6, 0.7],
            "discounted_ppmi": [3.0, 2.0, 100.0, 4.0, 5.0],
        }
    ).write_parquet(edges)

    out_path = summarize_community_general_tags(edges, communities, out_dir, top_k=1, min_co_count=10)
    result = pl.read_parquet(out_path).sort("community_id")

    assert result["general_tag"].to_list() == ["halo", "animal_ears"]
    assert result["member_hits"].to_list() == [2, 2]


def test_build_character_copyright_profile_tie_breaks_by_name(tmp_path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_dir = tmp_path / "evaluation"
    pl.DataFrame(
        {
            "id": [1, 2, 3],
            "is_deleted": [False, False, False],
            "is_banned": [False, False, False],
            "tag_string_character": ["tie_char", "tie_char", "other_char"],
            "tag_string_copyright": ["ip_b", "ip_a", "ip_z"],
        }
    ).write_parquet(raw_path)

    out_path = build_character_copyright_profile(str(raw_path), out_dir, min_character_count=1)
    profile = pl.read_parquet(out_path)
    row = profile.filter(pl.col("character") == "tie_char").row(0, named=True)

    assert row["dominant_copyright"] == "ip_a"
    assert row["dominant_copyright_count"] == 1
    assert row["character_count"] == 2


def test_evaluation_cli_smoke(tmp_path) -> None:
    communities = tmp_path / "communities.json"
    profile = tmp_path / "profile.parquet"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "raw.parquet"
    edges = tmp_path / "edges_character_general.parquet"
    out_dir = tmp_path / "evaluation"
    _write_communities(communities)
    pl.DataFrame(
        {
            "id": [1, 2],
            "is_deleted": [False, False],
            "is_banned": [False, False],
            "tag_string_character": ["a b", "x y"],
            "tag_string_copyright": ["ip_a", "ip_x"],
        }
    ).write_parquet(raw_path)
    pl.DataFrame(
        {
            "character": ["a", "b", "c", "x", "y"],
            "dominant_copyright": ["ip_a", "ip_a", "ip_b", "ip_x", "ip_x"],
            "dominant_copyright_count": [10, 9, 7, 11, 10],
            "character_count": [10, 10, 10, 11, 10],
            "dominant_copyright_share": [1.0, 0.9, 0.7, 1.0, 1.0],
        }
    ).write_parquet(profile)
    pl.DataFrame(
        {
            "tag_a": ["a", "b", "x", "y"],
            "tag_b": ["halo", "halo", "animal_ears", "animal_ears"],
            "co_count": [20, 15, 12, 11],
            "npmi": [0.5, 0.4, 0.6, 0.7],
            "discounted_ppmi": [3.0, 2.0, 4.0, 5.0],
        }
    ).write_parquet(edges)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-copyright-profile",
            "--input",
            str(raw_dir),
            "--out",
            str(out_dir),
            "--min-character-count",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "character_copyright_profile.parquet").exists()

    result = runner.invoke(
        app,
        [
            "evaluate-purity",
            "--communities",
            str(communities),
            "--profile",
            str(profile),
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "communities_purity.parquet").exists()

    result = runner.invoke(
        app,
        [
            "summarize-general",
            "--communities",
            str(communities),
            "--edges",
            str(edges),
            "--out",
            str(out_dir),
            "--top-k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "communities_general_summary.parquet").exists()
