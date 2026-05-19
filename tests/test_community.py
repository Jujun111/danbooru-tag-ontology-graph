from __future__ import annotations

import json

import polars as pl
from typer.testing import CliRunner

from danbooru_graph.cli import app
from danbooru_graph.community import detect_communities, inspect_community


def _write_edges(processed_dir) -> None:
    processed_dir.mkdir()
    pl.DataFrame(
        {
            "tag_a": ["a", "a", "b", "x", "x", "y", "c"],
            "tag_b": ["b", "c", "c", "y", "z", "z", "x"],
            "npmi": [0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.1],
            "co_count": [20, 20, 20, 18, 18, 18, 100],
            "discounted_ppmi": [5.0, 4.0, 3.0, 6.0, 5.0, 4.0, 99.0],
        }
    ).write_parquet(processed_dir / "edges_character_character.parquet")


def test_detect_communities_filters_weak_edges_and_exports_schema(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    _write_edges(processed_dir)

    communities = detect_communities(
        processed_dir,
        min_npmi=0.5,
        min_co_count=10,
        resolution=1.0,
        seed=42,
        out_name="fixture",
    )

    assert len(communities) == 2
    assert {community["size"] for community in communities} == {3}
    assert all("core_members" in community for community in communities)
    assert all("all_members" in community for community in communities)
    assert inspect_community(processed_dir / "communities" / "fixture.json", "a") is not None

    with (processed_dir / "communities" / "fixture.json").open(encoding="utf-8") as handle:
        exported = json.load(handle)
    assert exported == communities

    summary = pl.read_csv(processed_dir / "communities" / "fixture_summary.csv")
    assert summary.columns == ["community_id", "size", "core_members"]


def test_community_cli_smoke(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    _write_edges(processed_dir)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "detect-communities",
            "--processed",
            str(processed_dir),
            "--min-npmi",
            "0.5",
            "--min-co-count",
            "10",
            "--resolution",
            "1.0",
            "--out-name",
            "fixture",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "inspect-community",
            "--communities",
            str(processed_dir / "communities" / "fixture.json"),
            "--tag",
            "a",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"community_id"' in result.output
