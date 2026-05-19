from __future__ import annotations

import json

import polars as pl
from typer.testing import CliRunner

from danbooru_graph.cli import app
from danbooru_graph.recommendation import RecommendationEngine


def _write_recommendation_fixtures(tmp_path):
    edges = tmp_path / "edges.parquet"
    communities = tmp_path / "communities.json"
    summary = tmp_path / "general_summary.parquet"

    pl.DataFrame(
        {
            "tag_a": ["asuna", "asuna", "neru", "karin"],
            "tag_b": ["karin", "akane", "karin", "akane"],
            "discounted_ppmi": [10.0, 8.0, 9.0, 3.0],
            "confidence_a_to_b": [0.2, 0.4, 0.5, 0.1],
            "confidence_b_to_a": [0.3, 0.1, 0.6, 0.2],
            "co_count": [20, 15, 18, 8],
            "npmi": [0.5, 0.4, 0.55, 0.2],
        }
    ).write_parquet(edges)

    communities.write_text(
        json.dumps(
            [
                {
                    "community_id": 1,
                    "size": 10,
                    "core_members": ["asuna", "karin"],
                    "all_members": ["asuna", "karin", "neru"],
                },
                {
                    "community_id": 2,
                    "size": 3,
                    "core_members": ["asuna"],
                    "all_members": ["asuna"],
                },
            ]
        ),
        encoding="utf-8",
    )

    pl.DataFrame(
        {
            "community_id": [1, 1, 2],
            "size": [10, 10, 3],
            "rank": [1, 2, 1],
            "general_tag": ["sukajan", "sig_mpx", "halo"],
            "co_count": [100, 80, 50],
            "npmi": [0.6, 0.5, 0.7],
            "discounted_ppmi": [7.0, 6.0, 9.0],
        }
    ).write_parquet(summary)

    return edges, communities, summary


def test_single_tag_character_recommendation_uses_combined_score(tmp_path) -> None:
    edges, communities, summary = _write_recommendation_fixtures(tmp_path)
    engine = RecommendationEngine.from_artifacts(edges, communities, summary)

    recommendations = engine.recommend(["asuna"], target_category="character", top_k=2)

    assert recommendations[0]["tag"] == "karin"
    assert recommendations[0]["strategy"] == "neighbors"
    assert recommendations[0]["score"] > recommendations[1]["score"]


def test_multi_tag_weighted_sum_aggregation(tmp_path) -> None:
    edges, communities, summary = _write_recommendation_fixtures(tmp_path)
    engine = RecommendationEngine.from_artifacts(edges, communities, summary)

    recommendations = engine.recommend(["asuna", "neru"], target_category="character", top_k=3)
    karin = next(item for item in recommendations if item["tag"] == "karin")

    assert sorted(karin["source_tags"]) == ["asuna", "neru"]
    assert len(karin["score_components"]) == 2


def test_community_motif_chooses_most_matched_community(tmp_path) -> None:
    edges, communities, summary = _write_recommendation_fixtures(tmp_path)
    engine = RecommendationEngine.from_artifacts(edges, communities, summary)

    recommendations = engine.recommend(["asuna", "neru"], target_category="general", top_k=2)

    assert recommendations[0]["tag"] == "sukajan"
    assert recommendations[0]["community_id"] == 1
    assert recommendations[1]["tag"] == "sig_mpx"


def test_recommendation_cli_smoke(tmp_path) -> None:
    edges, communities, summary = _write_recommendation_fixtures(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "recommend-tags",
            "--tags",
            "asuna,neru",
            "--target-category",
            "character",
            "--edges",
            str(edges),
            "--communities",
            str(communities),
            "--general-summary",
            str(summary),
            "--top-k",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "karin" in result.output

    result = runner.invoke(
        app,
        [
            "recommend-tags",
            "--tags",
            "asuna,neru",
            "--target-category",
            "general",
            "--edges",
            str(edges),
            "--communities",
            str(communities),
            "--general-summary",
            str(summary),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "sukajan" in result.output
