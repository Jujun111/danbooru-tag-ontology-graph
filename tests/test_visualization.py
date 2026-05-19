from __future__ import annotations

import json

import networkx as nx
import polars as pl
from typer.testing import CliRunner

from danbooru_graph.cli import app
from danbooru_graph.visualization import export_community_graph


def _write_fixture(tmp_path):
    communities = tmp_path / "communities.json"
    edges = tmp_path / "edges.parquet"
    communities.write_text(
        json.dumps(
            [
                {
                    "community_id": 12,
                    "size": 3,
                    "core_members": ["asuna", "karin"],
                    "all_members": ["asuna", "karin", "neru"],
                }
            ]
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "tag_a": ["asuna", "asuna", "asuna"],
            "tag_b": ["karin", "neru", "outside"],
            "co_count": [20, 5, 100],
            "lift": [10.0, 3.0, 1.0],
            "pmi": [3.0, 1.5, 0.0],
            "ppmi": [3.0, 1.5, 0.0],
            "npmi": [0.6, 0.2, 0.9],
            "discounted_ppmi": [2.0, 0.5, 9.0],
            "confidence_a_to_b": [0.2, 0.1, 0.9],
            "confidence_b_to_a": [0.3, 0.2, 0.8],
        }
    ).write_parquet(edges)
    return communities, edges


def test_export_community_graph_gexf_filters_and_adds_attributes(tmp_path) -> None:
    communities, edges = _write_fixture(tmp_path)
    out = tmp_path / "community.gexf"

    export_community_graph(
        edges,
        communities,
        out,
        community_id=12,
        min_npmi=0.5,
        min_co_count=10,
    )

    graph = nx.read_gexf(out)
    assert set(graph.nodes) == {"asuna", "karin", "neru"}
    assert set(graph.edges) == {("asuna", "karin")}
    assert float(graph.edges["asuna", "karin"]["weight"]) == 2.0
    assert int(graph.nodes["asuna"]["degree"]) == 1
    assert graph.nodes["asuna"]["is_core"] in {True, "true", "True"}


def test_export_community_graph_cli_smoke(tmp_path) -> None:
    communities, edges = _write_fixture(tmp_path)
    out = tmp_path / "community.graphml"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export-community-graph",
            "--edges",
            str(edges),
            "--communities",
            str(communities),
            "--out",
            str(out),
            "--community-id",
            "12",
            "--min-npmi",
            "0.5",
            "--min-co-count",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    graph = nx.read_graphml(out)
    assert set(graph.edges) == {("asuna", "karin")}
