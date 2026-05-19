from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl


def _load_communities(communities_path: Path) -> list[dict[str, Any]]:
    with Path(communities_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_community(
    communities_path: Path,
    community_id: int | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    if community_id is None and tag is None:
        raise ValueError("Provide either community_id or tag.")

    communities = _load_communities(communities_path)
    for community in communities:
        if community_id is not None and int(community["community_id"]) == community_id:
            return community
        if tag is not None and tag in community.get("all_members", []):
            return community

    selector = f"community_id={community_id}" if community_id is not None else f"tag={tag!r}"
    raise ValueError(f"No community found for {selector}.")


def _format_from_path(out_path: Path, graph_format: str | None) -> str:
    if graph_format:
        normalized = graph_format.lower()
    else:
        normalized = out_path.suffix.lower().lstrip(".")
    if normalized not in {"gexf", "graphml"}:
        raise ValueError("graph_format must be 'gexf' or 'graphml'.")
    return normalized


def export_community_graph(
    edges_path: Path,
    communities_path: Path,
    out_path: Path,
    community_id: int | None = None,
    tag: str | None = None,
    min_npmi: float = 0.15,
    min_co_count: int = 15,
    graph_format: str | None = None,
    include_isolates: bool = True,
) -> Path:
    """Export a community's internal weighted graph for Gephi."""
    community = _select_community(communities_path, community_id=community_id, tag=tag)
    members = set(community.get("all_members", []))
    core_members = set(community.get("core_members", []))
    if not members:
        raise ValueError("Selected community has no members.")

    edges = (
        pl.scan_parquet(edges_path)
        .filter(
            pl.col("tag_a").is_in(members)
            & pl.col("tag_b").is_in(members)
            & (pl.col("npmi") >= min_npmi)
            & (pl.col("co_count") >= min_co_count)
        )
        .select(
            "tag_a",
            "tag_b",
            "co_count",
            "lift",
            "pmi",
            "ppmi",
            "npmi",
            "discounted_ppmi",
            "confidence_a_to_b",
            "confidence_b_to_a",
        )
        .collect()
    )

    graph = nx.Graph(
        community_id=int(community["community_id"]),
        community_size=int(community["size"]),
        min_npmi=float(min_npmi),
        min_co_count=int(min_co_count),
    )
    if include_isolates:
        for member in sorted(members):
            graph.add_node(member, label=member, tag=member, is_core=member in core_members)

    for row in edges.iter_rows(named=True):
        graph.add_edge(
            row["tag_a"],
            row["tag_b"],
            weight=float(row["discounted_ppmi"]),
            discounted_ppmi=float(row["discounted_ppmi"]),
            npmi=float(row["npmi"]),
            co_count=int(row["co_count"]),
            lift=float(row["lift"]),
            pmi=float(row["pmi"]),
            ppmi=float(row["ppmi"]),
            confidence_a_to_b=float(row["confidence_a_to_b"]),
            confidence_b_to_a=float(row["confidence_b_to_a"]),
        )
        for node in (row["tag_a"], row["tag_b"]):
            graph.nodes[node].setdefault("label", node)
            graph.nodes[node].setdefault("tag", node)
            graph.nodes[node].setdefault("is_core", node in core_members)

    for node in graph.nodes:
        graph.nodes[node]["degree"] = int(graph.degree(node))
        graph.nodes[node]["weighted_degree"] = float(graph.degree(node, weight="weight"))
        graph.nodes[node]["gephi_size"] = 5.0 + graph.nodes[node]["weighted_degree"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = _format_from_path(out_path, graph_format)
    if output_format == "gexf":
        nx.write_gexf(graph, out_path)
    else:
        nx.write_graphml(graph, out_path)
    return out_path
