from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl


def _format_float(value: float) -> str:
    return f"{value:g}"


def _default_stem(min_npmi: float, min_co_count: int, resolution: float) -> str:
    return (
        f"character_communities_npmi{_format_float(min_npmi)}"
        f"_co{min_co_count}_res{_format_float(resolution)}"
    )


def _load_filtered_edges(processed_dir: Path, min_npmi: float, min_co_count: int) -> pl.DataFrame:
    edges_path = processed_dir / "edges_character_character.parquet"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing scored character edge table: {edges_path}")

    return (
        pl.scan_parquet(edges_path)
        .filter((pl.col("npmi") >= min_npmi) & (pl.col("co_count") >= min_co_count))
        .select("tag_a", "tag_b", "discounted_ppmi")
        .collect()
    )


def _build_graph(edges: pl.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges.iter_rows(), weight="weight")
    return graph


def _community_record(
    community_id: int,
    graph: nx.Graph,
    members: set[str],
    top_members: int,
) -> dict[str, Any]:
    subgraph = graph.subgraph(members)
    central_nodes = sorted(
        subgraph.degree(weight="weight"),
        key=lambda item: (-item[1], item[0]),
    )
    all_members = sorted(members)
    return {
        "community_id": community_id,
        "size": len(members),
        "core_members": [node for node, _degree in central_nodes[:top_members]],
        "all_members": all_members,
    }


def detect_communities(
    processed_dir: Path,
    min_npmi: float = 0.15,
    min_co_count: int = 15,
    resolution: float = 1.2,
    seed: int = 42,
    min_size: int = 3,
    top_members: int = 15,
    out_name: str | None = None,
) -> list[dict[str, Any]]:
    """Detect weighted Louvain communities from the scored character-character graph."""
    if min_size < 1:
        raise ValueError("min_size must be at least 1.")
    if top_members < 1:
        raise ValueError("top_members must be at least 1.")

    processed_dir = Path(processed_dir)
    edges = _load_filtered_edges(processed_dir, min_npmi, min_co_count)
    graph = _build_graph(edges)

    if graph.number_of_edges() == 0:
        communities: list[set[str]] = []
    else:
        communities = [
            set(community)
            for community in nx.community.louvain_communities(
                graph,
                weight="weight",
                resolution=resolution,
                seed=seed,
            )
        ]

    filtered = [community for community in communities if len(community) >= min_size]
    sorted_communities = sorted(filtered, key=lambda members: (-len(members), sorted(members)[0]))
    records = [
        _community_record(index, graph, community, top_members)
        for index, community in enumerate(sorted_communities)
    ]

    out_dir = processed_dir / "communities"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_name or _default_stem(min_npmi, min_co_count, resolution)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    pl.DataFrame(
        {
            "community_id": [record["community_id"] for record in records],
            "size": [record["size"] for record in records],
            "core_members": ["|".join(record["core_members"]) for record in records],
        },
        schema={"community_id": pl.Int64, "size": pl.Int64, "core_members": pl.String},
    ).write_csv(csv_path)

    return records


def inspect_community(communities_path: Path, tag: str) -> dict[str, Any] | None:
    """Return the community record containing a tag, if present."""
    with Path(communities_path).open("r", encoding="utf-8") as handle:
        communities = json.load(handle)

    for community in communities:
        if tag in community.get("all_members", []):
            return community
    return None
