from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import polars as pl


DEFAULT_EDGES_PATH = Path("data/processed/edges_character_character.parquet")
DEFAULT_COMMUNITIES_PATH = Path(
    "data/processed/communities/character_communities_npmi0.15_co15_res1.2.json"
)
DEFAULT_GENERAL_SUMMARY_PATH = Path(
    "data/processed/evaluation/character_communities_npmi0.15_co15_res1.2_general_summary.parquet"
)


@dataclass
class RecommendationAccumulator:
    tag: str
    strategy: str
    score: float = 0.0
    source_tags: set[str] = field(default_factory=set)
    community_id: int | None = None
    score_components: list[dict[str, Any]] = field(default_factory=list)

    def add(self, score: float, source_tag: str, component: dict[str, Any]) -> None:
        self.score += score
        self.source_tags.add(source_tag)
        self.score_components.append(component)

    def to_dict(self) -> dict[str, Any]:
        community = "" if self.community_id is None else f" from community {self.community_id}"
        sources = ", ".join(sorted(self.source_tags))
        return {
            "tag": self.tag,
            "score": self.score,
            "strategy": self.strategy,
            "source_tags": sorted(self.source_tags),
            "community_id": self.community_id,
            "score_components": self.score_components,
            "explanation": f"{self.strategy} recommendation{community}; supported by {sources}.",
        }


def parse_tags(tags: str | Iterable[str]) -> list[str]:
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    return [tag.strip() for tag in tags if tag.strip()]


class RecommendationEngine:
    def __init__(
        self,
        edges: pl.DataFrame | None = None,
        communities: list[dict[str, Any]] | None = None,
        general_summary: pl.DataFrame | None = None,
    ) -> None:
        self.edges = edges
        self.communities = communities or []
        self.general_summary = general_summary
        self._communities_by_tag = self._build_community_index(self.communities)

    @classmethod
    def from_artifacts(
        cls,
        edges_path: Path = DEFAULT_EDGES_PATH,
        communities_path: Path = DEFAULT_COMMUNITIES_PATH,
        general_summary_path: Path = DEFAULT_GENERAL_SUMMARY_PATH,
    ) -> "RecommendationEngine":
        edges = pl.read_parquet(edges_path) if Path(edges_path).exists() else None
        if Path(communities_path).exists():
            with Path(communities_path).open("r", encoding="utf-8") as handle:
                communities = json.load(handle)
        else:
            communities = []
        general_summary = (
            pl.read_parquet(general_summary_path)
            if Path(general_summary_path).exists()
            else None
        )
        return cls(edges=edges, communities=communities, general_summary=general_summary)

    @staticmethod
    def _build_community_index(communities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        for community in communities:
            for tag in community.get("all_members", []):
                index.setdefault(tag, []).append(community)
        return index

    def recommend(
        self,
        current_tags: str | Iterable[str],
        target_category: str = "character",
        top_k: int = 10,
        strategies: Iterable[str] = ("neighbors", "community"),
    ) -> list[dict[str, Any]]:
        tags = parse_tags(current_tags)
        if not tags or top_k <= 0:
            return []
        if target_category not in {"character", "general"}:
            raise ValueError("target_category must be either 'character' or 'general'.")

        strategy_set = set(strategies)
        input_tags = set(tags)
        accumulators: dict[tuple[str, str, int | None], RecommendationAccumulator] = {}

        if target_category == "character" and "neighbors" in strategy_set:
            self._add_neighbor_recommendations(tags, input_tags, accumulators)
        if target_category == "general" and "community" in strategy_set:
            self._add_community_recommendations(tags, input_tags, accumulators)

        ranked = sorted(
            (acc.to_dict() for acc in accumulators.values()),
            key=lambda item: (-item["score"], item["tag"], item["strategy"]),
        )
        return ranked[:top_k]

    def _get_accumulator(
        self,
        accumulators: dict[tuple[str, str, int | None], RecommendationAccumulator],
        tag: str,
        strategy: str,
        community_id: int | None = None,
    ) -> RecommendationAccumulator:
        key = (tag, strategy, community_id)
        if key not in accumulators:
            accumulators[key] = RecommendationAccumulator(
                tag=tag,
                strategy=strategy,
                community_id=community_id,
            )
        return accumulators[key]

    def _add_neighbor_recommendations(
        self,
        tags: list[str],
        input_tags: set[str],
        accumulators: dict[tuple[str, str, int | None], RecommendationAccumulator],
    ) -> None:
        if self.edges is None:
            return

        for source_tag in tags:
            left = (
                self.edges.filter(pl.col("tag_a") == source_tag)
                .select(
                    pl.col("tag_b").alias("tag"),
                    "discounted_ppmi",
                    pl.col("confidence_a_to_b").alias("confidence"),
                    "co_count",
                    "npmi",
                )
            )
            right = (
                self.edges.filter(pl.col("tag_b") == source_tag)
                .select(
                    pl.col("tag_a").alias("tag"),
                    "discounted_ppmi",
                    pl.col("confidence_b_to_a").alias("confidence"),
                    "co_count",
                    "npmi",
                )
            )
            neighbors = pl.concat([left, right])
            for row in neighbors.iter_rows(named=True):
                candidate = row["tag"]
                if candidate in input_tags:
                    continue
                score = 0.7 * float(row["discounted_ppmi"]) + 0.3 * float(row["confidence"])
                accumulator = self._get_accumulator(accumulators, candidate, "neighbors")
                accumulator.add(
                    score,
                    source_tag,
                    {
                        "source_tag": source_tag,
                        "score": score,
                        "discounted_ppmi": float(row["discounted_ppmi"]),
                        "confidence": float(row["confidence"]),
                        "co_count": int(row["co_count"]),
                        "npmi": float(row["npmi"]),
                    },
                )

    def _dominant_community(self, tags: list[str]) -> dict[str, Any] | None:
        matches: dict[int, tuple[dict[str, Any], set[str]]] = {}
        for tag in tags:
            for community in self._communities_by_tag.get(tag, []):
                community_id = int(community["community_id"])
                if community_id not in matches:
                    matches[community_id] = (community, set())
                matches[community_id][1].add(tag)

        if not matches:
            return None

        return sorted(
            matches.values(),
            key=lambda item: (-len(item[1]), int(item[0]["size"]), int(item[0]["community_id"])),
        )[0][0]

    def _add_community_recommendations(
        self,
        tags: list[str],
        input_tags: set[str],
        accumulators: dict[tuple[str, str, int | None], RecommendationAccumulator],
    ) -> None:
        if self.general_summary is None:
            return

        community = self._dominant_community(tags)
        if community is None:
            return
        community_id = int(community["community_id"])
        summary = self.general_summary.filter(pl.col("community_id") == community_id)

        community_members = set(community.get("all_members", []))
        source_tags = [tag for tag in tags if tag in community_members] or ["community"]
        for row in summary.iter_rows(named=True):
            candidate = row["general_tag"]
            if candidate in input_tags:
                continue
            score = float(row["discounted_ppmi"])
            accumulator = self._get_accumulator(accumulators, candidate, "community", community_id)
            for source_tag in source_tags:
                accumulator.add(
                    score,
                    source_tag,
                    {
                        "source_tag": source_tag,
                        "score": score,
                        "community_id": community_id,
                        "rank": int(row["rank"]),
                        "co_count": int(row["co_count"]),
                        "npmi": float(row["npmi"]),
                    },
                )
