from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq


def _clean_tag_column(posts: pl.LazyFrame, column: str, output_name: str) -> pl.LazyFrame:
    raw = pl.col(column).fill_null("").str.strip_chars()
    return (
        posts.select(pl.col("id").alias("post_id"), raw.alias("raw_tags"))
        .with_columns(pl.col("raw_tags").str.split(" ").alias(output_name))
        .explode(output_name)
        .filter(pl.col(output_name).is_not_null() & (pl.col(output_name) != ""))
        .select("post_id", output_name)
        .unique(["post_id", output_name])
    )


def _scan_raw_character_copyright(raw_input: str) -> pl.LazyFrame:
    input_path = Path(raw_input)
    parquet_input = str(input_path / "*.parquet") if input_path.is_dir() else raw_input
    return (
        pl.scan_parquet(parquet_input)
        .select("id", "is_deleted", "is_banned", "tag_string_character", "tag_string_copyright")
        .filter((~pl.col("is_deleted").fill_null(False)) & (~pl.col("is_banned").fill_null(False)))
    )


def build_character_copyright_profile(
    raw_input: str,
    out_dir: Path,
    min_character_count: int = 50,
) -> Path:
    """Build a dominant-copyright profile for each sufficiently frequent character."""
    out_dir.mkdir(parents=True, exist_ok=True)
    posts = _scan_raw_character_copyright(raw_input)
    characters = _clean_tag_column(posts, "tag_string_character", "character")
    copyrights = _clean_tag_column(posts, "tag_string_copyright", "copyright")

    character_counts = characters.group_by("character").agg(pl.len().alias("character_count"))
    pair_counts = (
        characters.join(copyrights, on="post_id")
        .unique(["post_id", "character", "copyright"])
        .group_by("character", "copyright")
        .agg(pl.len().alias("copyright_co_count"))
    )

    profile = (
        pair_counts.join(character_counts, on="character")
        .filter(pl.col("character_count") >= min_character_count)
        .sort(["character", "copyright_co_count", "copyright"], descending=[False, True, False])
        .group_by("character", maintain_order=True)
        .agg(
            pl.first("copyright").alias("dominant_copyright"),
            pl.first("copyright_co_count").alias("dominant_copyright_count"),
            pl.first("character_count").alias("character_count"),
        )
        .with_columns(
            (
                pl.col("dominant_copyright_count").cast(pl.Float64)
                / pl.col("character_count").cast(pl.Float64)
            ).alias("dominant_copyright_share")
        )
        .sort("character")
    )

    out_path = out_dir / "character_copyright_profile.parquet"
    profile.sink_parquet(out_path, compression="zstd")
    return out_path


def _load_communities(communities_json: Path) -> list[dict[str, Any]]:
    with Path(communities_json).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_community_purity(
    communities_json: Path,
    copyright_profile: Path,
    out_dir: Path,
) -> Path:
    """Evaluate each community against dominant copyright labels as held-out ground truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    communities = _load_communities(communities_json)
    profile = pl.read_parquet(copyright_profile)
    copyright_by_character = dict(profile.select("character", "dominant_copyright").iter_rows())

    records = []
    for community in communities:
        members = community.get("all_members", [])
        labels = [
            copyright_by_character[member]
            for member in members
            if member in copyright_by_character
        ]
        counts = sorted(Counter(labels).items(), key=lambda item: (-item[1], item[0]))
        known_members = len(labels)
        size = int(community["size"])
        dominant_copyright = counts[0][0] if counts else None
        dominant_count = counts[0][1] if counts else 0
        records.append(
            {
                "community_id": int(community["community_id"]),
                "size": size,
                "known_members": known_members,
                "coverage": known_members / size if size else 0.0,
                "dominant_copyright": dominant_copyright,
                "dominant_copyright_members": dominant_count,
                "purity": dominant_count / known_members if known_members else None,
                "top_copyrights": "|".join(f"{name}:{count}" for name, count in counts[:10]),
            }
        )

    result = pl.DataFrame(records).sort(["purity", "coverage", "size"], descending=[True, True, True])
    stem = Path(communities_json).stem
    parquet_path = out_dir / f"{stem}_purity.parquet"
    csv_path = out_dir / f"{stem}_purity.csv"
    result.write_parquet(parquet_path)
    result.write_csv(csv_path)
    return parquet_path


def _community_members_frame(communities: list[dict[str, Any]], use_core_members: bool) -> pl.DataFrame:
    rows = []
    field = "core_members" if use_core_members else "all_members"
    for community in communities:
        for member in community.get(field, []):
            rows.append(
                {
                    "community_id": int(community["community_id"]),
                    "size": int(community["size"]),
                    "character": member,
                }
            )
    return pl.DataFrame(
        rows,
        schema={"community_id": pl.Int64, "size": pl.Int64, "character": pl.String},
    )


def summarize_community_general_tags(
    edges_character_general: Path,
    communities_json: Path,
    out_dir: Path,
    top_k: int = 20,
    min_co_count: int = 10,
    use_core_members: bool = True,
) -> Path:
    """Aggregate high-scoring general tags for each character community."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    out_dir.mkdir(parents=True, exist_ok=True)
    communities = _load_communities(communities_json)
    members = _community_members_frame(communities, use_core_members)

    edges = (
        pl.scan_parquet(edges_character_general)
        .filter(pl.col("co_count") >= min_co_count)
        .select(
            pl.col("tag_a").alias("character"),
            pl.col("tag_b").alias("general_tag"),
            "co_count",
            "npmi",
            "discounted_ppmi",
        )
        .collect()
    )

    summary = (
        members.join(edges, on="character")
        .group_by("community_id", "size", "general_tag")
        .agg(
            pl.n_unique("character").alias("member_hits"),
            pl.sum("co_count").alias("total_co_count"),
            pl.mean("npmi").alias("mean_npmi"),
            pl.mean("discounted_ppmi").alias("mean_discounted_ppmi"),
            pl.sum("discounted_ppmi").alias("sum_discounted_ppmi"),
        )
        .sort(
            ["community_id", "sum_discounted_ppmi", "member_hits", "general_tag"],
            descending=[False, True, True, False],
        )
        .with_columns((pl.int_range(pl.len()).over("community_id") + 1).alias("rank"))
        .filter(pl.col("rank") <= top_k)
        .select(
            "community_id",
            "size",
            "rank",
            "general_tag",
            "member_hits",
            "total_co_count",
            "mean_npmi",
            "mean_discounted_ppmi",
            "sum_discounted_ppmi",
        )
    )

    stem = Path(communities_json).stem
    parquet_path = out_dir / f"{stem}_general_summary.parquet"
    csv_path = out_dir / f"{stem}_general_summary.csv"
    summary.write_parquet(parquet_path)
    summary.write_csv(csv_path)
    return parquet_path


def _iter_raw_parquet_files(raw_input: str) -> list[Path]:
    input_path = Path(raw_input)
    if input_path.is_dir():
        return sorted(input_path.glob("*.parquet"))
    if any(marker in raw_input for marker in "*?[]"):
        return sorted(input_path.parent.glob(input_path.name))
    return [input_path]


def _split_tags(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {tag for tag in text.split(" ") if tag}


def summarize_community_general_tags_from_raw(
    raw_input: str,
    communities_json: Path,
    out_dir: Path,
    top_k: int = 20,
    min_co_count: int = 10,
    use_core_members: bool = True,
    discount_k: float = 10.0,
    batch_size: int = 50_000,
) -> Path:
    """Summarize community-level general tags directly from raw metadata.

    This avoids materializing the full character-general graph when only a compact
    community explanation table is needed.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    out_dir.mkdir(parents=True, exist_ok=True)
    communities = _load_communities(communities_json)
    field = "core_members" if use_core_members else "all_members"
    community_sizes = {int(community["community_id"]): int(community["size"]) for community in communities}
    character_to_communities: dict[str, set[int]] = {}
    for community in communities:
        community_id = int(community["community_id"])
        for member in community.get(field, []):
            character_to_communities.setdefault(member, set()).add(community_id)

    parquet_files = _iter_raw_parquet_files(raw_input)
    columns = ["is_deleted", "is_banned", "tag_string_character", "tag_string_general"]
    community_post_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, str]] = Counter()
    candidate_general_tags: set[str] = set()

    for path in parquet_files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pydict()
            for deleted, banned, character_text, general_text in zip(
                data["is_deleted"],
                data["is_banned"],
                data["tag_string_character"],
                data["tag_string_general"],
            ):
                if deleted or banned:
                    continue
                community_ids: set[int] = set()
                for character in _split_tags(character_text):
                    community_ids.update(character_to_communities.get(character, ()))
                if not community_ids:
                    continue

                general_tags = _split_tags(general_text)
                if not general_tags:
                    continue
                candidate_general_tags.update(general_tags)
                for community_id in community_ids:
                    community_post_counts[community_id] += 1
                    for general_tag in general_tags:
                        pair_counts[(community_id, general_tag)] += 1

    total_posts = 0
    general_counts: Counter[str] = Counter()
    for path in parquet_files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["is_deleted", "is_banned", "tag_string_general"],
        ):
            data = batch.to_pydict()
            for deleted, banned, general_text in zip(
                data["is_deleted"],
                data["is_banned"],
                data["tag_string_general"],
            ):
                if deleted or banned:
                    continue
                total_posts += 1
                for general_tag in _split_tags(general_text):
                    if general_tag in candidate_general_tags:
                        general_counts[general_tag] += 1

    rows = []
    total_posts_float = float(total_posts)
    for (community_id, general_tag), co_count in pair_counts.items():
        if co_count < min_co_count:
            continue
        community_count = community_post_counts[community_id]
        general_count = general_counts[general_tag]
        if not community_count or not general_count or not total_posts:
            continue
        prob_c = community_count / total_posts_float
        prob_g = general_count / total_posts_float
        prob_cg = co_count / total_posts_float
        lift = prob_cg / (prob_c * prob_g)
        pmi = math.log2(lift)
        ppmi = max(pmi, 0.0)
        npmi = 1.0 if prob_cg == 1.0 else pmi / (-math.log2(prob_cg))
        rows.append(
            {
                "community_id": community_id,
                "size": community_sizes.get(community_id, 0),
                "general_tag": general_tag,
                "co_count": co_count,
                "community_post_count": community_count,
                "general_count": general_count,
                "lift": lift,
                "pmi": pmi,
                "ppmi": ppmi,
                "npmi": npmi,
                "discounted_ppmi": ppmi * co_count / (co_count + discount_k),
            }
        )

    if rows:
        summary = (
            pl.DataFrame(rows)
            .sort(
                ["community_id", "discounted_ppmi", "co_count", "general_tag"],
                descending=[False, True, True, False],
            )
            .with_columns((pl.int_range(pl.len()).over("community_id") + 1).alias("rank"))
            .filter(pl.col("rank") <= top_k)
            .select(
                "community_id",
                "size",
                "rank",
                "general_tag",
                "co_count",
                "community_post_count",
                "general_count",
                "lift",
                "pmi",
                "ppmi",
                "npmi",
                "discounted_ppmi",
            )
        )
    else:
        summary = pl.DataFrame(
            schema={
                "community_id": pl.Int64,
                "size": pl.Int64,
                "rank": pl.Int64,
                "general_tag": pl.String,
                "co_count": pl.Int64,
                "community_post_count": pl.Int64,
                "general_count": pl.Int64,
                "lift": pl.Float64,
                "pmi": pl.Float64,
                "ppmi": pl.Float64,
                "npmi": pl.Float64,
                "discounted_ppmi": pl.Float64,
            }
        )

    stem = Path(communities_json).stem
    parquet_path = out_dir / f"{stem}_general_summary.parquet"
    csv_path = out_dir / f"{stem}_general_summary.csv"
    summary.write_parquet(parquet_path)
    summary.write_csv(csv_path)
    return parquet_path
