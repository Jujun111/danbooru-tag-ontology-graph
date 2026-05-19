from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import polars as pl

from danbooru_graph.constants import TAG_COLUMNS


BASE_COLUMNS = [
    "id",
    "rating",
    "is_deleted",
    "is_banned",
    *TAG_COLUMNS.values(),
]


def parse_ratings(ratings: str | None) -> list[str] | None:
    if not ratings:
        return None
    parsed = [rating.strip() for rating in ratings.split(",") if rating.strip()]
    return parsed or None


def parse_categories(categories: str | None) -> list[str]:
    if not categories:
        return list(TAG_COLUMNS)
    parsed = [category.strip() for category in categories.split(",") if category.strip()]
    unknown = sorted(set(parsed) - set(TAG_COLUMNS))
    if unknown:
        choices = ", ".join(sorted(TAG_COLUMNS))
        raise ValueError(f"Unknown categories {unknown}; expected one of: {choices}.")
    return parsed


def scan_posts(input_glob: str, ratings: Iterable[str] | None = None) -> pl.LazyFrame:
    """Read only columns required for tag graph construction."""
    input_path = Path(input_glob)
    parquet_input = str(input_path / "*.parquet") if input_path.is_dir() else input_glob
    lf = pl.scan_parquet(parquet_input).select(BASE_COLUMNS)
    lf = lf.filter(
        (~pl.col("is_deleted").fill_null(False))
        & (~pl.col("is_banned").fill_null(False))
    )

    selected_ratings = list(ratings or [])
    if selected_ratings:
        lf = lf.filter(pl.col("rating").is_in(selected_ratings))

    return lf


def explode_tag_column(posts: pl.LazyFrame, column: str, category: str) -> pl.LazyFrame:
    """Turn one space-separated tag column into clean long-form tag rows."""
    raw = pl.col(column).fill_null("").str.strip_chars()
    return (
        posts.select(pl.col("id").alias("post_id"), raw.alias("raw_tags"))
        .with_columns(pl.col("raw_tags").str.split(" ").alias("tag"))
        .explode("tag")
        .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
        .select(
            "post_id",
            pl.lit(category).alias("category"),
            pl.col("tag"),
        )
    )


def build_tag_long(posts: pl.LazyFrame) -> pl.LazyFrame:
    tag_frames = [
        explode_tag_column(posts, column, category)
        for category, column in TAG_COLUMNS.items()
    ]
    return pl.concat(tag_frames).unique(["post_id", "category", "tag"])


def _write_lazy_parquet(lf: pl.LazyFrame, path: Path) -> None:
    """Write a lazy query directly to parquet to keep peak memory lower."""
    lf.sink_parquet(path, compression="zstd")


def prepare_vocab(
    input_glob: str,
    out_dir: Path,
    min_tag_count: int = 50,
    ratings: str | None = None,
    categories: str | None = None,
) -> None:
    """Create posts, tag vocabulary, and post-tag mapping parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    posts_path = out_dir / "posts.parquet"
    vocab_path = out_dir / "tag_vocab.parquet"
    post_tags_path = out_dir / "post_tags.parquet"
    tmp_dir = out_dir / "_tmp_post_tags"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    posts_source = scan_posts(input_glob, parse_ratings(ratings))
    selected_categories = parse_categories(categories)
    posts = (
        posts_source.select(pl.col("id").alias("post_id"))
        .unique()
        .sort("post_id")
        .with_row_index("post_idx")
    )
    _write_lazy_parquet(posts, posts_path)

    vocab_parts = []
    for category in selected_categories:
        column = TAG_COLUMNS[category]
        category_vocab = (
            explode_tag_column(posts_source, column, category)
            .unique(["post_id", "category", "tag"])
            .group_by("category", "tag")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= min_tag_count)
            .collect()
        )
        vocab_parts.append(category_vocab)

    if vocab_parts:
        tag_vocab = (
            pl.concat(vocab_parts)
            .lazy()
            .sort(["category", "count", "tag"], descending=[False, True, False])
            .with_row_index("tag_id")
        )
    else:
        tag_vocab = pl.DataFrame(
            schema={"tag_id": pl.UInt32, "category": pl.String, "tag": pl.String, "count": pl.UInt32}
        ).lazy()
    _write_lazy_parquet(tag_vocab, vocab_path)

    for category in selected_categories:
        column = TAG_COLUMNS[category]
        category_post_tags = (
            explode_tag_column(posts_source, column, category)
            .unique(["post_id", "category", "tag"])
            .join(
                pl.scan_parquet(vocab_path)
                .filter(pl.col("category") == category)
                .select("tag_id", "category", "tag"),
                on=["category", "tag"],
            )
            .join(pl.scan_parquet(posts_path), on="post_id")
            .select("post_idx", "tag_id", "category")
            .sort(["post_idx", "tag_id"])
        )
        _write_lazy_parquet(category_post_tags, tmp_dir / f"post_tags_{category}.parquet")

    post_tag_files = sorted(str(path) for path in tmp_dir.glob("post_tags_*.parquet"))
    if post_tag_files:
        post_tags = pl.scan_parquet(post_tag_files).sort(["post_idx", "tag_id"])
    else:
        post_tags = pl.DataFrame(
            schema={"post_idx": pl.UInt32, "tag_id": pl.UInt32, "category": pl.String}
        ).lazy()
    _write_lazy_parquet(post_tags, post_tags_path)

    shutil.rmtree(tmp_dir)
