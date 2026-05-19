from __future__ import annotations

from pathlib import Path

import polars as pl

from danbooru_graph.pairs import edge_file_stem, parse_pair


SCORE_SORT_COLUMNS = frozenset(
    {
        "discounted_ppmi",
        "npmi",
        "ppmi",
        "pmi",
        "lift",
        "co_count",
    }
)


def default_top_path(processed_dir: Path, pair: str, sort_by: str, top_k: int) -> Path:
    return processed_dir / f"top_{pair.replace('-', '_')}_{sort_by}_{top_k}.csv"


def _validate_sort_by(sort_by: str) -> None:
    if sort_by not in SCORE_SORT_COLUMNS:
        choices = ", ".join(sorted(SCORE_SORT_COLUMNS))
        raise ValueError(f"Unknown sort column {sort_by!r}; expected one of: {choices}.")


def score_edges(
    processed_dir: Path,
    pair: str,
    discount_k: float = 10.0,
    sort_by: str = "discounted_ppmi",
    top_k: int = 0,
    top_out: Path | None = None,
) -> Path:
    """Add tag metadata plus robust association scores to an edge count table."""
    parse_pair(pair)
    _validate_sort_by(sort_by)
    if discount_k < 0:
        raise ValueError("discount_k must be non-negative.")
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")

    num_posts = pl.scan_parquet(processed_dir / "posts.parquet").select(pl.len()).collect().item()
    if num_posts <= 0:
        raise ValueError("Cannot score edges without any posts.")
    num_posts_lit = pl.lit(float(num_posts))
    vocab = pl.read_parquet(processed_dir / "tag_vocab.parquet")
    counts_path = processed_dir / f"{edge_file_stem(pair)}_counts.parquet"
    edges = pl.read_parquet(counts_path)

    vocab_a = vocab.rename(
        {
            "tag_id": "tag_a_id",
            "category": "category_a",
            "tag": "tag_a",
            "count": "count_a",
        }
    )
    vocab_b = vocab.rename(
        {
            "tag_id": "tag_b_id",
            "category": "category_b",
            "tag": "tag_b",
            "count": "count_b",
        }
    )

    scored = (
        edges.join(vocab_a, on="tag_a_id")
        .join(vocab_b, on="tag_b_id")
        .with_columns(
            (pl.col("count_a").cast(pl.Float64) / num_posts_lit).alias("prob_a"),
            (pl.col("count_b").cast(pl.Float64) / num_posts_lit).alias("prob_b"),
            (pl.col("co_count").cast(pl.Float64) / num_posts_lit).alias("prob_ab"),
            (pl.col("co_count").cast(pl.Float64) / pl.col("count_a").cast(pl.Float64)).alias(
                "confidence_a_to_b"
            ),
            (pl.col("co_count").cast(pl.Float64) / pl.col("count_b").cast(pl.Float64)).alias(
                "confidence_b_to_a"
            ),
        )
        .with_columns((pl.col("prob_ab") / (pl.col("prob_a") * pl.col("prob_b"))).alias("lift"))
        .with_columns(pl.col("lift").log(base=2).alias("pmi"))
        .with_columns(
            pl.max_horizontal(pl.col("pmi"), pl.lit(0.0)).alias("ppmi"),
            pl.when(pl.col("prob_ab") == 1.0)
            .then(pl.lit(1.0))
            .otherwise(pl.col("pmi") / (-pl.col("prob_ab").log(base=2)))
            .alias("npmi"),
            (
                pl.max_horizontal(pl.col("pmi"), pl.lit(0.0))
                * pl.col("co_count").cast(pl.Float64)
                / (pl.col("co_count").cast(pl.Float64) + pl.lit(float(discount_k)))
            ).alias("discounted_ppmi"),
        )
        .select(
            "tag_a_id",
            "tag_b_id",
            "category_a",
            "tag_a",
            "category_b",
            "tag_b",
            "count_a",
            "count_b",
            "co_count",
            "prob_a",
            "prob_b",
            "prob_ab",
            "lift",
            "pmi",
            "ppmi",
            "npmi",
            "confidence_a_to_b",
            "confidence_b_to_a",
            "discounted_ppmi",
        )
        .sort([sort_by, "co_count", "tag_a_id", "tag_b_id"], descending=[True, True, False, False])
    )

    out_path = processed_dir / f"{edge_file_stem(pair)}.parquet"
    scored.write_parquet(out_path)

    if top_k:
        top_path = top_out or default_top_path(processed_dir, pair, sort_by, top_k)
        top_path.parent.mkdir(parents=True, exist_ok=True)
        scored.head(top_k).write_csv(top_path)

    return out_path
