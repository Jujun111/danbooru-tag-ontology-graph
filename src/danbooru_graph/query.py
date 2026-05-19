from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import polars as pl


DEFAULT_RAW_INPUT = "data/raw/danbooru-2026/output_part_*.parquet"
DEFAULT_CHARACTER_VOCAB_PATH = Path("data/processed/tag_vocab.parquet")
DEFAULT_GENDER_PROFILE_PATH = Path("data/processed/evaluation/character_gender_profile.parquet")
QUERY_RANK_COLUMNS = {
    "co_count",
    "confidence_query_to_character",
    "confidence_character_to_query",
    "lift",
    "pmi",
    "ppmi",
}


def parse_tag_list(tags: str | Iterable[str] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    return [tag.strip() for tag in tags if tag.strip()]


def _parquet_input(raw_input: str) -> str:
    input_path = Path(raw_input)
    return str(input_path / "*.parquet") if input_path.is_dir() else raw_input


def _scan_raw_character_general(raw_input: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet(_parquet_input(raw_input))
        .select("id", "is_deleted", "is_banned", "tag_string_character", "tag_string_general")
        .filter((~pl.col("is_deleted").fill_null(False)) & (~pl.col("is_banned").fill_null(False)))
    )


def _has_general_tag(tag: str) -> pl.Expr:
    pattern = rf"(^| ){re.escape(tag)}( |$)"
    return pl.col("tag_string_general").fill_null("").str.contains(pattern)


def _general_condition(tags: list[str], mode: str) -> pl.Expr:
    if mode not in {"and", "or"}:
        raise ValueError("mode must be either 'and' or 'or'.")
    if not tags:
        return pl.lit(True)

    expressions = [_has_general_tag(tag) for tag in tags]
    condition = expressions[0]
    for expression in expressions[1:]:
        condition = condition & expression if mode == "and" else condition | expression
    return condition


def _explode_characters(posts: pl.LazyFrame, extra_columns: list[str] | None = None) -> pl.LazyFrame:
    extra_columns = extra_columns or []
    raw = pl.col("tag_string_character").fill_null("").str.strip_chars()
    return (
        posts.select(
            pl.col("id").alias("post_id"),
            raw.alias("raw_characters"),
            *extra_columns,
        )
        .with_columns(pl.col("raw_characters").str.split(" ").alias("character"))
        .explode("character")
        .filter(pl.col("character").is_not_null() & (pl.col("character") != ""))
        .select("post_id", "character", *extra_columns)
        .unique(["post_id", "character"])
    )


def _character_counts(posts: pl.LazyFrame, character_vocab: Path | None) -> pl.LazyFrame:
    if character_vocab and Path(character_vocab).exists():
        return (
            pl.scan_parquet(character_vocab)
            .filter(pl.col("category") == "character")
            .select(
                pl.col("tag").alias("character"),
                pl.col("count").cast(pl.Int64).alias("character_count"),
            )
        )
    return _explode_characters(posts).group_by("character").agg(pl.len().alias("character_count"))


def build_character_gender_profile(
    raw_input: str,
    out_dir: Path,
    min_character_count: int = 50,
    min_gender_evidence: int = 10,
    female_threshold: float = 0.8,
) -> Path:
    """Build an empirical gender profile from solo 1girl/1boy metadata signals."""
    if not 0.5 <= female_threshold <= 1.0:
        raise ValueError("female_threshold must be between 0.5 and 1.0.")

    out_dir.mkdir(parents=True, exist_ok=True)
    posts = _scan_raw_character_general(raw_input)
    characters = _explode_characters(posts)
    character_counts = characters.group_by("character").agg(pl.len().alias("character_count"))

    gendered_posts = (
        posts.with_columns(
            _has_general_tag("solo").alias("_solo"),
            _has_general_tag("1girl").alias("_female_signal"),
            _has_general_tag("1boy").alias("_male_signal"),
        )
        .filter(pl.col("_solo") & (pl.col("_female_signal") | pl.col("_male_signal")))
        .select("id", "tag_string_character", "_female_signal", "_male_signal")
    )
    gender_counts = (
        _explode_characters(gendered_posts, ["_female_signal", "_male_signal"])
        .group_by("character")
        .agg(
            pl.sum("_female_signal").cast(pl.Int64).alias("female_solo_count"),
            pl.sum("_male_signal").cast(pl.Int64).alias("male_solo_count"),
        )
    )

    evidence = pl.col("female_solo_count") + pl.col("male_solo_count")
    profile = (
        character_counts.join(gender_counts, on="character", how="left")
        .with_columns(
            pl.col("female_solo_count").fill_null(0),
            pl.col("male_solo_count").fill_null(0),
        )
        .with_columns(evidence.alias("gender_evidence"))
        .with_columns(
            pl.when(pl.col("gender_evidence") > 0)
            .then(pl.col("female_solo_count") / pl.col("gender_evidence"))
            .otherwise(None)
            .alias("female_score")
        )
        .with_columns(
            pl.when(pl.col("gender_evidence") < min_gender_evidence)
            .then(pl.lit("unknown"))
            .when(pl.col("female_score") >= female_threshold)
            .then(pl.lit("female"))
            .when(pl.col("female_score") <= (1.0 - female_threshold))
            .then(pl.lit("male"))
            .otherwise(pl.lit("ambiguous"))
            .alias("gender")
        )
        .filter(pl.col("character_count") >= min_character_count)
        .sort(["gender", "female_score", "character"], descending=[False, True, False])
    )

    out_path = out_dir / "character_gender_profile.parquet"
    profile.sink_parquet(out_path, compression="zstd")
    return out_path


def query_characters_by_general_tags(
    raw_input: str,
    include_general: str | Iterable[str] | None = None,
    mode: str = "and",
    top_k: int = 50,
    rank_by: str = "co_count",
    character_vocab: Path | None = DEFAULT_CHARACTER_VOCAB_PATH,
    gender_profile: Path | None = None,
    female_only: bool = False,
) -> pl.DataFrame:
    """Return characters most associated with a set of general-tag constraints."""
    include_tags = parse_tag_list(include_general)
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if rank_by not in QUERY_RANK_COLUMNS:
        choices = ", ".join(sorted(QUERY_RANK_COLUMNS))
        raise ValueError(f"rank_by must be one of: {choices}.")
    if female_only and not gender_profile:
        raise ValueError("female_only requires a gender_profile path.")

    posts = _scan_raw_character_general(raw_input)
    total_posts = int(posts.select(pl.len().alias("n")).collect().item())
    filtered = posts.filter(_general_condition(include_tags, mode))
    query_post_count = int(filtered.select(pl.n_unique("id").alias("n")).collect().item())

    empty_schema = {
        "character": pl.String,
        "co_count": pl.Int64,
        "query_post_count": pl.Int64,
        "character_count": pl.Int64,
        "confidence_query_to_character": pl.Float64,
        "confidence_character_to_query": pl.Float64,
        "lift": pl.Float64,
        "pmi": pl.Float64,
        "ppmi": pl.Float64,
    }
    if query_post_count == 0:
        return pl.DataFrame(schema=empty_schema)

    counts = _explode_characters(filtered).group_by("character").agg(pl.len().alias("co_count"))
    result = (
        counts.join(_character_counts(posts, character_vocab), on="character", how="left")
        .with_columns(pl.coalesce(["character_count", "co_count"]).cast(pl.Int64).alias("character_count"))
        .with_columns(
            pl.lit(query_post_count).alias("query_post_count"),
            (pl.col("co_count") / pl.lit(query_post_count)).alias("confidence_query_to_character"),
            (pl.col("co_count") / pl.col("character_count")).alias("confidence_character_to_query"),
            (
                pl.col("co_count").cast(pl.Float64)
                * pl.lit(float(total_posts))
                / (pl.lit(float(query_post_count)) * pl.col("character_count").cast(pl.Float64))
            ).alias("lift"),
        )
        .with_columns(pl.col("lift").log(base=2).alias("pmi"))
        .with_columns(pl.max_horizontal(pl.col("pmi"), pl.lit(0.0)).alias("ppmi"))
    )

    if gender_profile:
        profile = pl.scan_parquet(gender_profile).select(
            "character",
            "gender",
            "female_score",
            "gender_evidence",
            "female_solo_count",
            "male_solo_count",
        )
        if female_only:
            profile = profile.filter(pl.col("gender") == "female")
            result = result.join(profile, on="character", how="inner")
        else:
            result = result.join(profile, on="character", how="left")

    result = result.sort(
        [rank_by, "co_count", "character"],
        descending=[True, True, False],
    ).head(top_k)
    return result.collect() if isinstance(result, pl.LazyFrame) else result


def dataframe_to_records(df: pl.DataFrame) -> list[dict[str, object]]:
    records = []
    for row in df.iter_rows(named=True):
        cleaned = {}
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records
