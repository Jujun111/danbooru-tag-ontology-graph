from __future__ import annotations

import json
from pathlib import Path

import typer

from danbooru_graph.constants import DEFAULT_MIN_PAIR_COUNT, DEFAULT_MIN_TAG_COUNT, DEFAULT_PAIRS
from danbooru_graph.community import detect_communities as detect_communities_pipeline
from danbooru_graph.community import inspect_community as inspect_community_pipeline
from danbooru_graph.etl import prepare_vocab as prepare_vocab_pipeline
from danbooru_graph.evaluation import build_character_copyright_profile as build_character_copyright_profile_pipeline
from danbooru_graph.evaluation import evaluate_community_purity as evaluate_community_purity_pipeline
from danbooru_graph.evaluation import summarize_community_general_tags as summarize_community_general_tags_pipeline
from danbooru_graph.evaluation import (
    summarize_community_general_tags_from_raw as summarize_community_general_tags_from_raw_pipeline,
)
from danbooru_graph.recommendation import (
    DEFAULT_COMMUNITIES_PATH,
    DEFAULT_EDGES_PATH,
    DEFAULT_GENERAL_SUMMARY_PATH,
    RecommendationEngine,
    parse_tags,
)
from danbooru_graph.query import (
    DEFAULT_CHARACTER_VOCAB_PATH,
    DEFAULT_GENDER_PROFILE_PATH,
    DEFAULT_RAW_INPUT,
    QUERY_RANK_COLUMNS,
    build_character_gender_profile as build_character_gender_profile_pipeline,
    dataframe_to_records,
    query_characters_by_general_tags,
)
from danbooru_graph.scoring import SCORE_SORT_COLUMNS, score_edges as score_edges_pipeline
from danbooru_graph.sparse_edges import build_edges as build_edges_pipeline
from danbooru_graph.visualization import export_community_graph as export_community_graph_pipeline

app = typer.Typer(help="Danbooru metadata tag graph mining CLI.")


@app.command("prepare-vocab")
def prepare_vocab(
    input: str = typer.Option(..., "--input", help="Parquet file path or glob."),
    out: Path = typer.Option(..., "--out", help="Processed output directory."),
    min_tag_count: int = typer.Option(DEFAULT_MIN_TAG_COUNT, "--min-tag-count", min=1),
    ratings: str | None = typer.Option(None, "--ratings", help="Optional comma list, e.g. g,s,q,e."),
    categories: str | None = typer.Option(
        None,
        "--categories",
        help="Optional comma list, e.g. character,general. Defaults to all categories.",
    ),
) -> None:
    prepare_vocab_pipeline(input, out, min_tag_count=min_tag_count, ratings=ratings, categories=categories)
    typer.echo(f"Wrote processed tables to {out}")


@app.command("build-edges")
def build_edges(
    processed: Path = typer.Option(..., "--processed", help="Processed directory."),
    pair: str = typer.Option(DEFAULT_PAIRS[1], "--pair", help="Category pair, e.g. character-general."),
    min_pair_count: int = typer.Option(DEFAULT_MIN_PAIR_COUNT, "--min-pair-count", min=1),
) -> None:
    out_path = build_edges_pipeline(processed, pair, min_pair_count=min_pair_count)
    typer.echo(f"Wrote edge counts to {out_path}")


@app.command("score-edges")
def score_edges(
    processed: Path = typer.Option(..., "--processed", help="Processed directory."),
    pair: str = typer.Option(DEFAULT_PAIRS[1], "--pair", help="Category pair, e.g. character-general."),
    discount_k: float = typer.Option(10.0, "--discount-k", min=0.0, help="Discount strength for PPMI."),
    sort_by: str = typer.Option(
        "discounted_ppmi",
        "--sort-by",
        help=f"Sort column. One of: {', '.join(sorted(SCORE_SORT_COLUMNS))}.",
    ),
    top_k: int = typer.Option(0, "--top-k", min=0, help="Write a separate top-K CSV when greater than 0."),
    top_out: Path | None = typer.Option(None, "--top-out", help="Optional path for the top-K CSV."),
) -> None:
    out_path = score_edges_pipeline(
        processed,
        pair,
        discount_k=discount_k,
        sort_by=sort_by,
        top_k=top_k,
        top_out=top_out,
    )
    typer.echo(f"Wrote scored edges to {out_path}")


@app.command("detect-communities")
def detect_communities(
    processed: Path = typer.Option(..., "--processed", help="Processed directory."),
    min_npmi: float = typer.Option(0.15, "--min-npmi", help="Minimum NPMI edge threshold."),
    min_co_count: int = typer.Option(15, "--min-co-count", min=1, help="Minimum co-occurrence count."),
    resolution: float = typer.Option(1.2, "--resolution", min=0.0, help="Louvain resolution."),
    seed: int = typer.Option(42, "--seed", help="Deterministic Louvain seed."),
    min_size: int = typer.Option(3, "--min-size", min=1, help="Minimum exported community size."),
    top_members: int = typer.Option(15, "--top-members", min=1, help="Core members per community."),
    out_name: str | None = typer.Option(None, "--out-name", help="Optional output file stem."),
) -> None:
    communities = detect_communities_pipeline(
        processed,
        min_npmi=min_npmi,
        min_co_count=min_co_count,
        resolution=resolution,
        seed=seed,
        min_size=min_size,
        top_members=top_members,
        out_name=out_name,
    )
    typer.echo(f"Detected {len(communities)} exported communities under {processed / 'communities'}")


@app.command("inspect-community")
def inspect_community(
    communities: Path = typer.Option(..., "--communities", help="Community JSON path."),
    tag: str = typer.Option(..., "--tag", help="Tag to search for."),
) -> None:
    community = inspect_community_pipeline(communities, tag)
    if community is None:
        raise typer.Exit(code=1)
    typer.echo(json.dumps(community, ensure_ascii=False, indent=2))


@app.command("build-copyright-profile")
def build_copyright_profile(
    input: str = typer.Option(..., "--input", help="Raw Danbooru Parquet path, directory, or glob."),
    out: Path = typer.Option(..., "--out", help="Evaluation output directory."),
    min_character_count: int = typer.Option(50, "--min-character-count", min=1),
) -> None:
    out_path = build_character_copyright_profile_pipeline(
        input,
        out,
        min_character_count=min_character_count,
    )
    typer.echo(f"Wrote character copyright profile to {out_path}")


@app.command("evaluate-purity")
def evaluate_purity(
    communities: Path = typer.Option(..., "--communities", help="Community JSON path."),
    profile: Path = typer.Option(..., "--profile", help="Character copyright profile parquet."),
    out: Path = typer.Option(None, "--out", help="Evaluation output directory."),
) -> None:
    out_dir = out or communities.parent.parent / "evaluation"
    out_path = evaluate_community_purity_pipeline(communities, profile, out_dir)
    typer.echo(f"Wrote community purity table to {out_path}")


@app.command("summarize-general")
def summarize_general(
    communities: Path = typer.Option(..., "--communities", help="Community JSON path."),
    edges: Path | None = typer.Option(None, "--edges", help="Scored character-general edge parquet."),
    raw_input: str | None = typer.Option(None, "--raw-input", help="Raw Parquet path, directory, or glob fallback."),
    out: Path = typer.Option(None, "--out", help="Evaluation output directory."),
    top_k: int = typer.Option(20, "--top-k", min=1),
    min_co_count: int = typer.Option(10, "--min-co-count", min=1),
    all_members: bool = typer.Option(False, "--all-members", help="Use all community members instead of core members."),
) -> None:
    out_dir = out or communities.parent.parent / "evaluation"
    if edges is None and raw_input is None:
        raise typer.BadParameter("Provide either --edges or --raw-input.")
    if edges is not None:
        out_path = summarize_community_general_tags_pipeline(
            edges,
            communities,
            out_dir,
            top_k=top_k,
            min_co_count=min_co_count,
            use_core_members=not all_members,
        )
    else:
        out_path = summarize_community_general_tags_from_raw_pipeline(
            raw_input or "",
            communities,
            out_dir,
            top_k=top_k,
            min_co_count=min_co_count,
            use_core_members=not all_members,
        )
    typer.echo(f"Wrote community general-tag summary to {out_path}")


@app.command("export-community-graph")
def export_community_graph(
    edges: Path = typer.Option(..., "--edges", help="Scored character-character edge parquet."),
    communities: Path = typer.Option(..., "--communities", help="Community JSON path."),
    out: Path = typer.Option(..., "--out", help="Output .gexf or .graphml path."),
    community_id: int | None = typer.Option(None, "--community-id", help="Community id to export."),
    tag: str | None = typer.Option(None, "--tag", help="Export the community containing this tag."),
    min_npmi: float = typer.Option(0.15, "--min-npmi", help="Minimum NPMI for exported internal edges."),
    min_co_count: int = typer.Option(15, "--min-co-count", min=1, help="Minimum co-count for exported edges."),
    graph_format: str | None = typer.Option(None, "--format", help="gexf or graphml; inferred from --out by default."),
    drop_isolates: bool = typer.Option(False, "--drop-isolates", help="Do not include members without exported edges."),
) -> None:
    out_path = export_community_graph_pipeline(
        edges,
        communities,
        out,
        community_id=community_id,
        tag=tag,
        min_npmi=min_npmi,
        min_co_count=min_co_count,
        graph_format=graph_format,
        include_isolates=not drop_isolates,
    )
    typer.echo(f"Wrote Gephi graph to {out_path}")


@app.command("recommend-tags")
def recommend_tags(
    tags: str = typer.Option(..., "--tags", help="Comma-separated current prompt tags."),
    target_category: str = typer.Option("character", "--target-category", help="character or general."),
    top_k: int = typer.Option(10, "--top-k", min=1),
    edges: Path = typer.Option(DEFAULT_EDGES_PATH, "--edges", help="Scored character-character edge parquet."),
    communities: Path = typer.Option(DEFAULT_COMMUNITIES_PATH, "--communities", help="Community JSON path."),
    general_summary: Path = typer.Option(
        DEFAULT_GENERAL_SUMMARY_PATH,
        "--general-summary",
        help="Community general summary parquet.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit full JSON instead of a compact table."),
) -> None:
    if target_category not in {"character", "general"}:
        raise typer.BadParameter("--target-category must be either character or general.")
    engine = RecommendationEngine.from_artifacts(edges, communities, general_summary)
    recommendations = engine.recommend(parse_tags(tags), target_category=target_category, top_k=top_k)
    if json_output:
        typer.echo(json.dumps(recommendations, ensure_ascii=False, indent=2))
        return

    for index, item in enumerate(recommendations, start=1):
        community = "" if item["community_id"] is None else f" c={item['community_id']}"
        sources = ",".join(item["source_tags"])
        typer.echo(
            f"{index:>2}. {item['tag']}  score={item['score']:.4f} "
            f"strategy={item['strategy']}{community} sources={sources}"
        )


@app.command("build-gender-profile")
def build_gender_profile(
    input: str = typer.Option(DEFAULT_RAW_INPUT, "--input", help="Raw Danbooru Parquet path, directory, or glob."),
    out: Path = typer.Option(Path("data/processed/evaluation"), "--out", help="Evaluation output directory."),
    min_character_count: int = typer.Option(50, "--min-character-count", min=1),
    min_gender_evidence: int = typer.Option(10, "--min-gender-evidence", min=1),
    female_threshold: float = typer.Option(0.8, "--female-threshold", min=0.5, max=1.0),
) -> None:
    out_path = build_character_gender_profile_pipeline(
        input,
        out,
        min_character_count=min_character_count,
        min_gender_evidence=min_gender_evidence,
        female_threshold=female_threshold,
    )
    typer.echo(f"Wrote character gender profile to {out_path}")


@app.command("query-characters")
def query_characters(
    input: str = typer.Option(DEFAULT_RAW_INPUT, "--input", help="Raw Danbooru Parquet path, directory, or glob."),
    include_general: str = typer.Option(
        ...,
        "--include-general",
        help="Comma-separated general tags, e.g. dark-skinned_male,dark-skinned_female.",
    ),
    mode: str = typer.Option("and", "--mode", help="and or or."),
    top_k: int = typer.Option(50, "--top-k", min=1),
    rank_by: str = typer.Option(
        "co_count",
        "--rank-by",
        help=f"Sort column. One of: {', '.join(sorted(QUERY_RANK_COLUMNS))}.",
    ),
    tag_vocab: Path | None = typer.Option(
        DEFAULT_CHARACTER_VOCAB_PATH,
        "--tag-vocab",
        help="Optional tag vocabulary parquet for global character counts.",
    ),
    gender_profile: Path | None = typer.Option(
        None,
        "--gender-profile",
        help="Optional character gender profile parquet.",
    ),
    female_only: bool = typer.Option(False, "--female-only", help="Keep only empirically female-profiled characters."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a compact table."),
) -> None:
    if mode not in {"and", "or"}:
        raise typer.BadParameter("--mode must be either and or or.")
    if rank_by not in QUERY_RANK_COLUMNS:
        raise typer.BadParameter(f"--rank-by must be one of: {', '.join(sorted(QUERY_RANK_COLUMNS))}.")

    resolved_gender_profile = gender_profile
    if female_only and resolved_gender_profile is None:
        resolved_gender_profile = DEFAULT_GENDER_PROFILE_PATH
    if female_only and (resolved_gender_profile is None or not resolved_gender_profile.exists()):
        raise typer.BadParameter("Use build-gender-profile first, or pass --gender-profile.")

    result = query_characters_by_general_tags(
        input,
        include_general=include_general,
        mode=mode,
        top_k=top_k,
        rank_by=rank_by,
        character_vocab=tag_vocab,
        gender_profile=resolved_gender_profile,
        female_only=female_only,
    )
    if json_output:
        typer.echo(json.dumps(dataframe_to_records(result), ensure_ascii=False, indent=2))
        return

    metric_columns = [
        "co_count",
        "confidence_query_to_character",
        "confidence_character_to_query",
        "lift",
        "ppmi",
    ]
    for index, row in enumerate(result.iter_rows(named=True), start=1):
        metrics = " ".join(
            f"{column}={row[column]:.4f}" if isinstance(row[column], float) else f"{column}={row[column]}"
            for column in metric_columns
            if column in row
        )
        gender = f" gender={row['gender']}" if "gender" in row and row["gender"] is not None else ""
        typer.echo(f"{index:>2}. {row['character']}  {metrics}{gender}")


if __name__ == "__main__":
    app()
