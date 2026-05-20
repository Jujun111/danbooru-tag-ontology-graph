from __future__ import annotations

import csv
import json
from io import StringIO
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
from danbooru_graph.embeddings import (
    EMBEDDING_WEIGHT_COLUMNS,
    SVD_SOLVERS,
    TagEmbeddingIndex,
    build_item2vec_embeddings as build_item2vec_embeddings_pipeline,
    build_svd_embeddings as build_svd_embeddings_pipeline,
    diagnose_embedding_graph as diagnose_embedding_graph_pipeline,
    export_neighbor_case_studies as export_neighbor_case_studies_pipeline,
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


@app.command("build-embeddings")
def build_embeddings(
    processed: Path = typer.Option(..., "--processed", help="Processed directory."),
    pair: str = typer.Option("character-character", "--pair", help="Same-category pair, e.g. character-character."),
    method: str = typer.Option("svd", "--method", help="Embedding method: svd or item2vec."),
    dim: int = typer.Option(128, "--dim", min=1, help="Embedding dimension."),
    categories: str = typer.Option(
        "character",
        "--categories",
        help="Item2Vec training categories. V1 supports only character.",
    ),
    weight_column: str = typer.Option(
        "discounted_ppmi",
        "--weight-column",
        help=f"Scored edge weight. One of: {', '.join(sorted(EMBEDDING_WEIGHT_COLUMNS))}.",
    ),
    alpha: float = typer.Option(
        0.5,
        "--alpha",
        min=0.0,
        max=1.0,
        help="Singular-value exponent. 0.5 uses U * sqrt(S); 0.0 uses U only.",
    ),
    drop_components: int = typer.Option(
        0,
        "--drop-components",
        min=0,
        help="All-but-the-top post-processing: mean-center and remove the top N dense components.",
    ),
    min_npmi: float | None = typer.Option(
        None,
        "--min-npmi",
        min=-1.0,
        max=1.0,
        help="Filter scored edges to NPMI >= this value before factorization.",
    ),
    min_co_count: int | None = typer.Option(
        None,
        "--min-co-count",
        min=1,
        help="Filter scored edges to co_count >= this value before factorization.",
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic SVD seed."),
    solver: str = typer.Option(
        "auto",
        "--solver",
        help=f"SVD solver. One of: {', '.join(sorted(SVD_SOLVERS))}.",
    ),
    window: int = typer.Option(50, "--window", min=1, help="Item2Vec context window."),
    negative: int = typer.Option(10, "--negative", min=1, help="Item2Vec negative samples."),
    sample: float = typer.Option(1e-4, "--sample", min=0.0, help="Item2Vec frequent-token subsampling."),
    epochs: int = typer.Option(5, "--epochs", min=1, help="Item2Vec training epochs."),
    workers: int = typer.Option(1, "--workers", min=1, help="Item2Vec worker threads."),
    min_sentence_length: int = typer.Option(
        2,
        "--min-sentence-length",
        min=2,
        help="Skip Item2Vec post sentences shorter than this.",
    ),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="L2-normalize rows for cosine search."),
    out: Path | None = typer.Option(None, "--out", help="Optional embedding output directory."),
) -> None:
    if method not in {"svd", "item2vec"}:
        raise typer.BadParameter("--method must be either svd or item2vec.")
    if weight_column not in EMBEDDING_WEIGHT_COLUMNS:
        raise typer.BadParameter(f"--weight-column must be one of: {', '.join(sorted(EMBEDDING_WEIGHT_COLUMNS))}.")
    if solver not in SVD_SOLVERS:
        raise typer.BadParameter(f"--solver must be one of: {', '.join(sorted(SVD_SOLVERS))}.")
    if method == "item2vec":
        if categories.strip() != "character":
            raise typer.BadParameter("--categories currently supports only character for item2vec.")
        if pair != "character-character":
            raise typer.BadParameter("--pair is SVD-only and must stay at the default for item2vec.")
        if weight_column != "discounted_ppmi":
            raise typer.BadParameter("--weight-column is SVD-only and must stay at the default for item2vec.")
        if not (alpha == 0.5 and drop_components == 0 and min_npmi is None and min_co_count is None and solver == "auto"):
            raise typer.BadParameter(
                "SVD-only options are not applied to item2vec: "
                "--alpha, --drop-components, --min-npmi, --min-co-count, and --solver."
            )
        out_dir = build_item2vec_embeddings_pipeline(
            processed,
            category="character",
            dim=dim,
            window=window,
            negative=negative,
            sample=sample,
            epochs=epochs,
            workers=workers,
            min_sentence_length=min_sentence_length,
            normalize=normalize,
            seed=seed,
            out_dir=out,
        )
        typer.echo(f"Wrote Item2Vec embeddings to {out_dir}")
        return

    out_dir = build_svd_embeddings_pipeline(
        processed,
        pair=pair,
        dim=dim,
        weight_column=weight_column,
        alpha=alpha,
        drop_components=drop_components,
        min_npmi=min_npmi,
        min_co_count=min_co_count,
        normalize=normalize,
        seed=seed,
        solver=solver,
        out_dir=out,
    )
    typer.echo(f"Wrote SVD embeddings to {out_dir}")


@app.command("diagnose-embedding-graph")
def diagnose_embedding_graph(
    processed: Path = typer.Option(..., "--processed", help="Processed directory."),
    pair: str = typer.Option("character-character", "--pair", help="Same-category pair, e.g. character-character."),
    min_npmi: float | None = typer.Option(
        None,
        "--min-npmi",
        min=-1.0,
        max=1.0,
        help="Filter scored edges to NPMI >= this value before diagnostics.",
    ),
    min_co_count: int | None = typer.Option(
        None,
        "--min-co-count",
        min=1,
        help="Filter scored edges to co_count >= this value before diagnostics.",
    ),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated target tags to inspect."),
    weight_column: str = typer.Option(
        "discounted_ppmi",
        "--weight-column",
        help=f"Scored edge weight. One of: {', '.join(sorted(EMBEDDING_WEIGHT_COLUMNS))}.",
    ),
    top_components: int = typer.Option(10, "--top-components", min=1, help="Largest component sizes to print."),
    json_output: bool = typer.Option(False, "--json", help="Emit full JSON instead of text."),
) -> None:
    if weight_column not in EMBEDDING_WEIGHT_COLUMNS:
        raise typer.BadParameter(f"--weight-column must be one of: {', '.join(sorted(EMBEDDING_WEIGHT_COLUMNS))}.")
    selected_tags = parse_tags(tags or "")
    diagnostics = diagnose_embedding_graph_pipeline(
        processed,
        pair=pair,
        min_npmi=min_npmi,
        min_co_count=min_co_count,
        target_tags=selected_tags,
        weight_column=weight_column,
        top_components=top_components,
    )
    if json_output:
        typer.echo(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        return

    degree = diagnostics["degree"]
    typer.echo(f"pair={diagnostics['pair']} weight={diagnostics['weight_column']}")
    typer.echo(f"filters: min_npmi={diagnostics['min_npmi']} min_co_count={diagnostics['min_co_count']}")
    typer.echo(
        "edges: "
        f"source={diagnostics['source_edge_count']} filtered={diagnostics['filtered_edge_count']} "
        f"matrix_nnz={diagnostics['matrix_nnz']}"
    )
    typer.echo(
        "nodes: "
        f"total={diagnostics['num_tags']} retained={diagnostics['retained_node_count']} "
        f"isolated={diagnostics['isolated_node_count']}"
    )
    typer.echo(
        "degree(active): "
        f"min={degree['active_min']} median={degree['active_median']:.2f} "
        f"mean={degree['active_mean']:.2f} p90={degree['active_p90']:.2f} max={degree['active_max']}"
    )
    typer.echo(
        "components: "
        f"count={diagnostics['component_count']} largest={diagnostics['largest_component_sizes']}"
    )
    if diagnostics["target_tags"]:
        typer.echo("targets:")
        for item in diagnostics["target_tags"]:
            typer.echo(
                f"- {item['tag']}: status={item['status']} "
                f"degree={item['degree']} component_size={item['component_size']}"
            )


@app.command("nearest-tags")
def nearest_tags(
    embeddings: Path = typer.Option(..., "--embeddings", help="Embedding artifact directory."),
    tag: str = typer.Option(..., "--tag", help="Query tag."),
    top_k: int = typer.Option(20, "--top-k", min=1),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a compact table."),
) -> None:
    index = TagEmbeddingIndex.from_dir(embeddings)
    results = index.nearest(tag, top_k=top_k)
    if json_output:
        typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return
    for item in results:
        typer.echo(f"{item['rank']:>2}. {item['tag']}  score={item['score']:.6f} category={item['category']}")


@app.command("export-neighbor-case-studies")
def export_neighbor_case_studies(
    embeddings: Path = typer.Option(..., "--embeddings", help="Embedding artifact directory."),
    tags: str = typer.Option(..., "--tags", help="Comma-separated query tags."),
    out: Path = typer.Option(..., "--out", help="Output stem; writes .csv and .md files."),
    top_k: int = typer.Option(20, "--top-k", min=1),
    domain_labels: Path | None = typer.Option(
        None,
        "--domain-labels",
        help="Optional JSON mapping from tags/base tags to school, club, and event labels.",
    ),
) -> None:
    selected_tags = parse_tags(tags)
    if not selected_tags:
        raise typer.BadParameter("--tags must contain at least one tag.")
    csv_path, markdown_path, records = export_neighbor_case_studies_pipeline(
        embeddings,
        selected_tags,
        out,
        top_k=top_k,
        domain_labels_path=domain_labels,
    )
    typer.echo(f"Wrote {len(records)} nearest-neighbor rows to {csv_path}")
    typer.echo(f"Wrote Markdown case study to {markdown_path}")


@app.command("similarity-tags")
def similarity_tags(
    embeddings: Path = typer.Option(..., "--embeddings", help="Embedding artifact directory."),
    tag_a: str = typer.Option(..., "--tag-a", help="First tag."),
    tag_b: str = typer.Option(..., "--tag-b", help="Second tag."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    index = TagEmbeddingIndex.from_dir(embeddings)
    score = index.similarity(tag_a, tag_b)
    if json_output:
        typer.echo(json.dumps({"tag_a": tag_a, "tag_b": tag_b, "score": score}, ensure_ascii=False, indent=2))
        return
    typer.echo(f"cosine({tag_a}, {tag_b}) = {score:.6f}")


@app.command("evaluate-embeddings")
def evaluate_embeddings(
    embeddings: Path = typer.Option(..., "--embeddings", help="Embedding artifact directory."),
    tags: str = typer.Option(..., "--tags", help="Comma-separated tags to compare."),
    output_format: str = typer.Option("text", "--format", help="text, csv, or json."),
) -> None:
    selected_tags = parse_tags(tags)
    if not selected_tags:
        raise typer.BadParameter("--tags must contain at least one tag.")
    if output_format not in {"text", "csv", "json"}:
        raise typer.BadParameter("--format must be one of: text, csv, json.")

    index = TagEmbeddingIndex.from_dir(embeddings)
    matrix = index.similarity_matrix(selected_tags)

    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "tags": selected_tags,
                    "matrix": matrix.tolist(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if output_format == "csv":
        output = StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["tag", *selected_tags])
        for tag, row in zip(selected_tags, matrix):
            writer.writerow([tag, *(f"{float(value):.6f}" for value in row)])
        typer.echo(output.getvalue().rstrip())
        return

    typer.echo("\t".join(["tag", *selected_tags]))
    for tag, row in zip(selected_tags, matrix):
        values = [f"{float(value):.6f}" for value in row]
        typer.echo("\t".join([tag, *values]))


if __name__ == "__main__":
    app()
