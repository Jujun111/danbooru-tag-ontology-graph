from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds

from danbooru_graph.pairs import edge_file_stem, parse_pair


EMBEDDING_WEIGHT_COLUMNS = frozenset(
    {
        "discounted_ppmi",
        "ppmi",
        "pmi",
        "npmi",
        "lift",
        "co_count",
    }
)
SVD_SOLVERS = frozenset({"auto", "arpack", "lobpcg", "propack"})
_TAG_QUALIFIER_RE = re.compile(r"\(([^()]*)\)")


def _format_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _format_threshold(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def default_embedding_dir(
    processed_dir: Path,
    pair: str,
    method: str,
    dim: int,
    alpha: float = 0.5,
    drop_components: int = 0,
    min_npmi: float | None = None,
    min_co_count: int | None = None,
) -> Path:
    stem = pair.replace("-", "_")
    suffix_parts = []
    if not math.isclose(alpha, 0.5):
        suffix_parts.append(f"a{_format_alpha(alpha)}")
    if drop_components:
        suffix_parts.append(f"drop{drop_components}")
    if min_npmi is not None:
        suffix_parts.append(f"npmi{_format_threshold(min_npmi)}")
    if min_co_count is not None:
        suffix_parts.append(f"co{min_co_count}")
    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)
    return processed_dir / "embeddings" / f"{stem}_{method}_d{dim}{suffix}"


def default_item2vec_embedding_dir(processed_dir: Path, category: str, dim: int) -> Path:
    return processed_dir / "embeddings" / f"{category}_item2vec_d{dim}"


def _validate_weight_column(weight_column: str) -> None:
    if weight_column not in EMBEDDING_WEIGHT_COLUMNS:
        choices = ", ".join(sorted(EMBEDDING_WEIGHT_COLUMNS))
        raise ValueError(f"Unknown embedding weight {weight_column!r}; expected one of: {choices}.")


def _validate_edge_filters(min_npmi: float | None = None, min_co_count: int | None = None) -> None:
    if min_npmi is not None and not -1.0 <= min_npmi <= 1.0:
        raise ValueError("min_npmi must be between -1.0 and 1.0.")
    if min_co_count is not None and min_co_count < 1:
        raise ValueError("min_co_count must be at least 1.")


def _tag_base(tag: str) -> str:
    return tag.split("_(", 1)[0]


def _tag_qualifiers(tag: str) -> list[str]:
    return _TAG_QUALIFIER_RE.findall(tag)


def _tag_franchise(tag: str, preferred: str | None = None) -> str | None:
    qualifiers = _tag_qualifiers(tag)
    if preferred is not None and preferred in qualifiers:
        return preferred
    return qualifiers[-1] if qualifiers else None


def label_neighbor_relation(query_tag: str, neighbor_tag: str) -> dict[str, Any]:
    """Heuristic report labels for nearest-neighbor case studies."""
    query_base = _tag_base(query_tag)
    query_franchise = _tag_franchise(query_tag)
    neighbor_base = _tag_base(neighbor_tag)
    neighbor_franchise = _tag_franchise(neighbor_tag, preferred=query_franchise)
    same_base = query_base == neighbor_base
    same_franchise = query_franchise is not None and query_franchise == neighbor_franchise

    if query_tag == neighbor_tag:
        label = "self"
    elif same_base and same_franchise:
        label = "variant"
    elif same_base:
        label = "same-base"
    elif same_franchise:
        label = "same-franchise"
    elif query_franchise is not None and neighbor_franchise is not None:
        label = "cross-franchise"
    else:
        label = "unknown"

    return {
        "label": label,
        "query_base": query_base,
        "neighbor_base": neighbor_base,
        "query_franchise": query_franchise,
        "neighbor_franchise": neighbor_franchise,
        "same_base": same_base,
        "same_franchise": same_franchise,
    }


def _embedding_vocab(vocab: pl.DataFrame, category: str) -> pl.DataFrame:
    selected = vocab.filter(pl.col("category") == category).sort("tag_id")
    if selected.is_empty():
        raise ValueError(f"No tags found for category {category!r}.")
    tag_ids = selected["tag_id"].to_list()
    if len(tag_ids) != len(set(tag_ids)):
        raise ValueError("tag_id values must be unique.")
    return selected.with_row_index("embedding_idx")


def _processed_vocab_and_post_tags(processed_dir: Path, category: str) -> tuple[pl.DataFrame, Path, Path]:
    vocab_path = processed_dir / "tag_vocab.parquet"
    post_tags_path = processed_dir / "post_tags.parquet"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing tag vocabulary: {vocab_path}")
    if not post_tags_path.exists():
        raise FileNotFoundError(f"Missing post-tag table: {post_tags_path}")
    return _embedding_vocab(pl.read_parquet(vocab_path), category), vocab_path, post_tags_path


class Item2VecCorpus:
    """Repeatable character-only sentence iterator backed by post_tags.parquet."""

    def __init__(
        self,
        processed_dir: Path,
        category: str = "character",
        min_sentence_length: int = 2,
        batch_size: int = 200_000,
    ) -> None:
        if min_sentence_length < 2:
            raise ValueError("min_sentence_length must be at least 2.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.vocab, _, self.post_tags_path = _processed_vocab_and_post_tags(processed_dir, category)
        self.category = category
        self.min_sentence_length = min_sentence_length
        self.batch_size = batch_size
        self._tag_by_id = {
            int(tag_id): tag
            for tag_id, tag in self.vocab.select("tag_id", "tag").iter_rows()
        }

    def __iter__(self):
        current_post_idx: int | None = None
        current_tags: list[tuple[int, str]] = []

        def maybe_yield_sentence():
            if len(current_tags) >= self.min_sentence_length:
                return [tag for _, tag in sorted(current_tags, key=lambda item: item[0])]
            return None

        parquet_file = pq.ParquetFile(self.post_tags_path)
        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=["post_idx", "tag_id", "category"],
        ):
            post_indices = batch.column(batch.schema.get_field_index("post_idx")).to_pylist()
            tag_ids = batch.column(batch.schema.get_field_index("tag_id")).to_pylist()
            categories = batch.column(batch.schema.get_field_index("category")).to_pylist()
            for post_idx, tag_id, category in zip(post_indices, tag_ids, categories):
                post_idx = int(post_idx)
                if current_post_idx is None:
                    current_post_idx = post_idx
                elif post_idx != current_post_idx:
                    sentence = maybe_yield_sentence()
                    if sentence is not None:
                        yield sentence
                    current_post_idx = post_idx
                    current_tags = []

                if category != self.category:
                    continue
                tag = self._tag_by_id.get(int(tag_id))
                if tag is not None:
                    current_tags.append((int(tag_id), tag))

        sentence = maybe_yield_sentence()
        if sentence is not None:
            yield sentence


def build_symmetric_weight_matrix(
    edges: pl.DataFrame,
    embedding_vocab: pl.DataFrame,
    weight_column: str = "discounted_ppmi",
) -> csr_matrix:
    """Build a symmetric sparse matrix using embedding row indices."""
    _validate_weight_column(weight_column)
    required = {"tag_a_id", "tag_b_id", weight_column}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise ValueError(f"Edge table is missing columns: {missing}.")

    vocab_ids = embedding_vocab["tag_id"].to_numpy()
    num_nodes = embedding_vocab.height
    lookup = np.full(int(vocab_ids.max()) + 1, -1, dtype=np.int64)
    lookup[vocab_ids.astype(np.int64)] = np.arange(num_nodes, dtype=np.int64)

    tag_a_ids = edges["tag_a_id"].to_numpy().astype(np.int64, copy=False)
    tag_b_ids = edges["tag_b_id"].to_numpy().astype(np.int64, copy=False)
    weights = edges[weight_column].to_numpy().astype(np.float64, copy=False)

    in_lookup = (
        (tag_a_ids >= 0)
        & (tag_b_ids >= 0)
        & (tag_a_ids < lookup.shape[0])
        & (tag_b_ids < lookup.shape[0])
    )
    tag_a_ids = tag_a_ids[in_lookup]
    tag_b_ids = tag_b_ids[in_lookup]
    weights = weights[in_lookup]

    rows = lookup[tag_a_ids]
    cols = lookup[tag_b_ids]
    valid = (rows >= 0) & (cols >= 0) & (rows != cols) & np.isfinite(weights) & (weights != 0.0)
    rows = rows[valid]
    cols = cols[valid]
    weights = weights[valid]

    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_weights = np.concatenate([weights, weights])
    matrix = coo_matrix((all_weights, (all_rows, all_cols)), shape=(num_nodes, num_nodes)).tocsr()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix


def _filtered_same_category_edges(
    processed_dir: Path,
    pair: str,
    min_npmi: float | None = None,
    min_co_count: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    left_category, right_category = parse_pair(pair)
    if left_category != right_category:
        raise ValueError("Embedding graph diagnostics require a same-category pair, such as character-character.")
    _validate_edge_filters(min_npmi=min_npmi, min_co_count=min_co_count)

    vocab_path = processed_dir / "tag_vocab.parquet"
    edges_path = processed_dir / f"{edge_file_stem(pair)}.parquet"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing tag vocabulary: {vocab_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing scored edge table: {edges_path}")

    vocab = _embedding_vocab(pl.read_parquet(vocab_path), left_category)
    edges = pl.read_parquet(edges_path)
    if "category_a" in edges.columns and "category_b" in edges.columns:
        edges = edges.filter((pl.col("category_a") == left_category) & (pl.col("category_b") == right_category))
    source_edge_count = edges.height
    if min_npmi is not None:
        edges = edges.filter(pl.col("npmi") >= min_npmi)
    if min_co_count is not None:
        edges = edges.filter(pl.col("co_count") >= min_co_count)
    metadata = {
        "pair": pair,
        "category": left_category,
        "min_npmi": min_npmi,
        "min_co_count": min_co_count,
        "source_edge_count": source_edge_count,
        "filtered_edge_count": edges.height,
        "source_edges": str(edges_path),
        "source_vocab": str(vocab_path),
    }
    return vocab, edges, metadata


def _degree_summary(degrees: np.ndarray) -> dict[str, float | int]:
    active_degrees = degrees[degrees > 0]
    if active_degrees.size == 0:
        return {
            "active_min": 0,
            "active_max": 0,
            "active_mean": 0.0,
            "active_median": 0.0,
            "active_p90": 0.0,
        }
    return {
        "active_min": int(active_degrees.min()),
        "active_max": int(active_degrees.max()),
        "active_mean": float(active_degrees.mean()),
        "active_median": float(np.median(active_degrees)),
        "active_p90": float(np.percentile(active_degrees, 90)),
    }


def diagnose_embedding_graph(
    processed_dir: Path,
    pair: str = "character-character",
    min_npmi: float | None = None,
    min_co_count: int | None = None,
    target_tags: list[str] | None = None,
    weight_column: str = "discounted_ppmi",
    top_components: int = 10,
) -> dict[str, Any]:
    """Summarize the sparse graph that would be factorized by SVD."""
    if top_components < 1:
        raise ValueError("top_components must be at least 1.")
    _validate_weight_column(weight_column)
    vocab, edges, metadata = _filtered_same_category_edges(
        processed_dir,
        pair,
        min_npmi=min_npmi,
        min_co_count=min_co_count,
    )
    matrix = build_symmetric_weight_matrix(edges, vocab, weight_column=weight_column)
    degrees = np.asarray(matrix.getnnz(axis=1)).astype(np.int64, copy=False)
    retained_node_count = int(np.count_nonzero(degrees))
    isolated_node_count = int(vocab.height - retained_node_count)

    active_mask = degrees > 0
    if retained_node_count:
        active_indices = np.flatnonzero(active_mask)
        active_matrix = matrix[active_indices][:, active_indices]
        component_count, labels = connected_components(active_matrix, directed=False, return_labels=True)
        component_sizes = np.bincount(labels).astype(np.int64)
        largest_component_sizes = sorted((int(value) for value in component_sizes), reverse=True)[:top_components]
        component_label_by_embedding_idx = {
            int(embedding_idx): int(label)
            for embedding_idx, label in zip(active_indices, labels)
        }
    else:
        component_count = 0
        component_sizes = np.array([], dtype=np.int64)
        largest_component_sizes = []
        component_label_by_embedding_idx = {}

    vocab_rows = vocab.select("embedding_idx", "tag_id", "tag").to_dicts()
    row_by_tag = {row["tag"]: row for row in vocab_rows}
    target_reports = []
    isolated_targets = []
    for tag in target_tags or []:
        if tag not in row_by_tag:
            target_reports.append({"tag": tag, "status": "missing", "degree": None, "component_size": None})
            isolated_targets.append(tag)
            continue
        row = row_by_tag[tag]
        embedding_idx = int(row["embedding_idx"])
        degree = int(degrees[embedding_idx])
        if degree == 0:
            target_reports.append({"tag": tag, "status": "isolated", "degree": 0, "component_size": 0})
            isolated_targets.append(tag)
            continue
        component_label = component_label_by_embedding_idx[embedding_idx]
        target_reports.append(
            {
                "tag": tag,
                "status": "active",
                "degree": degree,
                "component_size": int(component_sizes[component_label]),
            }
        )

    return {
        **metadata,
        "weight_column": weight_column,
        "num_tags": vocab.height,
        "matrix_nnz": int(matrix.nnz),
        "retained_node_count": retained_node_count,
        "isolated_node_count": isolated_node_count,
        "degree": _degree_summary(degrees),
        "component_count": int(component_count),
        "largest_component_sizes": largest_component_sizes,
        "target_tags": target_reports,
        "isolated_target_tags": isolated_targets,
    }


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    normalized = embeddings.copy()
    normalized[nonzero] = normalized[nonzero] / norms[nonzero]
    return normalized


def _remove_top_components(embeddings: np.ndarray, drop_components: int) -> tuple[np.ndarray, list[float]]:
    """Mean-center embeddings and remove their top principal components."""
    if drop_components <= 0:
        return embeddings, []
    if drop_components >= embeddings.shape[1]:
        raise ValueError("drop_components must be smaller than the embedding dimension.")

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:drop_components]
    corrected = centered - (centered @ components.T) @ components
    return corrected, [float(value) for value in singular_values[:drop_components]]


def _run_svds(matrix: csr_matrix, dim: int, seed: int, solver: str) -> tuple[np.ndarray, np.ndarray, str]:
    if solver not in SVD_SOLVERS:
        choices = ", ".join(sorted(SVD_SOLVERS))
        raise ValueError(f"Unknown SVD solver {solver!r}; expected one of: {choices}.")

    solvers = ["propack", "arpack", "lobpcg"] if solver == "auto" else [solver]
    errors = []
    for candidate in solvers:
        try:
            u, singular_values, _ = svds(
                matrix,
                k=dim,
                which="LM",
                random_state=seed,
                solver=candidate,
            )
            return u, singular_values, candidate
        except Exception as exc:  # pragma: no cover - exercised by SciPy backend differences.
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors)
    raise RuntimeError(f"SVD failed for all requested solvers. {detail}")


def build_item2vec_embeddings(
    processed_dir: Path,
    category: str = "character",
    dim: int = 128,
    window: int = 50,
    negative: int = 10,
    sample: float = 1e-4,
    epochs: int = 5,
    workers: int = 1,
    min_sentence_length: int = 2,
    normalize: bool = True,
    seed: int = 42,
    out_dir: Path | None = None,
) -> Path:
    """Train character-only Item2Vec embeddings from post-level tag sets."""
    if category != "character":
        raise ValueError("Item2Vec v1 supports only category='character'.")
    if dim < 1:
        raise ValueError("dim must be at least 1.")
    if window < 1:
        raise ValueError("window must be at least 1.")
    if negative < 1:
        raise ValueError("negative must be at least 1.")
    if sample < 0:
        raise ValueError("sample must be non-negative.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if min_sentence_length < 2:
        raise ValueError("min_sentence_length must be at least 2.")

    try:
        from gensim.models import Word2Vec
    except ImportError as exc:  # pragma: no cover - depends on installation state.
        raise RuntimeError("Item2Vec requires gensim. Install with: python -m pip install -e .") from exc

    corpus = Item2VecCorpus(
        processed_dir,
        category=category,
        min_sentence_length=min_sentence_length,
    )
    model = Word2Vec(
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        hs=0,
        negative=negative,
        sample=sample,
        ns_exponent=0.75,
        workers=workers,
        seed=seed,
    )
    model.build_vocab(corpus)
    if model.corpus_count == 0:
        raise ValueError("Cannot train Item2Vec: no sentences met min_sentence_length.")
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)

    trained_tags = set(model.wv.key_to_index)
    trained_vocab = (
        corpus.vocab.drop("embedding_idx")
        .filter(pl.col("tag").is_in(trained_tags))
        .sort("tag_id")
        .with_row_index("embedding_idx")
    )
    if trained_vocab.is_empty():
        raise ValueError("Cannot train Item2Vec: no tags were retained in the trained vocabulary.")

    embeddings = np.vstack([model.wv.get_vector(tag) for tag in trained_vocab["tag"].to_list()]).astype(np.float32)
    if normalize:
        embeddings = _normalize_rows(embeddings)

    output_dir = out_dir or default_item2vec_embedding_dir(processed_dir, category, dim)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    trained_vocab.write_parquet(output_dir / "embedding_vocab.parquet")
    config = {
        "method": "item2vec",
        "category": category,
        "dim": dim,
        "window": window,
        "negative": negative,
        "sample": sample,
        "epochs": epochs,
        "workers": workers,
        "min_sentence_length": min_sentence_length,
        "min_count": 1,
        "sg": 1,
        "hs": 0,
        "ns_exponent": 0.75,
        "normalize": normalize,
        "seed": seed,
        "sentence_count": int(model.corpus_count),
        "total_words": int(model.corpus_total_words),
        "trained_tag_count": trained_vocab.height,
        "source_post_tags": str(corpus.post_tags_path),
        "source_vocab": str(processed_dir / "tag_vocab.parquet"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_dir


def build_svd_embeddings(
    processed_dir: Path,
    pair: str = "character-character",
    dim: int = 128,
    weight_column: str = "discounted_ppmi",
    alpha: float = 0.5,
    drop_components: int = 0,
    min_npmi: float | None = None,
    min_co_count: int | None = None,
    normalize: bool = True,
    seed: int = 42,
    solver: str = "auto",
    out_dir: Path | None = None,
) -> Path:
    """Factor a scored same-category edge table into dense tag embeddings."""
    left_category, right_category = parse_pair(pair)
    if left_category != right_category:
        raise ValueError("SVD embedding v1 requires a same-category pair, such as character-character.")
    if dim < 1:
        raise ValueError("dim must be at least 1.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")
    if drop_components < 0:
        raise ValueError("drop_components must be non-negative.")
    if drop_components >= dim:
        raise ValueError("drop_components must be smaller than dim.")
    _validate_edge_filters(min_npmi=min_npmi, min_co_count=min_co_count)
    _validate_weight_column(weight_column)

    vocab, edges, edge_metadata = _filtered_same_category_edges(
        processed_dir,
        pair,
        min_npmi=min_npmi,
        min_co_count=min_co_count,
    )
    if dim >= vocab.height:
        raise ValueError(f"dim must be smaller than the number of tags ({vocab.height}).")
    matrix = build_symmetric_weight_matrix(edges, vocab, weight_column=weight_column)
    if matrix.nnz == 0:
        raise ValueError("Cannot build embeddings from an empty sparse matrix.")

    u, singular_values, actual_solver = _run_svds(matrix, dim, seed, solver)
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    u = u[:, order]
    embeddings = u * np.power(singular_values, alpha)[None, :]
    dropped_component_singular_values: list[float] = []
    if drop_components:
        embeddings, dropped_component_singular_values = _remove_top_components(embeddings, drop_components)
    if normalize:
        embeddings = _normalize_rows(embeddings)

    output_dir = out_dir or default_embedding_dir(
        processed_dir,
        pair,
        "svd",
        dim,
        alpha=alpha,
        drop_components=drop_components,
        min_npmi=min_npmi,
        min_co_count=min_co_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    vocab.write_parquet(output_dir / "embedding_vocab.parquet")
    config = {
        "method": "svd",
        "pair": pair,
        "category": left_category,
        "dim": dim,
        "weight_column": weight_column,
        "alpha": alpha,
        "drop_components": drop_components,
        "min_npmi": min_npmi,
        "min_co_count": min_co_count,
        "source_edge_count": edge_metadata["source_edge_count"],
        "filtered_edge_count": edge_metadata["filtered_edge_count"],
        "mean_centered": bool(drop_components),
        "dropped_component_singular_values": dropped_component_singular_values,
        "normalize": normalize,
        "seed": seed,
        "requested_solver": solver,
        "solver": actual_solver,
        "num_tags": vocab.height,
        "matrix_nnz": int(matrix.nnz),
        "singular_values": [float(value) for value in singular_values],
        "source_edges": edge_metadata["source_edges"],
        "source_vocab": edge_metadata["source_vocab"],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_dir


class TagEmbeddingIndex:
    def __init__(self, embeddings: np.ndarray, vocab: pl.DataFrame, config: dict[str, Any] | None = None) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")
        if embeddings.shape[0] != vocab.height:
            raise ValueError("embedding rows must match embedding_vocab rows.")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.vocab = vocab
        self.config = config or {}
        self._rows_by_tag = {
            tag: int(index)
            for index, tag in vocab.select("embedding_idx", "tag").iter_rows()
        }

    @classmethod
    def from_dir(cls, embeddings_dir: Path) -> "TagEmbeddingIndex":
        embeddings_path = embeddings_dir / "embeddings.npy"
        vocab_path = embeddings_dir / "embedding_vocab.parquet"
        config_path = embeddings_dir / "config.json"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings array: {embeddings_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing embedding vocabulary: {vocab_path}")
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        return cls(np.load(embeddings_path), pl.read_parquet(vocab_path), config=config)

    def _row_for_tag(self, tag: str) -> int:
        if tag not in self._rows_by_tag:
            raise ValueError(f"Unknown tag {tag!r} in embedding vocabulary.")
        return self._rows_by_tag[tag]

    def similarity(self, tag_a: str, tag_b: str) -> float:
        row_a = self._row_for_tag(tag_a)
        row_b = self._row_for_tag(tag_b)
        return float(np.dot(self.embeddings[row_a], self.embeddings[row_b]))

    def similarity_matrix(self, tags: list[str]) -> np.ndarray:
        if not tags:
            raise ValueError("At least one tag is required.")
        rows = [self._row_for_tag(tag) for tag in tags]
        vectors = self.embeddings[rows]
        return vectors @ vectors.T

    def nearest(self, tag: str, top_k: int = 20) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        row = self._row_for_tag(tag)
        vector = self.embeddings[row]
        if not np.isfinite(vector).all() or math.isclose(float(np.linalg.norm(vector)), 0.0):
            raise ValueError(f"Tag {tag!r} has a zero or invalid embedding.")

        scores = self.embeddings @ vector
        rows = self.vocab.select("embedding_idx", "tag_id", "category", "tag").to_dicts()
        candidates = []
        for item in rows:
            embedding_idx = int(item["embedding_idx"])
            if embedding_idx == row:
                continue
            score = float(scores[embedding_idx])
            if not math.isfinite(score):
                continue
            candidates.append(
                {
                    "tag": item["tag"],
                    "score": score,
                    "tag_id": int(item["tag_id"]),
                    "category": item["category"],
                }
            )

        ranked = sorted(candidates, key=lambda item: (-item["score"], item["tag"]))[:top_k]
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked


def neighbor_case_study_records(
    embeddings_dir: Path,
    tags: list[str],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    if not tags:
        raise ValueError("At least one tag is required.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    index = TagEmbeddingIndex.from_dir(embeddings_dir)
    records = []
    for query_tag in tags:
        for item in index.nearest(query_tag, top_k=top_k):
            labels = label_neighbor_relation(query_tag, item["tag"])
            records.append(
                {
                    "query_tag": query_tag,
                    "rank": int(item["rank"]),
                    "neighbor": item["tag"],
                    "score": float(item["score"]),
                    "label": labels["label"],
                    "query_base": labels["query_base"],
                    "neighbor_base": labels["neighbor_base"],
                    "query_franchise": labels["query_franchise"],
                    "neighbor_franchise": labels["neighbor_franchise"],
                    "same_base": labels["same_base"],
                    "same_franchise": labels["same_franchise"],
                }
            )
    return records


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def export_neighbor_case_studies(
    embeddings_dir: Path,
    tags: list[str],
    out_stem: Path,
    top_k: int = 20,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    records = neighbor_case_study_records(embeddings_dir, tags, top_k=top_k)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_stem.with_suffix(".csv")
    markdown_path = out_stem.with_suffix(".md")

    columns = [
        "query_tag",
        "rank",
        "neighbor",
        "score",
        "label",
        "query_base",
        "neighbor_base",
        "query_franchise",
        "neighbor_franchise",
        "same_base",
        "same_franchise",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {column: record[column] for column in columns}
            row["score"] = f"{record['score']:.6f}"
            writer.writerow(row)

    lines = [
        "# Nearest-Neighbor Case Studies",
        "",
        f"Embedding artifact: `{embeddings_dir}`",
        f"Top K: {top_k}",
        "",
    ]
    for query_tag in tags:
        query_records = [record for record in records if record["query_tag"] == query_tag]
        lines.extend(
            [
                f"## `{query_tag}`",
                "",
                "| Rank | Neighbor | Score | Label |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for record in query_records:
            lines.append(
                "| "
                f"{record['rank']} | "
                f"`{_markdown_escape(record['neighbor'])}` | "
                f"{record['score']:.6f} | "
                f"{record['label']} |"
            )
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path, records
