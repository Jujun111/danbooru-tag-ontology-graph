from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
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


def default_embedding_dir(processed_dir: Path, pair: str, method: str, dim: int) -> Path:
    stem = pair.replace("-", "_")
    return processed_dir / "embeddings" / f"{stem}_{method}_d{dim}"


def _validate_weight_column(weight_column: str) -> None:
    if weight_column not in EMBEDDING_WEIGHT_COLUMNS:
        choices = ", ".join(sorted(EMBEDDING_WEIGHT_COLUMNS))
        raise ValueError(f"Unknown embedding weight {weight_column!r}; expected one of: {choices}.")


def _embedding_vocab(vocab: pl.DataFrame, category: str) -> pl.DataFrame:
    selected = vocab.filter(pl.col("category") == category).sort("tag_id")
    if selected.is_empty():
        raise ValueError(f"No tags found for category {category!r}.")
    tag_ids = selected["tag_id"].to_list()
    if len(tag_ids) != len(set(tag_ids)):
        raise ValueError("tag_id values must be unique.")
    return selected.with_row_index("embedding_idx")


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


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    normalized = embeddings.copy()
    normalized[nonzero] = normalized[nonzero] / norms[nonzero]
    return normalized


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


def build_svd_embeddings(
    processed_dir: Path,
    pair: str = "character-character",
    dim: int = 128,
    weight_column: str = "discounted_ppmi",
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
    _validate_weight_column(weight_column)

    vocab_path = processed_dir / "tag_vocab.parquet"
    edges_path = processed_dir / f"{edge_file_stem(pair)}.parquet"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing tag vocabulary: {vocab_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing scored edge table: {edges_path}")

    vocab = _embedding_vocab(pl.read_parquet(vocab_path), left_category)
    if dim >= vocab.height:
        raise ValueError(f"dim must be smaller than the number of tags ({vocab.height}).")

    edges = pl.read_parquet(edges_path)
    if "category_a" in edges.columns and "category_b" in edges.columns:
        edges = edges.filter((pl.col("category_a") == left_category) & (pl.col("category_b") == right_category))
    matrix = build_symmetric_weight_matrix(edges, vocab, weight_column=weight_column)
    if matrix.nnz == 0:
        raise ValueError("Cannot build embeddings from an empty sparse matrix.")

    u, singular_values, actual_solver = _run_svds(matrix, dim, seed, solver)
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    u = u[:, order]
    embeddings = u * np.sqrt(singular_values)[None, :]
    if normalize:
        embeddings = _normalize_rows(embeddings)

    output_dir = out_dir or default_embedding_dir(processed_dir, pair, "svd", dim)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    vocab.write_parquet(output_dir / "embedding_vocab.parquet")
    config = {
        "method": "svd",
        "pair": pair,
        "category": left_category,
        "dim": dim,
        "weight_column": weight_column,
        "normalize": normalize,
        "seed": seed,
        "requested_solver": solver,
        "solver": actual_solver,
        "num_tags": vocab.height,
        "matrix_nnz": int(matrix.nnz),
        "singular_values": [float(value) for value in singular_values],
        "source_edges": str(edges_path),
        "source_vocab": str(vocab_path),
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
