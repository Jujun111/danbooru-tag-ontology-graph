from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

from danbooru_graph.pairs import edge_file_stem, parse_pair


def _category_matrix(
    processed_dir: Path,
    tag_vocab: pl.DataFrame,
    category: str,
    num_posts: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    vocab_ids = (
        tag_vocab.filter(pl.col("category") == category)
        .select("tag_id")
        .to_numpy()
        .reshape(-1)
        .astype(np.int64)
    )
    if vocab_ids.size == 0:
        return sparse.csr_matrix((num_posts, 0), dtype=np.uint8), vocab_ids

    local_index = {int(tag_id): idx for idx, tag_id in enumerate(vocab_ids)}
    rows_df = (
        pl.scan_parquet(processed_dir / "post_tags.parquet")
        .filter(pl.col("category") == category)
        .select("post_idx", "tag_id")
        .collect()
    )
    if rows_df.height == 0:
        return sparse.csr_matrix((num_posts, vocab_ids.size), dtype=np.uint8), vocab_ids

    rows = rows_df["post_idx"].to_numpy().astype(np.int64)
    tag_ids = rows_df["tag_id"].to_numpy().astype(np.int64)
    cols = np.fromiter((local_index[int(tag_id)] for tag_id in tag_ids), dtype=np.int64)
    data = np.ones(rows.shape[0], dtype=np.uint8)

    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(num_posts, vocab_ids.size),
        dtype=np.uint8,
    ).tocsr()
    return matrix, vocab_ids


def build_edges(processed_dir: Path, pair: str, min_pair_count: int = 5) -> Path:
    """Build sparse co-occurrence edges for one category pair."""
    left_category, right_category = parse_pair(pair)

    num_posts = pl.scan_parquet(processed_dir / "posts.parquet").select(pl.len()).collect().item()
    tag_vocab = pl.read_parquet(processed_dir / "tag_vocab.parquet")

    left_matrix, left_ids = _category_matrix(processed_dir, tag_vocab, left_category, num_posts)
    if left_category == right_category:
        right_matrix, right_ids = left_matrix, left_ids
    else:
        right_matrix, right_ids = _category_matrix(processed_dir, tag_vocab, right_category, num_posts)
    coo = (left_matrix.T @ right_matrix).tocoo()

    if left_category == right_category:
        keep = (coo.row < coo.col) & (coo.data >= min_pair_count)
    else:
        keep = coo.data >= min_pair_count

    rows = coo.row[keep]
    cols = coo.col[keep]
    counts = coo.data[keep].astype(np.int64)

    edges = pl.DataFrame(
        {
            "tag_a_id": left_ids[rows] if rows.size else np.array([], dtype=np.int64),
            "tag_b_id": right_ids[cols] if cols.size else np.array([], dtype=np.int64),
            "co_count": counts,
        },
        schema={"tag_a_id": pl.UInt32, "tag_b_id": pl.UInt32, "co_count": pl.UInt32},
    ).sort(["co_count", "tag_a_id", "tag_b_id"], descending=[True, False, False])

    out_path = processed_dir / f"{edge_file_stem(pair)}_counts.parquet"
    edges.write_parquet(out_path)
    return out_path
