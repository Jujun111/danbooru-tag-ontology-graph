from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from danbooru_graph.cli import app
from danbooru_graph.embeddings import (
    TagEmbeddingIndex,
    build_svd_embeddings,
    build_symmetric_weight_matrix,
    default_embedding_dir,
    _remove_top_components,
)


def _write_processed_fixture(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    pl.DataFrame(
        {
            "tag_id": [0, 1, 2, 3],
            "category": ["character"] * 4,
            "tag": ["a", "b", "c", "d"],
            "count": [100, 90, 80, 70],
        }
    ).write_parquet(processed / "tag_vocab.parquet")
    pl.DataFrame(
        {
            "tag_a_id": [0, 0, 1, 2],
            "tag_b_id": [1, 2, 2, 3],
            "category_a": ["character"] * 4,
            "tag_a": ["a", "a", "b", "c"],
            "category_b": ["character"] * 4,
            "tag_b": ["b", "c", "c", "d"],
            "count_a": [100, 100, 90, 80],
            "count_b": [90, 80, 80, 70],
            "co_count": [50, 10, 8, 40],
            "discounted_ppmi": [4.0, 0.5, 0.3, 3.0],
            "ppmi": [4.4, 0.7, 0.5, 3.5],
            "pmi": [4.4, 0.7, 0.5, 3.5],
            "npmi": [0.8, 0.2, 0.1, 0.7],
            "lift": [20.0, 2.0, 1.5, 15.0],
        }
    ).write_parquet(processed / "edges_character_character.parquet")
    return processed


def _write_manual_embedding_dir(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    pl.DataFrame(
        {
            "embedding_idx": [0, 1, 2, 3],
            "tag_id": [0, 1, 2, 3],
            "category": ["character"] * 4,
            "tag": ["a", "b", "c", "d"],
            "count": [100, 90, 80, 70],
        }
    ).write_parquet(embeddings_dir / "embedding_vocab.parquet")
    (embeddings_dir / "config.json").write_text(json.dumps({"method": "svd"}), encoding="utf-8")
    return embeddings_dir


def test_build_symmetric_weight_matrix() -> None:
    edges = pl.DataFrame(
        {
            "tag_a_id": [0, 1],
            "tag_b_id": [1, 2],
            "discounted_ppmi": [2.0, 3.0],
        }
    )
    vocab = pl.DataFrame(
        {
            "embedding_idx": [0, 1, 2],
            "tag_id": [0, 1, 2],
            "category": ["character"] * 3,
            "tag": ["a", "b", "c"],
            "count": [10, 10, 10],
        }
    )

    matrix = build_symmetric_weight_matrix(edges, vocab)

    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == pytest.approx(2.0)
    assert matrix[1, 0] == pytest.approx(2.0)
    assert matrix[1, 2] == pytest.approx(3.0)
    assert matrix[0, 0] == pytest.approx(0.0)


def test_build_svd_embeddings_writes_artifacts(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)

    out_dir = build_svd_embeddings(processed, dim=2, seed=0)

    embeddings = np.load(out_dir / "embeddings.npy")
    vocab = pl.read_parquet(out_dir / "embedding_vocab.parquet")
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    norms = np.linalg.norm(embeddings, axis=1)

    assert embeddings.shape == (4, 2)
    assert vocab["tag"].to_list() == ["a", "b", "c", "d"]
    assert config["dim"] == 2
    assert config["alpha"] == pytest.approx(0.5)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_alpha_changes_default_output_dir_and_config(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)

    assert default_embedding_dir(processed, "character-character", "svd", 2).name == "character_character_svd_d2"
    assert (
        default_embedding_dir(processed, "character-character", "svd", 2, alpha=0.0).name
        == "character_character_svd_d2_a0"
    )

    out_dir = build_svd_embeddings(processed, dim=2, seed=0, alpha=0.0)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))

    assert out_dir.name == "character_character_svd_d2_a0"
    assert config["alpha"] == pytest.approx(0.0)


def test_remove_top_components_centers_and_reduces_dominant_direction() -> None:
    embeddings = np.array(
        [
            [10.0, 1.0, 0.0],
            [10.0, 0.9, 0.1],
            [10.0, -1.0, 0.0],
            [10.0, -0.9, -0.1],
        ]
    )

    corrected, singular_values = _remove_top_components(embeddings, 1)

    assert len(singular_values) == 1
    assert np.allclose(corrected.mean(axis=0), 0.0, atol=1e-10)
    assert np.linalg.norm(corrected[:, 1]) < np.linalg.norm((embeddings - embeddings.mean(axis=0))[:, 1])


def test_drop_components_changes_default_output_dir_and_config(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)

    assert (
        default_embedding_dir(processed, "character-character", "svd", 2, drop_components=1).name
        == "character_character_svd_d2_drop1"
    )
    assert (
        default_embedding_dir(processed, "character-character", "svd", 2, alpha=0.0, drop_components=1).name
        == "character_character_svd_d2_a0_drop1"
    )

    out_dir = build_svd_embeddings(processed, dim=2, seed=0, drop_components=1)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))

    assert out_dir.name == "character_character_svd_d2_drop1"
    assert config["drop_components"] == 1
    assert config["mean_centered"] is True
    assert len(config["dropped_component_singular_values"]) == 1


def test_nearest_tags_excludes_self_and_missing_tag_errors(tmp_path) -> None:
    embeddings_dir = _write_manual_embedding_dir(tmp_path)
    index = TagEmbeddingIndex.from_dir(embeddings_dir)

    nearest = index.nearest("a", top_k=2)

    assert nearest[0]["tag"] == "b"
    assert nearest[0]["rank"] == 1
    assert all(item["tag"] != "a" for item in nearest)
    assert index.similarity("a", "b") == pytest.approx(0.994937, abs=1e-5)
    with pytest.raises(ValueError, match="Unknown tag"):
        index.nearest("missing")


def test_embedding_cli_smoke(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)
    manual_embeddings = _write_manual_embedding_dir(tmp_path)
    out_dir = tmp_path / "svd_out"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "build-embeddings",
            "--processed",
            str(processed),
            "--dim",
            "2",
            "--alpha",
            "0.0",
            "--drop-components",
            "1",
            "--seed",
            "0",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "embeddings.npy").exists()

    result = runner.invoke(
        app,
        [
            "nearest-tags",
            "--embeddings",
            str(manual_embeddings),
            "--tag",
            "a",
            "--top-k",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"tag": "b"' in result.output
    assert '"rank": 1' in result.output

    result = runner.invoke(
        app,
        [
            "similarity-tags",
            "--embeddings",
            str(manual_embeddings),
            "--tag-a",
            "a",
            "--tag-b",
            "b",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cosine(a, b)" in result.output
