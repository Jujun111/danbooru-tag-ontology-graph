from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from danbooru_graph.cli import app
from danbooru_graph.embeddings import (
    TagEmbeddingIndex,
    Item2VecCorpus,
    build_item2vec_embeddings,
    build_svd_embeddings,
    build_symmetric_weight_matrix,
    default_item2vec_embedding_dir,
    default_embedding_dir,
    diagnose_embedding_graph,
    export_neighbor_case_studies,
    label_neighbor_relation,
    neighbor_case_study_records,
    _remove_top_components,
)


def _write_processed_fixture(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    pl.DataFrame(
        {
            "tag_id": [0, 1, 2, 3, 4],
            "category": ["character", "character", "character", "character", "general"],
            "tag": ["a", "b", "c", "d", "blue_sky"],
            "count": [100, 90, 80, 70, 60],
        }
    ).write_parquet(processed / "tag_vocab.parquet")
    pl.DataFrame(
        {
            "post_idx": [0, 0, 0, 1, 2, 2, 3, 4, 4],
            "tag_id": [0, 1, 4, 2, 1, 3, 4, 2, 3],
            "category": [
                "character",
                "character",
                "general",
                "character",
                "character",
                "character",
                "general",
                "character",
                "character",
            ],
        }
    ).write_parquet(processed / "post_tags.parquet")
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
    assert (
        default_embedding_dir(
            processed,
            "character-character",
            "svd",
            2,
            min_npmi=0.15,
            min_co_count=25,
        ).name
        == "character_character_svd_d2_npmi0p15_co25"
    )

    out_dir = build_svd_embeddings(processed, dim=2, seed=0, drop_components=1)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))

    assert out_dir.name == "character_character_svd_d2_drop1"
    assert config["drop_components"] == 1
    assert config["mean_centered"] is True
    assert len(config["dropped_component_singular_values"]) == 1


def test_edge_filtering_updates_config_and_matrix_density(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)

    out_dir = build_svd_embeddings(processed, dim=2, seed=0, min_npmi=0.7, min_co_count=40)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))

    assert out_dir.name == "character_character_svd_d2_npmi0p7_co40"
    assert config["min_npmi"] == pytest.approx(0.7)
    assert config["min_co_count"] == 40
    assert config["source_edge_count"] == 4
    assert config["filtered_edge_count"] == 2
    assert config["matrix_nnz"] == 4


def test_item2vec_corpus_yields_character_sentences_in_tag_order(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)
    corpus = Item2VecCorpus(processed, min_sentence_length=2)

    assert list(corpus) == [["a", "b"], ["b", "d"], ["c", "d"]]


def test_build_item2vec_embeddings_writes_compatible_artifacts(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)

    assert default_item2vec_embedding_dir(processed, "character", 8).name == "character_item2vec_d8"
    out_dir = build_item2vec_embeddings(processed, dim=8, epochs=2, seed=0)

    embeddings = np.load(out_dir / "embeddings.npy")
    vocab = pl.read_parquet(out_dir / "embedding_vocab.parquet")
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    norms = np.linalg.norm(embeddings, axis=1)

    assert out_dir.name == "character_item2vec_d8"
    assert embeddings.shape == (4, 8)
    assert vocab["tag"].to_list() == ["a", "b", "c", "d"]
    assert config["method"] == "item2vec"
    assert config["category"] == "character"
    assert config["sentence_count"] == 3
    assert config["trained_tag_count"] == 4
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_diagnose_embedding_graph_reports_filter_structure(tmp_path) -> None:
    processed = _write_processed_fixture(tmp_path)

    diagnostics = diagnose_embedding_graph(
        processed,
        min_npmi=0.7,
        min_co_count=40,
        target_tags=["a", "c", "missing"],
        top_components=2,
    )

    assert diagnostics["source_edge_count"] == 4
    assert diagnostics["filtered_edge_count"] == 2
    assert diagnostics["matrix_nnz"] == 4
    assert diagnostics["retained_node_count"] == 4
    assert diagnostics["isolated_node_count"] == 0
    assert diagnostics["degree"]["active_min"] == 1
    assert diagnostics["degree"]["active_max"] == 1
    assert diagnostics["component_count"] == 2
    assert diagnostics["largest_component_sizes"] == [2, 2]
    assert diagnostics["target_tags"] == [
        {"tag": "a", "status": "active", "degree": 1, "component_size": 2},
        {"tag": "c", "status": "active", "degree": 1, "component_size": 2},
        {"tag": "missing", "status": "missing", "degree": None, "component_size": None},
    ]
    assert diagnostics["isolated_target_tags"] == ["missing"]


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


def test_neighbor_case_study_labels_and_exports(tmp_path) -> None:
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.9, 0.2],
            [0.0, 1.0],
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
            "tag": [
                "asuna_(blue_archive)",
                "asuna_(bunny)_(blue_archive)",
                "karin_(blue_archive)",
                "amiya_(arknights)",
            ],
            "count": [100, 80, 90, 70],
        }
    ).write_parquet(embeddings_dir / "embedding_vocab.parquet")
    (embeddings_dir / "config.json").write_text(json.dumps({"method": "manual"}), encoding="utf-8")

    assert label_neighbor_relation("asuna_(blue_archive)", "asuna_(bunny)_(blue_archive)")["label"] == "variant"
    assert label_neighbor_relation("asuna_(blue_archive)", "karin_(blue_archive)")["label"] == "same-franchise"
    assert label_neighbor_relation("asuna_(blue_archive)", "amiya_(arknights)")["label"] == "cross-franchise"

    records = neighbor_case_study_records(embeddings_dir, ["asuna_(blue_archive)"], top_k=3)

    assert [record["label"] for record in records] == ["variant", "same-franchise", "cross-franchise"]
    assert records[0]["same_base"] is True
    assert records[1]["same_franchise"] is True

    domain_labels_path = tmp_path / "domain_labels.json"
    domain_labels_path.write_text(
        json.dumps(
            {
                "labels": {
                    "asuna": {"franchise": "blue_archive", "school": "Millennium", "club": "C&C"},
                    "karin": {"franchise": "blue_archive", "school": "Millennium", "club": "C&C"},
                    "amiya": {"franchise": "arknights", "school": "Rhodes Island", "club": "Leadership"},
                    "asuna_(bunny)_(blue_archive)": {"event": "Bunny Chasers on Board"},
                }
            }
        ),
        encoding="utf-8",
    )
    enriched = neighbor_case_study_records(
        embeddings_dir,
        ["asuna_(blue_archive)"],
        top_k=3,
        domain_labels_path=domain_labels_path,
    )

    assert enriched[0]["neighbor_event"] == "Bunny Chasers on Board"
    assert enriched[1]["domain_relation"] == "same-club"
    assert enriched[2]["domain_relation"] == "different-domain"

    csv_path, markdown_path, exported = export_neighbor_case_studies(
        embeddings_dir,
        ["asuna_(blue_archive)"],
        tmp_path / "case_study",
        top_k=3,
        domain_labels_path=domain_labels_path,
    )

    assert exported == enriched
    assert csv_path.exists()
    assert markdown_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "query_tag,rank,neighbor,score,label" in csv_text
    assert "domain_relation" in csv_text
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## `asuna_(blue_archive)`" in markdown
    assert "`asuna_(bunny)_(blue_archive)`" in markdown
    assert "| variant | same-club |" in markdown
    assert "| same-franchise | same-club |" in markdown


def test_similarity_matrix_is_symmetric_and_ordered(tmp_path) -> None:
    embeddings_dir = _write_manual_embedding_dir(tmp_path)
    index = TagEmbeddingIndex.from_dir(embeddings_dir)

    matrix = index.similarity_matrix(["a", "b", "c"])

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] == pytest.approx(index.similarity("a", "b"))
    assert matrix[0, 2] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Unknown tag"):
        index.similarity_matrix(["a", "missing"])


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
            "--min-npmi",
            "0.1",
            "--min-co-count",
            "8",
            "--seed",
            "0",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "embeddings.npy").exists()

    item2vec_out = tmp_path / "item2vec_out"
    result = runner.invoke(
        app,
        [
            "build-embeddings",
            "--processed",
            str(processed),
            "--method",
            "item2vec",
            "--dim",
            "8",
            "--epochs",
            "2",
            "--seed",
            "0",
            "--out",
            str(item2vec_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (item2vec_out / "embeddings.npy").exists()

    result = runner.invoke(
        app,
        [
            "nearest-tags",
            "--embeddings",
            str(item2vec_out),
            "--tag",
            "a",
            "--top-k",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "score=" in result.output

    result = runner.invoke(
        app,
        [
            "evaluate-embeddings",
            "--embeddings",
            str(item2vec_out),
            "--tags",
            "a,b",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "tag\ta\tb" in result.output

    result = runner.invoke(
        app,
        [
            "diagnose-embedding-graph",
            "--processed",
            str(processed),
            "--min-npmi",
            "0.7",
            "--min-co-count",
            "40",
            "--tags",
            "a,c",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "filtered=2" in result.output
    assert "largest=[2, 2]" in result.output
    assert "a: status=active degree=1 component_size=2" in result.output

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
            "export-neighbor-case-studies",
            "--embeddings",
            str(manual_embeddings),
            "--tags",
            "a,b",
            "--out",
            str(tmp_path / "neighbors"),
            "--top-k",
            "2",
            "--domain-labels",
            str(tmp_path / "missing_labels.json"),
        ],
    )
    assert result.exit_code != 0

    domain_labels_path = tmp_path / "cli_domain_labels.json"
    domain_labels_path.write_text(
        json.dumps({"a": {"franchise": "demo", "school": "school", "club": "club"}}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "export-neighbor-case-studies",
            "--embeddings",
            str(manual_embeddings),
            "--tags",
            "a,b",
            "--out",
            str(tmp_path / "neighbors"),
            "--top-k",
            "2",
            "--domain-labels",
            str(domain_labels_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "neighbors.csv").exists()
    assert (tmp_path / "neighbors.md").exists()
    assert "Wrote 4 nearest-neighbor rows" in result.output

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

    result = runner.invoke(
        app,
        [
            "evaluate-embeddings",
            "--embeddings",
            str(manual_embeddings),
            "--tags",
            "a,b,c",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "tag\ta\tb\tc" in result.output
    assert "a\t1.000000" in result.output

    result = runner.invoke(
        app,
        [
            "evaluate-embeddings",
            "--embeddings",
            str(manual_embeddings),
            "--tags",
            "a,b",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"tags": [' in result.output
    assert '"matrix": [' in result.output
