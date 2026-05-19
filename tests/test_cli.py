from __future__ import annotations

import polars as pl
from typer.testing import CliRunner

from danbooru_graph.cli import app


def test_cli_pipeline(tmp_path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_dir = tmp_path / "processed"
    pl.DataFrame(
        {
            "id": [1, 2],
            "rating": ["g", "g"],
            "is_deleted": [False, False],
            "is_banned": [False, False],
            "tag_string_artist": ["", ""],
            "tag_string_character": ["asuna_(blue_archive)", "asuna_(blue_archive)"],
            "tag_string_copyright": ["blue_archive", "blue_archive"],
            "tag_string_general": ["blue_eyes smile", "blue_eyes"],
            "tag_string_meta": ["", ""],
        }
    ).write_parquet(raw_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "prepare-vocab",
            "--input",
            str(raw_path),
            "--out",
            str(out_dir),
            "--min-tag-count",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "build-edges",
            "--processed",
            str(out_dir),
            "--pair",
            "character-general",
            "--min-pair-count",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "score-edges",
            "--processed",
            str(out_dir),
            "--pair",
            "character-general",
            "--discount-k",
            "10",
            "--sort-by",
            "discounted_ppmi",
            "--top-k",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "edges_character_general.parquet").exists()
    top_path = out_dir / "top_character_general_discounted_ppmi_50.csv"
    assert top_path.exists()
    top = pl.read_csv(top_path)
    assert "discounted_ppmi" in top.columns
    assert top.height > 0
