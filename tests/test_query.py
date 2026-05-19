from __future__ import annotations

import polars as pl
from typer.testing import CliRunner

from danbooru_graph.cli import app
from danbooru_graph.query import build_character_gender_profile, query_characters_by_general_tags


def _write_raw_fixture(path) -> None:
    pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "is_deleted": [False, False, False, False, False, False, True],
            "is_banned": [False, False, False, False, False, False, False],
            "tag_string_general": [
                "dark-skinned_male 1girl solo",
                "dark-skinned_male dark-skinned_female 2girls",
                "dark-skinned_male dark-skinned_female",
                "dark-skinned_male 1boy solo",
                "dark_skin 1girl solo",
                "dark-skinned_female 1girl solo",
                "dark-skinned_male dark-skinned_female 1girl solo",
            ],
            "tag_string_character": [
                "girl_a",
                "girl_a girl_b",
                "girl_b guy_a",
                "guy_a",
                "girl_c",
                "girl_b",
                "deleted_girl",
            ],
        }
    ).write_parquet(path)


def test_query_characters_uses_strict_general_tag_and_counts(tmp_path) -> None:
    raw_path = tmp_path / "raw.parquet"
    _write_raw_fixture(raw_path)

    result = query_characters_by_general_tags(
        str(raw_path),
        include_general="dark-skinned_male,dark-skinned_female",
        mode="and",
        top_k=5,
        character_vocab=None,
    )

    assert result["character"].to_list()[:3] == ["girl_b", "girl_a", "guy_a"]
    assert result["co_count"].to_list()[:3] == [2, 1, 1]
    assert result["query_post_count"].to_list()[0] == 2

    exact = query_characters_by_general_tags(
        str(raw_path),
        include_general="dark_skin",
        mode="and",
        top_k=5,
        character_vocab=None,
    )
    assert exact["character"].to_list() == ["girl_c"]


def test_gender_profile_filters_empirical_female_characters(tmp_path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_dir = tmp_path / "evaluation"
    _write_raw_fixture(raw_path)

    profile_path = build_character_gender_profile(
        str(raw_path),
        out_dir,
        min_character_count=1,
        min_gender_evidence=1,
    )
    profile = pl.read_parquet(profile_path)

    assert profile.filter(pl.col("character") == "girl_a").row(0, named=True)["gender"] == "female"
    assert profile.filter(pl.col("character") == "guy_a").row(0, named=True)["gender"] == "male"

    result = query_characters_by_general_tags(
        str(raw_path),
        include_general="dark-skinned_male,dark-skinned_female",
        mode="and",
        top_k=5,
        character_vocab=None,
        gender_profile=profile_path,
        female_only=True,
    )

    assert result["character"].to_list() == ["girl_b", "girl_a"]
    assert set(result["gender"].to_list()) == {"female"}


def test_query_cli_smoke(tmp_path) -> None:
    raw_path = tmp_path / "raw.parquet"
    out_dir = tmp_path / "evaluation"
    _write_raw_fixture(raw_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "build-gender-profile",
            "--input",
            str(raw_path),
            "--out",
            str(out_dir),
            "--min-character-count",
            "1",
            "--min-gender-evidence",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    profile_path = out_dir / "character_gender_profile.parquet"
    assert profile_path.exists()

    result = runner.invoke(
        app,
        [
            "query-characters",
            "--input",
            str(raw_path),
            "--include-general",
            "dark-skinned_male,dark-skinned_female",
            "--gender-profile",
            str(profile_path),
            "--female-only",
            "--top-k",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "girl_b" in result.output
    assert "guy_a" not in result.output
