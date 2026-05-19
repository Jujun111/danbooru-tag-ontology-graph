from __future__ import annotations

import polars as pl

from danbooru_graph.etl import build_tag_long


def test_empty_and_null_tags_are_removed() -> None:
    posts = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "tag_string_artist": ["", None, "artist_a"],
            "tag_string_character": ["asuna_(blue_archive) karin_(blue_archive)", "   ", None],
            "tag_string_copyright": ["blue_archive", "", None],
            "tag_string_general": ["1girl  blue_eyes", None, ""],
            "tag_string_meta": ["highres", "", None],
        }
    ).lazy()

    tags = build_tag_long(posts).collect()

    assert "" not in tags["tag"].to_list()
    assert "asuna_(blue_archive)" in tags["tag"].to_list()
    assert "karin_(blue_archive)" in tags["tag"].to_list()
    assert "blue_eyes" in tags["tag"].to_list()
    assert tags.filter(pl.col("tag").is_null()).height == 0
