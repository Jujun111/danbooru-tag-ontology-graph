"""Shared constants for Danbooru metadata processing."""

TAG_COLUMNS = {
    "artist": "tag_string_artist",
    "character": "tag_string_character",
    "copyright": "tag_string_copyright",
    "general": "tag_string_general",
    "meta": "tag_string_meta",
}

DEFAULT_PAIRS = (
    "character-character",
    "character-general",
    "copyright-character",
)

DEFAULT_MIN_TAG_COUNT = 50
DEFAULT_MIN_PAIR_COUNT = 5
