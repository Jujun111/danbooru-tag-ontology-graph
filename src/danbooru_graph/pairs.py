from __future__ import annotations

from danbooru_graph.constants import TAG_COLUMNS


def parse_pair(pair: str) -> tuple[str, str]:
    parts = pair.split("-")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Pair must look like 'character-general', got {pair!r}.")

    left, right = parts
    valid = set(TAG_COLUMNS)
    if left not in valid or right not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"Unknown category in {pair!r}; expected one of: {choices}.")

    return left, right


def edge_file_stem(pair: str) -> str:
    return f"edges_{pair.replace('-', '_')}"
