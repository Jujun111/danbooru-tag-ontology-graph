# Project Backlog

## Current research question

Can character-character graph embeddings separate fine-grained Blue Archive motifs, instead of collapsing all same-IP characters together?

## Active

No active task after T001 completion.

## Next

### T003: Add edge filtering before factorization

Depends on:
T001

Goal:
Test whether filtering weak background edges helps recover subgroup structure.

Status:
not started

## Parked

### T004: Add eigsh backend

Reason:
Useful for understanding symmetric matrix factorization, but unlikely to solve the motif-collapse problem before evaluation is standardized.

Status:
parked

## Done / investigated

### T002: Compare NPMI-weighted SVD against discounted PPMI

Finding:
NPMI-weighted SVD slightly reduces same-IP cosine scores, but does not solve Blue Archive motif collapse.

Selected pairwise cosine comparison:

| Pair | discounted PPMI | NPMI |
| --- | ---: | ---: |
| Asuna / Karin | 0.999926 | 0.999564 |
| Asuna / Neru | 0.999855 | 0.999423 |
| Asuna / Hina | 0.999934 | 0.999703 |
| Karin / Akane | 0.999999 | 0.999972 |
| Neru / Hina | 0.999667 | 0.999136 |

Artifact:
`data/processed/embeddings/character_character_svd_d128_npmi`

Status:
done

### T001: Add reusable embedding evaluation command

Finding:
Added `evaluate-embeddings` to compute pairwise cosine matrices from existing embedding artifacts.

Acceptance criteria:
- CLI accepts `--tags`
- outputs pairwise cosine matrix
- supports text, CSV, and JSON
- core cosine-matrix logic is tested

Status:
done

### T000: Add alpha and drop-components post-processing

Finding:
Alpha scaling and All-but-the-top did not meaningfully reduce Asuna/Karin similarity.
