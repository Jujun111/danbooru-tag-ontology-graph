# Project Backlog

## Current research question

Can character-character graph embeddings separate fine-grained Blue Archive motifs, instead of collapsing all same-IP characters together?

## Active

No active task after T003 completion.

## Next

No immediate next task selected.

## Parked

### T004: Add eigsh backend

Reason:
Useful for understanding symmetric matrix factorization, but unlikely to solve the motif-collapse problem before evaluation is standardized.

Status:
parked

## Done / investigated

### T003: Add edge filtering before factorization

Finding:
Edge filtering before SVD is now implemented with `--min-npmi` and `--min-co-count`.
It can break the all-Blue-Archive cosine collapse, but the strict test graph over-fragments and produces degenerate near-identical micro-components.

Artifacts:
- Strict: `data/processed/embeddings/character_character_svd_d128_npmi0p5_co25`
- Mild: `data/processed/embeddings/character_character_svd_d128_npmi0p15_co15`

Strict run details:
- Source edges: 837,806
- Filtered edges: 98,793
- Matrix nnz: 197,586

Selected strict cosine comparison (`min_npmi=0.50`, `min_co_count=25`):

| Pair | Cosine |
| --- | ---: |
| Asuna / Karin | -0.033344 |
| Asuna / Neru | -0.033343 |
| Asuna / Hina | 0.988677 |
| Karin / Neru | 1.000000 |
| Karin / Akane | 1.000000 |
| Neru / Hina | -0.039762 |

Mild run details:
- Source edges: 837,806
- Filtered edges: 346,014
- Matrix nnz: 692,028
- The five-tag Blue Archive probe still collapses to cosine `1.000000` across all pairs.

Interpretation:
Filtering weak edges is necessary but not sufficient. A very mild filter preserves the original IP-dominant geometry, while a strict filter removes too much connective tissue and makes SVD unstable for sparse local neighborhoods. The next embedding improvement should account for filtered-node degree, active-vocabulary coverage, or a threshold sweep before changing solvers.

Status:
done

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
