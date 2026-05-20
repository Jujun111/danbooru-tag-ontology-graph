# Project Backlog

## Current research question

Can character-character graph embeddings separate fine-grained Blue Archive motifs, instead of collapsing all same-IP characters together?

## Active

No active task after full Item2Vec experiment completion.

## Next

No immediate next task selected.

## Parked

### T004: Add eigsh backend

Reason:
Useful for understanding symmetric matrix factorization, but unlikely to solve the motif-collapse problem before evaluation is standardized.

Status:
parked

## Done / investigated

### T006: Run full Item2Vec BA probe

Finding:
Full character-only Item2Vec training successfully breaks the SVD same-IP cosine collapse on the Blue Archive probe.
The model was trained with Skip-gram negative sampling from post-level character tag sets.

Command:
`danbooru-graph build-embeddings --processed data/processed --method item2vec --categories character --dim 128 --window 50 --negative 10 --sample 1e-4 --epochs 5 --workers 8`

Artifact:
`data/processed/embeddings/character_item2vec_d128`

Training summary:
- Sentences: 3,273,233 posts with at least two retained character tags
- Total words: 9,489,198 character tokens
- Trained tags: 28,159
- Vector dimension: 128

Selected cosine comparison:

| Pair | SVD discounted PPMI | Item2Vec |
| --- | ---: | ---: |
| Asuna / Karin | 0.999926 | 0.824511 |
| Asuna / Neru | 0.999855 | 0.801601 |
| Asuna / Hina | 0.999934 | 0.577156 |
| Asuna / Akane | 0.999925 | 0.785102 |
| Karin / Neru | 0.999932 | 0.848246 |
| Karin / Hina | 0.999735 | 0.603196 |
| Karin / Akane | 0.999999 | 0.845002 |
| Neru / Hina | 0.999667 | 0.647144 |
| Neru / Akane | 0.999931 | 0.913994 |
| Hina / Akane | 0.999737 | 0.639465 |

Interpretation:
Item2Vec preserves broad franchise relatedness while recovering internal motif structure. C&C-adjacent members remain high similarity, especially Neru/Akane, while Hina is pushed much farther away. This is the first embedding result that clearly separates fine-grained BA substructure without graph-filter fragmentation.

Status:
done

### T005: Add Item2Vec character embedding baseline

Finding:
Added a character-only Item2Vec backend using Skip-gram negative sampling over post-level character tag sets.
This provides a local-context representation-learning baseline that avoids explicit factorization of the global scored edge matrix.

Acceptance criteria:
- `build-embeddings --method item2vec` trains from `post_tags.parquet`
- output remains compatible with `nearest-tags`, `similarity-tags`, and `evaluate-embeddings`
- corpus iterator skips singleton character posts and preserves tag-id order
- artifact config records training hyperparameters and source paths
- smoke tests pass on a small fixture

Default artifact:
`data/processed/embeddings/character_item2vec_d128`

Recommended BA probe:
`danbooru-graph evaluate-embeddings --embeddings data/processed/embeddings/character_item2vec_d128 --tags "asuna_(blue_archive),karin_(blue_archive),neru_(blue_archive),hina_(blue_archive),akane_(blue_archive)"`

Status:
done

### T003-diagnostics: Add embedding graph diagnostics

Finding:
Added `diagnose-embedding-graph` to inspect the exact filtered graph used before SVD factorization.
The command reports retained edges, retained nodes, isolated target tags, active-degree statistics, and largest connected component sizes.

Acceptance criteria:
- CLI accepts `--min-npmi` and `--min-co-count`
- CLI accepts comma-separated `--tags`
- reports degree statistics and connected component sizes
- core diagnostic logic is tested

Baseline-style filter (`min_npmi=0.15`, `min_co_count=15`):
- Filtered edges: 346,014 from 837,806
- Retained nodes: 24,764 / 28,682
- Isolated nodes: 3,918
- Components: 1,636
- Largest component sizes: `[17416, 99, 95, 75, 65, 55, 51, 51, 49, 48]`
- BA probe: Asuna, Karin, Neru, Hina, and Akane all remain in the giant 17,416-node component
- Target degrees: Asuna 67, Karin 81, Neru 108, Hina 124, Akane 90

Strict filter (`min_npmi=0.50`, `min_co_count=25`):
- Filtered edges: 98,793 from 837,806
- Retained nodes: 22,326 / 28,682
- Isolated nodes: 6,356
- Components: 2,329
- Largest component sizes: `[4850, 1661, 572, 451, 377, 289, 288, 276, 225, 189]`
- BA probe: Asuna and Hina become isolated; Karin, Neru, and Akane fall into a 12-node component
- Target degrees: Asuna 0, Karin 4, Neru 4, Hina 0, Akane 7

Interpretation:
The earlier strict-filter cosine behavior is a graph fragmentation artifact. The mild graph is still dominated by a giant component, while the strict graph isolates high-profile hubs and creates tiny components whose SVD vectors can become degenerate.

Status:
done

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
