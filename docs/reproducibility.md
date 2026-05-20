# Reproducibility

## Environment

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Data

This project was run on the Hugging Face dataset
`Shio-Koube/Danbooru-2026-parquet-metadata`.

Download the Parquet shards into:

```text
data/raw/danbooru-2026/
```

The `data/` directory is ignored by git. Raw Danbooru metadata and generated
tables are not committed.

## Character Graph

```powershell
danbooru-graph prepare-vocab `
  --input data/raw/danbooru-2026 `
  --out data/processed `
  --min-tag-count 50 `
  --categories character

danbooru-graph build-edges `
  --processed data/processed `
  --pair character-character `
  --min-pair-count 5

danbooru-graph score-edges `
  --processed data/processed `
  --pair character-character `
  --discount-k 10 `
  --sort-by discounted_ppmi `
  --top-k 50
```

## Communities

```powershell
danbooru-graph detect-communities `
  --processed data/processed `
  --min-npmi 0.15 `
  --min-co-count 15 `
  --resolution 1.2 `
  --out-name character_communities_npmi0.15_co15_res1.2

danbooru-graph detect-communities `
  --processed data/processed `
  --min-npmi 0.50 `
  --min-co-count 25 `
  --resolution 1.2 `
  --out-name character_communities_npmi0.5_co25_res1.2

danbooru-graph detect-communities `
  --processed data/processed `
  --min-npmi 0.60 `
  --min-co-count 25 `
  --resolution 1.5 `
  --out-name character_communities_npmi0.6_co25_res1.5
```

## SVD Embeddings

The first representation-learning baseline factorizes the scored
`character-character` discounted PPMI graph with truncated SVD. It does not
require Gensim, FAISS, or GPU training.

```powershell
danbooru-graph build-embeddings `
  --processed data/processed `
  --pair character-character `
  --method svd `
  --dim 128

danbooru-graph nearest-tags `
  --embeddings data/processed/embeddings/character_character_svd_d128 `
  --tag "asuna_(blue_archive)" `
  --top-k 20

danbooru-graph similarity-tags `
  --embeddings data/processed/embeddings/character_character_svd_d128 `
  --tag-a "asuna_(blue_archive)" `
  --tag-b "karin_(blue_archive)"
```

The `--alpha` parameter controls singular-value scaling. The default
`--alpha 0.5` uses `U * sqrt(S)`. To test whether removing singular-value
magnitude improves motif separation, run:

```powershell
danbooru-graph build-embeddings `
  --processed data/processed `
  --pair character-character `
  --method svd `
  --dim 128 `
  --alpha 0.0

danbooru-graph similarity-tags `
  --embeddings data/processed/embeddings/character_character_svd_d128_a0 `
  --tag-a "asuna_(blue_archive)" `
  --tag-b "karin_(blue_archive)"
```

For All-but-the-top post-processing, remove dominant dense components after
mean-centering:

```powershell
danbooru-graph build-embeddings `
  --processed data/processed `
  --pair character-character `
  --method svd `
  --dim 128 `
  --drop-components 3

danbooru-graph similarity-tags `
  --embeddings data/processed/embeddings/character_character_svd_d128_drop3 `
  --tag-a "asuna_(blue_archive)" `
  --tag-b "karin_(blue_archive)"
```

The generated `embeddings.npy`, `embedding_vocab.parquet`, and `config.json`
stay under ignored `data/processed/`.

## Item2Vec Embeddings

The Item2Vec baseline trains Skip-gram with negative sampling from post-level
character tag sets. It uses `post_tags.parquet` directly instead of the scored
edge table, so it can learn from local image-level contexts rather than only
from the global co-occurrence matrix.

```powershell
danbooru-graph build-embeddings `
  --processed data/processed `
  --method item2vec `
  --categories character `
  --dim 128 `
  --window 50 `
  --negative 10 `
  --sample 1e-4 `
  --epochs 5 `
  --workers 8

danbooru-graph evaluate-embeddings `
  --embeddings data/processed/embeddings/character_item2vec_d128 `
  --tags "asuna_(blue_archive),karin_(blue_archive),neru_(blue_archive),hina_(blue_archive),akane_(blue_archive)"

danbooru-graph nearest-tags `
  --embeddings data/processed/embeddings/character_item2vec_d128 `
  --tag "asuna_(blue_archive)" `
  --top-k 20
```

For exact repeatability during debugging, use `--workers 1`. For full local
training, `--workers 8` is faster but may not be bitwise deterministic.

## Held-Out Evaluation

```powershell
danbooru-graph build-copyright-profile `
  --input data/raw/danbooru-2026 `
  --out data/processed/evaluation `
  --min-character-count 50

danbooru-graph evaluate-purity `
  --communities data/processed/communities/character_communities_npmi0.15_co15_res1.2.json `
  --profile data/processed/evaluation/character_copyright_profile.parquet `
  --out data/processed/evaluation
```

## Community Explanations

The full `character-general` graph may be memory-heavy on local machines. The
recommended reproducible command uses the raw-scan fallback:

```powershell
danbooru-graph summarize-general `
  --communities data/processed/communities/character_communities_npmi0.15_co15_res1.2.json `
  --raw-input data/raw/danbooru-2026 `
  --out data/processed/evaluation `
  --top-k 20 `
  --min-co-count 10
```

## Gephi Export

```powershell
danbooru-graph export-community-graph `
  --edges data/processed/edges_character_character.parquet `
  --communities data/processed/communities/character_communities_npmi0.15_co15_res1.2.json `
  --community-id 12 `
  --min-npmi 0.15 `
  --min-co-count 15 `
  --out data/processed/visualization/blue_archive_community12.gexf
```

Open the GEXF in Gephi, run ForceAtlas2, size nodes by `weighted_degree` or
`gephi_size`, and size edges by `weight`.
