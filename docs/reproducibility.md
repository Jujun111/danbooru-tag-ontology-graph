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
