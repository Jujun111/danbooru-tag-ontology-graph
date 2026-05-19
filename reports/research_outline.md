# Unsupervised Ontology Reconstruction from Danbooru Meta-tags using Discounted NPMI

## Abstract

This project reconstructs fandom ontologies from Danbooru metadata without using
image pixels or supervised labels. It models posts as transactions, tags as
items, computes sparse co-occurrence graphs, corrects popularity bias with
NPMI and discounted PPMI, and applies Louvain community detection to recover
franchise-level and subgroup-level structures.

## Methodology

- Data: Danbooru 2026 Parquet metadata, text tags only.
- Engineering: Polars lazy ETL, integer tag vocabularies, SciPy CSR matrices,
  and `X.T @ X` sparse co-occurrence multiplication.
- Scoring: Lift, PMI, PPMI, NPMI, directional confidence, and discounted PPMI.
- Topology: weighted character-character graph with Louvain modularity
  optimization.
- Evaluation: copyright labels are held out during clustering and used only
  afterward to compute community purity.

## Experiments

| Experiment | Goal |
| --- | --- |
| Pairwise association | Detect hard semantic bindings such as twins, variants, and inseparable duos. |
| Threshold sweep | Compare broad IP communities against strict micro-clusters. |
| Copyright purity | Quantify whether unsupervised communities reconstruct franchise boundaries. |
| General-tag summaries | Explain clusters through visual and worldbuilding descriptors. |
| Network visualization | Export Gephi-ready GEXF/GraphML subgraphs for cover imagery. |

### Quantitative Results

| Community run | Mean purity | Median purity | Communities with purity >= 0.9 |
| --- | ---: | ---: | ---: |
| baseline (`npmi>=0.15`, `co>=15`, `res=1.2`) | 0.953 | 1.000 | 87.8% |
| strict (`npmi>=0.50`, `co>=25`, `res=1.2`) | 0.967 | 1.000 | 90.7% |
| fine (`npmi>=0.60`, `co>=25`, `res=1.5`) | 0.968 | 1.000 | 90.2% |

The baseline Blue Archive community has 377 members, 371 of which have
`blue_archive` as their held-out dominant copyright label, giving purity 0.984.

Visualization artifacts can be exported as GEXF/GraphML and rendered in Gephi
with ForceAtlas2. Use node `weighted_degree` or `gephi_size` for node size and
edge `weight` / `discounted_ppmi` for edge thickness.

## Case Studies

- Blue Archive baseline community: a nearly pure IP-level cluster recovered
  without copyright supervision.
- C&C strict subcluster: variant and event-driven tags cluster around
  Akane/Karin/Neru/Toki/Asuna variants.
- Automated explanations: C&C is summarized by
  `cleaning_&_clearing_(blue_archive)`, `sukajan`, `aqua_leotard`, and weapon
  tags; Arknights by `penguin_logistics_(arknights)`, `star_of_life`, and
  faction/cosplay descriptors.
- Asuna/Karin: high lift but moderate NPMI demonstrates hub-node penalty and
  core-periphery decoupling.

## Conclusion

Highly redundant UGC tag systems encode a rich ontology in their co-occurrence
structure. With careful sparse computation and robust association scoring, an
unsupervised graph pipeline can recover franchise boundaries, event motifs, and
semantic variants at Danbooru scale.
