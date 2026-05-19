# Phase 4: Semantic Topology and Community Detection

## Top-50 Semantic Interpretation

The highest `discounted_ppmi` character-character edges show that unsupervised tag
co-occurrence already recovers hard semantic bindings in Danbooru.

1. Inseparable duos and twins: pairs such as `timmy_(animal_crossing)` /
   `tommy_(animal_crossing)` and `mimiko_(jujutsu_kaisen)` /
   `nanako_(jujutsu_kaisen)` have near-symmetric confidence close to 1.0.
   This suggests absolute symbiosis in illustrator practice.
2. Entity/form or entity/companion variants: pairs such as character-form
   variants reveal Danbooru's tag granularity redundancy, useful for query
   expansion and alias-like graph exploration.
3. Asuna/Karin illustrates perception bias versus global statistics:
   `asuna_(blue_archive)` and `karin_(blue_archive)` co-occur 243 times,
   far above the random expectation of about 8.6, but their independent
   popularity lowers confidence and NPMI relative to tightly bound duos.

## Experimental Settings

Input: `data/processed/edges_character_character.parquet`

Planned runs:

| Name | min_npmi | min_co_count | resolution |
| --- | ---: | ---: | ---: |
| baseline | 0.15 | 15 | 1.2 |
| stricter semantic graph | 0.50 | 25 | 1.2 |
| finer semantic graph | 0.60 | 25 | 1.5 |

Outputs are written under `data/processed/communities/` and ignored by git.

## Community Findings

### Run Summary

| Name | min_npmi | min_co_count | resolution | exported communities |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.15 | 15 | 1.2 | 950 |
| stricter semantic graph | 0.50 | 25 | 1.2 | 1,237 |
| finer semantic graph | 0.60 | 25 | 1.5 | 1,745 |

### Blue Archive Reconstruction

In the baseline run, `asuna_(blue_archive)`, `karin_(blue_archive)`,
`neru_(blue_archive)`, `akane_(blue_archive)`, and `toki_(blue_archive)` all map
to community `12`.

- Size: 377 members
- Blue Archive members: 370
- Core members: `sensei_(blue_archive)`, `doodle_sensei_(blue_archive)`,
  `ayane_(blue_archive)`, `serika_(blue_archive)`, `hifumi_(blue_archive)`,
  `ako_(blue_archive)`, `azusa_(blue_archive)`, `shiroko_(blue_archive)`

This is a strong unsupervised recovery of an IP-level ontology: no copyright tag
was used in the community detection step, but the graph topology reconstructs a
nearly pure Blue Archive cluster.

### C&C Substructure Under Stricter Thresholds

In the stricter run, `asuna_(blue_archive)` drops out of exported communities,
but C&C-related tags remain clustered:

- `karin_(blue_archive)`, `neru_(blue_archive)`, and `akane_(blue_archive)` map
  to community `234`
- Size: 12 members
- Core members include `akane_(school_uniform)_(blue_archive)`,
  `akane_(blue_archive)`, `neru_(school_uniform)_(blue_archive)`,
  `karin_(school_uniform)_(blue_archive)`, `toki_(school_uniform)_(blue_archive)`,
  `akane_(bunny)_(blue_archive)`, `neru_(bunny)_(blue_archive)`,
  `asuna_(school_uniform)_(blue_archive)`

This supports the threshold interpretation: relaxed thresholds recover broad
IP-level communities, while stricter thresholds reveal harder semantic bindings
among variants, event outfits, and tightly co-drawn subgroups.

### Example Non-Blue-Archive Communities

The baseline graph also reconstructs recognizable franchise-level clusters:

- Fate community: size 1,266, core members include
  `fujimaru_ritsuka_(male)`, `fujimaru_ritsuka_(female)`,
  `artoria_pendragon_(fate)`, `mash_kyrielight`, `gilgamesh_(fate)`
- Arknights community: size 522, core members include
  `doctor_(arknights)`, `amiya_(arknights)`, `kal'tsit_(arknights)`,
  `ch'en_(arknights)`, `silverash_(arknights)`

These examples suggest that character-character topology alone is sufficient to
recover large parts of the fandom knowledge graph.

## Held-Out Copyright Evaluation

Copyright tags were not used in character-character community detection. They
were held out and used afterward as domain-knowledge labels to quantify whether
the unsupervised graph recovered franchise boundaries.

Purity is defined as:

```text
purity(C_i) = max_d |C_i intersect d| / known_members(C_i)
coverage(C_i) = known_members(C_i) / |C_i|
```

| Run | Mean purity | Median purity | Mean coverage | Purity >= 0.9 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.953 | 1.000 | 1.000 | 87.8% |
| strict | 0.967 | 1.000 | 1.000 | 90.7% |
| fine | 0.968 | 1.000 | 1.000 | 90.2% |

The baseline Blue Archive community `12` has:

- Size: 377
- Known members: 377
- Dominant copyright: `blue_archive`
- Dominant copyright members: 371
- Purity: 0.984

This turns the earlier manual observation into a quantitative claim: the
unsupervised character topology reconstructs a nearly pure Blue Archive IP
boundary without seeing copyright labels during clustering.

## Automated General-Tag Summaries

Attempting to materialize the full `character-general` graph exceeded local
memory during `character,general` ETL, so the explanation stage used a
core-member raw scan fallback. It computes community-level general descriptors
directly from raw Parquet without building the full intermediate graph.

Selected descriptors:

- Baseline Blue Archive community `12`: `problem_solver_68_(blue_archive)`,
  `game_development_department_(blue_archive)`, `tea_party_(blue_archive)`,
  `foreclosure_task_force_(blue_archive)`, `aqua_halo`
- Strict C&C-like community `234`: `cleaning_&_clearing_(blue_archive)`,
  `sukajan`, `aqua_leotard`, `asuna_(blue_archive)_(cosplay)`, `sig_mpx`
- Baseline Arknights community `7`: `penguin_logistics_(arknights)`,
  `star_of_life`, `penguin_logistics_logo`, `hannya_(arknights)`,
  `doctor_(arknights)_(cosplay)`

The descriptors show that communities are not only copyright-pure; their visual
and organizational tags also align with known in-universe factions, outfits,
weapons, and motifs.

## Gephi Visualization Export

Community subgraphs can be exported to GEXF or GraphML for Gephi. The exporter
preserves `discounted_ppmi` as edge `weight`, and adds node-level `degree`,
`weighted_degree`, `gephi_size`, and `is_core` attributes.

Generated examples:

- `data/processed/visualization/blue_archive_community12.gexf`
  - 377 nodes, 5,265 internal edges
  - top weighted-degree nodes: `sensei_(blue_archive)`,
    `doodle_sensei_(blue_archive)`, `ayane_(blue_archive)`,
    `serika_(blue_archive)`, `hifumi_(blue_archive)`
- `data/processed/visualization/arknights_community7.gexf`
  - 522 nodes, 3,340 internal edges
  - top weighted-degree nodes: `doctor_(arknights)`, `amiya_(arknights)`,
    `kal'tsit_(arknights)`, `ch'en_(arknights)`, `silverash_(arknights)`

Recommended Gephi workflow:

1. Import the GEXF file.
2. Run ForceAtlas2 with edge weight influence enabled.
3. Size nodes by `weighted_degree` or `gephi_size`.
4. Size edges by `weight` / `discounted_ppmi`.
5. Label only high-degree or core nodes for a clean cover image.
