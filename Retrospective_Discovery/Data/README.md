# Retrospective bZIP data

This directory contains the 20 ready-to-use partner panels used in the paper's retrospective discovery experiments. Each panel includes only variants measured across every partner in that panel.

- `biological/`: ten biologically grouped panels shared by the multi-source and single-source experiments.
- `biology_agnostic/`: ten panels shared by the nine-source and five-source experiments.
- `panels.json`: panel membership and candidate counts.
- `five_source_selection.json`: fixed five-source selections for each target.
- `initial_designs.json`: task definitions, campaign seeds and ordered initial candidate indices.
- `embedding_metadata/`: protein sequences and ESM-2 generation settings.

The CSV response `y` is the original pair-level bindingPCA `logFC`. The supplied panels preserve their evaluated measurements, identifiers and sequence order.

Data are derived from Table S1 of Bendel et al. (2025), *The genetic architecture of the human bZIP interaction network*: [dataset on Zenodo](https://zenodo.org/records/16913737), [study DOI](https://doi.org/10.1101/2025.08.21.671354). The upstream data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution remains with the original data creators.

See the [retrospective README](../README.md) for experiment and embedding commands.
