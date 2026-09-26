# SPIN-BO

Anonymous reproducibility code for **SPIN-BO: Source-Guided Protein–Protein Interaction Discovery for New Targets**, prepared for ICLR 2027.

SPIN-BO combines an adaptively calibrated source-response mean with a target-specific Gaussian-process residual. A frozen protein-pair encoder trained only on source interactions supplies the residual geometry. The repository reproduces the controlled simulations and retrospective discovery experiments in the main text and appendices.

## Repository contents

| Directory | Contents and manuscript correspondence |
| --- | --- |
| `Controlled_Simulation/` | Controlled Simulations and Simulation Details: the 20-condition association/smoothness grid, focal setting, four transfer variants and three external comparators. Includes sequence inputs, generator, encoder, GP/acquisition routines and numerical evaluation. |
| `Retrospective_Discovery/` | Retrospective Discovery and Retrospective bZIP Evaluation: biological multi-source, biological single-source, biology-agnostic 9-source and 5-source campaigns. |
| `Retrospective_Discovery/Data/` | The 20 final, ready-to-use panels, frozen task definitions, source selections and initialization metadata. Raw-data cleaning is not needed. |
| `requirements.txt` | Pinned numerical experiment dependencies. ESM extraction has a separate environment described in each experiment README. |

## Installation

The numerical campaign environment was inspected with Python 3.13.9. Use Python 3.13 for the pinned packages:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Read the experiment-specific instructions before launching campaigns:

- [Controlled simulation commands and model settings](Controlled_Simulation/README.md)
- [Retrospective discovery commands and model settings](Retrospective_Discovery/README.md)
- [Panel inventory, fields and data sources](Retrospective_Discovery/Data/README.md)

## Experiments covered

| Setting | Data / conditions | Repetitions | Assay schedule |
| --- | --- | --- | --- |
| Controlled simulation | 7,715 N-TIMP2 variants; MMP-10 target; 8 source MMPs; 4 association values × 5 smoothness values | 5 landscapes × 2 campaigns; 4 GP variants across the grid, all 7 methods at the focal setting | 24 initial + 24 batches of 16 = 408 |
| Biological multi-source | 10 panels, 54 target problems, 3–9 sources | 10 seeds × 7 methods | 20 initial + 9 batches of 20 = 200 |
| Biological single-source | Same 10 panels; 270 ordered source–target problems | 10 seeds × 7 methods | 20 initial + 9 batches of 20 = 200 |
| Biology-agnostic 9-source | 10 panels; 100 target problems | 5 seeds × 7 methods | 20 initial + 9 batches of 20 = 200 |
| Biology-agnostic 5-source | Same 10 panels and targets; fixed five-source subset for each target | 5 seeds × 7 methods | 20 initial + 9 batches of 20 = 200 |

The seven methods are SPIN-BO (Both), SPIN-BO (Response Transfer), SPIN-BO (Representation Transfer), Target-only BO, ALDE-style, EVOLVEpro-style, and random search. The external comparators are the paper's implemented adaptations. Initial assays are shared across matched methods and are included in the budget. Complete source profiles are available throughout; target outcomes are revealed to the model only after the corresponding assays are selected.

## ESM2 embeddings

Model weights, ESM2 embeddings, trained encoders and generated caches are intentionally absent. Both studies use layer 33 of `esm2_t33_650M_UR50D` (1,280 dimensions per protein). The supplied extraction entrypoints obtain the public checkpoint and generate local caches from the supplied sequences:

- Controlled simulation: mean over the seven designated interface sites for mutant embeddings; mean over all construct residues for partner embeddings.
- Retrospective bZIP: mean over all sequence residues, excluding special/padding tokens.

Use the separate embedding environment and exact pooling, model, dtype and batch settings in the corresponding README. For retrospective features, after setting up its embedding environment, the entrypoint is `python Retrospective_Discovery/generate_embeddings.py --cohort biological`; the `biology_agnostic` cohort uses the two generation sets and environments documented in its README. The original ESM extraction used MPS with float16 model evaluation and float32 pooling. Original encoder devices also differed by cohort. Alternative devices and numerical-library versions can change neural features, subsequent acquisition choices and individual campaign results; a new GPU/CPU run is not promised to be bitwise identical. The original initial designs, seeds and scientific routines are supplied.

Public model sources: [ESM2 checkpoint](https://huggingface.co/facebook/esm2_t33_650M_UR50D), [official ESM repository](https://github.com/facebookresearch/esm).

## Data sources

The bZIP panels contain measurements from the human bZIP bindingPCA deep mutational scan by Bendel and colleagues, using supplementary Table S1: [public dataset](https://zenodo.org/records/16913737), [study DOI](https://doi.org/10.1101/2025.08.21.671354). The data README records filtering, panel membership, sequence fields and the fixed 5-source selections. The published pair-level `logFC` is used directly. Source attribution does not imply a new license for the upstream dataset; its original terms apply.

The simulation uses real human protein sequences and synthetic responses. N-TIMP2 is derived from UniProt P16035; the supplied construct inventory records the target/source MMP accessions and intervals. The response generator independently controls source–target association and smooth response structure in shared ESM-derived coordinates.

Simulation campaigns are averaged within each landscape before taking the mean and standard error across the five landscapes. Retrospective campaigns are averaged over seeds within each problem, problems within each panel, and then equally over the ten panels.
