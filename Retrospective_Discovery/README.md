# Retrospective discovery

This directory contains the paper's retrospective bZIP discovery experiments: biological multi-source (`biological_multi`), biological single-source (`biological_single`), and biology-agnostic nine-source (`random_9`) and five-source (`random_5`) settings. Each campaign starts with 20 shared assays and reaches a total budget of 200.

[Data/](Data/README.md) supplies the prepared panels and fixed initial designs. `configs/` stores experiment settings; `stbo_single/` and `stbo_benchmark/` implement SPIN-BO, its ablations, and the comparison methods.

Install the repository's [requirements](../requirements.txt). Generate ESM2 features locally with `generate_embeddings.py`, which downloads the public [ESM2-650M checkpoint](https://huggingface.co/facebook/esm2_t33_650M_UR50D). Use a separate environment with the package versions and generation sets recorded in [specification.json](Data/embedding_metadata/specification.json); run `python generate_embeddings.py --help` for options. Weights and embeddings are not included.

From this directory, run and summarize a cohort:

```bash
python run.py --cohort biological_multi
python aggregate.py --cohort biological_multi
```

Replace the cohort name to run the other settings. `run.py --help` lists task, method, and seed options. Query sequences and discovery metrics are saved in `Results/`, with aggregated results in `Results/summary/`.
