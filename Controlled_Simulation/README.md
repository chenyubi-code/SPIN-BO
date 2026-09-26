# Controlled simulations

This directory contains the controlled simulations from the paper's main text and appendix: the 20-condition association/smoothness grid, focal comparison, and transfer ablations. Experiments use 7,715 N-TIMP2 variants, MMP-10 as the target, eight source MMPs, and synthetic responses.

`Configs/` stores experiment settings and shared initial designs, `Data/Sequence/` contains the protein sequences, and `simtransfer/` implements landscape generation, models, sampling, and numerical summaries.

Install the repository's [requirements](../requirements.txt). Generate ESM2 features separately: download the public [ESM2-650M checkpoint](https://huggingface.co/facebook/esm2_t33_650M_UR50D) into `Data/Embedding/checkpoints/esm2_t33_650M_UR50D/`, install [embedding dependencies](Data/Embedding/requirements.txt), and run [embed_sequences.py](Data/Embedding/embed_sequences.py) with the supplied [generation settings](Data/Embedding/embedding_settings.json). Model weights and embedding arrays are not included.

From this directory, run:

```bash
python run.py run --focal
python run.py run
python run.py summarize
```

Numerical results are saved in `Outputs/`, including queried candidates, observed responses, discovery trajectories, and aggregated summaries.
