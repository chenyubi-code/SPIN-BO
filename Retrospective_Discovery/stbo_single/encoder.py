from __future__ import annotations

import os
import random
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import StudyConfig
from .data import DatasetBundle
from .io_utils import (
    atomic_save_npy,
    atomic_write_json,
)


class PairEncoder(nn.Module):
    def __init__(self, input_dimension: int, hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, bottleneck, bias=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class _SourcePredictionModel(nn.Module):
    def __init__(self, encoder: PairEncoder, bottleneck: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporary_head = nn.Linear(bottleneck, 1, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.temporary_head(self.encoder(values)).squeeze(-1)


@dataclass(frozen=True)
class EncoderArtifacts:
    prepared_dir: Path
    source_rms: float
    source_latent: np.ndarray
    target_latent: np.ndarray
    manifest: dict[str, Any]


def _resolve_device(requested: str) -> str:
    name = str(requested).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for encoder training but is unavailable")
    if name == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested for encoder training but is unavailable")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("encoder device must be auto, cpu, cuda, or mps")
    return name


def _initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _source_balanced_order(
    source_count: int, candidates_per_source: int, seed: int, epoch: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + int(epoch))
    within_source = [rng.permutation(candidates_per_source) for _ in range(source_count)]
    source_order = rng.permutation(source_count)
    interleaved = np.column_stack(
        [
            int(source) * candidates_per_source + within_source[int(source)]
            for source in source_order
        ]
    ).reshape(-1)
    return interleaved.astype(np.int64, copy=False)


def _pair_inputs(
    mutant_embeddings: np.ndarray,
    partner_embeddings: np.ndarray,
) -> np.ndarray:
    mutants = np.asarray(mutant_embeddings, dtype=np.float32)
    partners = np.asarray(partner_embeddings, dtype=np.float32)
    if mutants.ndim != 2 or partners.ndim != 2 or mutants.shape[1] != partners.shape[1]:
        raise ValueError("mutant and partner embeddings have incompatible dimensions")
    source_count = partners.shape[0]
    candidate_count = mutants.shape[0]
    partner_block = np.repeat(partners[:, None, :], candidate_count, axis=1)
    mutant_block = np.broadcast_to(mutants[None, :, :], (source_count,) + mutants.shape)
    return np.concatenate((partner_block, mutant_block), axis=2).reshape(
        source_count * candidate_count, 2 * mutants.shape[1]
    )


def _encode_in_batches(
    encoder: PairEncoder,
    scaled_values: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(scaled_values), batch_size):
            batch = torch.from_numpy(
                np.asarray(scaled_values[start : start + batch_size], dtype=np.float32)
            ).to(device)
            outputs.append(encoder(batch).cpu().numpy().astype(np.float32))
    return np.vstack(outputs)


def _cache_settings(
    dataset: DatasetBundle,
    mutant_embeddings: np.ndarray,
    config: StudyConfig,
) -> dict[str, Any]:
    return {
        "dataset": dataset.filesystem_id,
        "candidate_count": dataset.candidate_count,
        "target": dataset.target,
        "sources": list(dataset.sources),
        "execution": {
            "device": _resolve_device(config.encoder_device),
            "torch_num_threads": torch.get_num_threads(),
        },
        "encoder_configuration": {
            "architecture": [
                int(2 * mutant_embeddings.shape[1]),
                config.encoder_hidden,
                config.encoder_bottleneck,
            ],
            "activation": "GELU",
            "temporary_head": [config.encoder_bottleneck, 1],
            "initialization": "xavier_uniform_weights_zero_bias",
            "optimizer": "AdamW",
            "learning_rate": config.encoder_learning_rate,
            "weight_decay": config.encoder_weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "batch_size": config.encoder_batch_size,
            "epochs": config.encoder_epochs,
            "gradient_clip": config.encoder_gradient_clip,
            "seed": config.encoder_seed,
            "output_transform": "none_raw_bottleneck",
        },
    }


def prepare_encoder(
    dataset: DatasetBundle,
    mutant_embeddings: np.ndarray,
    source_partner_embeddings: np.ndarray,
    target_partner_embedding: np.ndarray,
    prepared_dir: str | Path,
    config: StudyConfig,
    *,
    force: bool = False,
) -> EncoderArtifacts:
    destination = Path(prepared_dir).expanduser().resolve()
    manifest_path = destination / "manifest.json"
    expected = _cache_settings(
        dataset,
        mutant_embeddings,
        config,
    )
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError("cached encoder manifest is not complete")
        if manifest.get("settings") != expected:
            raise RuntimeError(
                "cached encoder settings differ from current configuration; use a fresh --cache-root to retrain"
            )
        source_latent = np.load(destination / "source_latent.npy", allow_pickle=False)
        target_latent = np.load(destination / "target_latent.npy", allow_pickle=False)
        source_rms = float(manifest["source_only_global_rms"])
        if not np.isfinite(source_rms) or source_rms <= 0.0:
            raise RuntimeError("cached encoder source RMS is invalid")
        if source_latent.shape != (
            len(dataset.sources),
            dataset.candidate_count,
            config.encoder_bottleneck,
        ):
            raise RuntimeError("cached source latent matrix has the wrong shape")
        if target_latent.shape != (dataset.candidate_count, config.encoder_bottleneck):
            raise RuntimeError("cached target latent matrix has the wrong shape")
        if not np.all(np.isfinite(source_latent)) or not np.all(np.isfinite(target_latent)):
            raise RuntimeError("cached encoder latent matrix contains NaN/Inf")
        return EncoderArtifacts(
            destination, source_rms, source_latent, target_latent, manifest
        )
    if destination.exists() and not force:
        raise RuntimeError(
            f"encoder directory exists without a reusable complete manifest: {destination}; "
            "use a fresh --cache-root to retrain"
        )

    mutant_embeddings = np.asarray(mutant_embeddings, dtype=np.float32)
    source_partner_embeddings = np.asarray(source_partner_embeddings, dtype=np.float32)
    target_partner_embedding = np.asarray(target_partner_embedding, dtype=np.float32).reshape(1, -1)
    if source_partner_embeddings.shape != (len(dataset.sources), mutant_embeddings.shape[1]):
        raise ValueError("source partner embeddings do not align with source labels")
    if target_partner_embedding.shape[1] != mutant_embeddings.shape[1]:
        raise ValueError("target partner embedding has the wrong width")
    source_inputs = _pair_inputs(mutant_embeddings, source_partner_embeddings)
    target_inputs = _pair_inputs(mutant_embeddings, target_partner_embedding)
    squared_sum = np.sum(np.square(source_inputs, dtype=np.float64), dtype=np.float64)
    source_rms = float(np.sqrt(squared_sum / source_inputs.size))
    if not np.isfinite(source_rms) or source_rms <= 0.0:
        raise ValueError("source-only input RMS is nonfinite or nonpositive")
    scaled_source = (source_inputs / source_rms).astype(np.float32)
    scaled_target = (target_inputs / source_rms).astype(np.float32)
    labels = dataset.source_outcomes.reshape(-1).astype(np.float32)

    device = _resolve_device(config.encoder_device)
    random.seed(config.encoder_seed)
    np.random.seed(config.encoder_seed % (2**32))
    torch.manual_seed(config.encoder_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.encoder_seed)
    encoder = PairEncoder(
        scaled_source.shape[1], config.encoder_hidden, config.encoder_bottleneck
    )
    training_model = _SourcePredictionModel(encoder, config.encoder_bottleneck)
    training_model.apply(_initialize_linear)
    training_model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=config.encoder_learning_rate,
        weight_decay=config.encoder_weight_decay,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    )
    objective = nn.MSELoss(reduction="mean")
    source_count = len(dataset.sources)
    candidate_count = dataset.candidate_count
    training_model.train()
    for epoch in range(config.encoder_epochs):
        order = _source_balanced_order(
            source_count, candidate_count, config.encoder_seed, epoch
        )
        for start in range(0, len(order), config.encoder_batch_size):
            indices = order[start : start + config.encoder_batch_size]
            x_batch = torch.from_numpy(scaled_source[indices]).to(device)
            y_batch = torch.from_numpy(labels[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = training_model(x_batch)
            loss = objective(predictions, y_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"encoder loss became nonfinite at epoch {epoch + 1}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                training_model.parameters(), config.encoder_gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"encoder gradient became nonfinite at epoch {epoch + 1}")
            optimizer.step()

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    source_latent_flat = _encode_in_batches(
        encoder,
        scaled_source,
        batch_size=config.encoder_batch_size,
        device=device,
    )
    target_latent = _encode_in_batches(
        encoder,
        scaled_target,
        batch_size=config.encoder_batch_size,
        device=device,
    )
    source_latent = source_latent_flat.reshape(
        source_count, candidate_count, config.encoder_bottleneck
    )
    if not np.all(np.isfinite(source_latent)) or not np.all(np.isfinite(target_latent)):
        raise RuntimeError("encoder produced nonfinite bottleneck coordinates")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    stage.mkdir()
    atomic_save_npy(stage / "source_latent.npy", source_latent)
    atomic_save_npy(stage / "target_latent.npy", target_latent)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "settings": expected,
        "source_only_global_rms": source_rms,
        "coordinate_mode": "raw_unwhitened_32d_bottleneck",
        "dtype": "float32",
    }
    atomic_write_json(stage / "manifest.json", manifest)
    previous: Path | None = None
    try:
        if destination.exists():
            previous = stage.with_name(stage.name + ".previous")
            os.replace(destination, previous)
        try:
            os.replace(stage, destination)
        except Exception:
            if previous is not None and previous.exists() and not destination.exists():
                os.replace(previous, destination)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return EncoderArtifacts(destination, source_rms, source_latent, target_latent, manifest)
