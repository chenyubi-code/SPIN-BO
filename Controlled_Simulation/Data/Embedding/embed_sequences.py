#!/usr/bin/env python3
"""Resumable ESM-2 650M embeddings for the N-TIMP2 simulation library.

The complete supplied sequence is passed to ESM in both roles. Candidate pooling
uses only mature N-TIMP2 residues 4, 35, 38, 68, 71, 97, 99, numbered from 1.
Partner pooling uses every residue and excludes all special/padding tokens.
Importing this module does not perform embedding inference.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_ID = "facebook/esm2_t33_650M_UR50D"
MODEL_NAME = MODEL_ID.split("/")[-1]
LOCAL_MODEL = ROOT / "checkpoints" / MODEL_NAME
POSITIONS = (4, 35, 38, 68, 71, 97, 99)
HIDDEN_SIZE = 1280
ROLES = ("candidate", "partner")
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def atomic_bytes(path: Path, value: bytes) -> None:
    """Publish one complete file; a crash never exposes a half-written index/mask."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_array(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


@contextmanager
def exclusive_output(output: Path) -> Iterator[None]:
    """OS lock releases automatically when a process terminates (macOS/Linux)."""
    with (output / ".embedding.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another embedding process is writing {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_sequences(path: Path) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_id", "sequence_type", "sequence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input must contain {sorted(required)}")
        records = list(reader)
    grouped: dict[str, list[dict]] = {role: [] for role in ROLES}
    seen: set[str] = set()
    for line, record in enumerate(records, start=2):
        identity = record["sequence_id"].strip()
        role = record["sequence_type"].strip()
        sequence = record["sequence"].strip().upper()
        if not identity or identity in seen:
            raise ValueError(f"Missing/duplicate sequence_id at CSV row {line}")
        if role not in ROLES or not AA.fullmatch(sequence):
            raise ValueError(f"Invalid role or noncanonical/blank sequence at CSV row {line}")
        if len(sequence) > 1022:
            raise ValueError(f"{identity}: sequence exceeds the ESM2 context limit")
        if role == "candidate" and len(sequence) != 127:
            raise ValueError(f"{identity}: expected the complete 127-residue N-TIMP2 construct")
        seen.add(identity)
        grouped[role].append({"sequence_id": identity, "sequence_type": role,
                              "sequence": sequence, "embedding_row": len(grouped[role]),
                              "length": len(sequence)})
    if not grouped["candidate"] or not grouped["partner"]:
        raise ValueError("Both candidates and partners are required")
    return grouped


def select_device_dtype(torch: Any, device_name: str, dtype_name: str) -> tuple[Any, Any]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but unavailable")
    if dtype_name == "auto":
        dtype_name = "float16" if device.type in {"cuda", "mps"} else "float32"
    if (device.type == "cpu" and dtype_name == "float16") or (device.type == "mps" and dtype_name == "bfloat16"):
        raise ValueError(f"Use float32 or auto for reliable inference on {device.type}")
    if device.type == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16")
    return device, getattr(torch, dtype_name)


def write_index(path: Path, records: list[dict]) -> None:
    fields = ["embedding_row", "sequence_id", "sequence_type", "sequence", "length"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    content = buffer.getvalue().encode()
    if path.exists():
        with path.open(newline="") as handle:
            previous = list(csv.DictReader(handle))
        if [(row["sequence_id"], row["sequence"]) for row in previous] != [
            (row["sequence_id"], row["sequence"]) for row in records
        ]:
            raise RuntimeError("Existing embedding index uses different sequences; choose a new output directory")
    atomic_bytes(path, content)


def bounded_batches(ids: np.ndarray, records: list[dict], batch_size: int, max_tokens: int) -> Iterator[list[int]]:
    batch: list[int] = []
    longest = 0
    for value in ids:
        index = int(value)
        length = records[index]["length"] + 2
        if length > max_tokens:
            raise ValueError(f"--max-tokens must be at least {length} for {records[index]['sequence_id']}")
        next_longest = max(longest, length)
        if batch and (len(batch) == batch_size or (len(batch) + 1) * next_longest > max_tokens):
            yield batch
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        yield batch


def embed_batch(torch: Any, tokenizer: Any, model: Any, device: Any,
                records: list[dict], role: str) -> np.ndarray:
    sequences = [r["sequence"] for r in records]
    encoded = tokenizer(sequences, add_special_tokens=True, padding=True, truncation=False,
                        return_attention_mask=True, return_special_tokens_mask=True, return_tensors="pt")
    special = encoded.pop("special_tokens_mask").bool()
    residue_mask = encoded["attention_mask"].bool() & ~special
    counts = residue_mask.sum(dim=1).cpu().numpy()
    if not np.array_equal(counts, [len(sequence) for sequence in sequences]):
        raise RuntimeError("Tokenizer did not produce exactly one valid token per residue")
    pooling_mask = residue_mask.clone()
    if role == "candidate":
        # cumsum counts only real residues. BOS/pad/EOS never shift biological numbering.
        residue_numbers = residue_mask.long().cumsum(dim=1)
        pooling_mask.zero_()
        for position in POSITIONS:
            pooling_mask |= residue_mask & (residue_numbers == position)
        if not torch.all(pooling_mask.sum(dim=1) == len(POSITIONS)):
            raise RuntimeError("The seven N-TIMP2 pooling positions did not map to seven residue tokens")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        # FP32 summation even when model inference uses FP16.
        mask = pooling_mask.to(device).unsqueeze(-1)
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
    vectors = pooled.cpu().numpy().astype(np.float32, copy=False)
    if vectors.shape != (len(records), HIDDEN_SIZE) or not np.isfinite(vectors).all():
        raise RuntimeError(f"Invalid/nonfinite embedding batch: {vectors.shape}")
    return vectors


def run(args: argparse.Namespace) -> None:
    records = read_sequences(args.input_csv.resolve())
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    device, dtype = select_device_dtype(torch, args.device, args.dtype)
    settings = {
        "model_id": MODEL_ID, "layer": 33,
        "candidate_positions_1based": list(POSITIONS),
        "partner_pooling": "mean over every valid residue vector",
        "inference_dtype": str(dtype).removeprefix("torch."),
        "pooling_dtype": "float32", "embedding_dtype": "float32",
        "device": str(device), "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_output(output):
        metadata_path = output / "run_metadata.json"
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous.get("settings") != settings:
                raise RuntimeError("Generation settings changed; choose a new output directory")
        metadata = {"settings": settings, "status": "running", "completed": {}}
        atomic_json(metadata_path, metadata)
        for role in ROLES:
            write_index(output / f"{role}_index.csv", records[role])
        print(f"Loading local {MODEL_ID} on {device}, dtype={dtype}", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
            model = AutoModel.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=dtype)
            model.eval().to(device)
            for role in ROLES:
                role_records = records[role]
                shape = (len(role_records), HIDDEN_SIZE)
                final = output / f"{role}_embeddings.npy"
                partial = output / f"{role}_embeddings.partial.npy"
                mask_path = output / f"{role}_completed.npy"
                if final.exists() and partial.exists():
                    raise RuntimeError(f"Both final and partial arrays exist for {role}")
                if final.exists() or partial.exists():
                    array_path = final if final.exists() else partial
                    matrix = np.lib.format.open_memmap(array_path, mode="r+")
                    complete = np.load(mask_path, allow_pickle=False)
                    if matrix.shape != shape or matrix.dtype != np.float32 or complete.shape != (shape[0],) or complete.dtype != bool:
                        raise RuntimeError(f"Embedding/progress dimensions or dtype do not match {role} sequences")
                    if not np.isfinite(matrix[complete]).all():
                        raise RuntimeError(f"Completed rows contain nonfinite values: {role}")
                    if final.exists() and not complete.all():
                        raise RuntimeError(f"Published {role} output has incomplete progress")
                else:
                    complete = np.zeros(shape[0], dtype=bool)
                    atomic_array(mask_path, complete)
                    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
                    array_path = partial
                for batch in bounded_batches(np.flatnonzero(~complete), role_records, args.batch_size, args.max_tokens):
                    vectors = embed_batch(torch, tokenizer, model, device, [role_records[i] for i in batch], role)
                    matrix[batch] = vectors
                    matrix.flush()
                    with array_path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    complete[batch] = True
                    atomic_array(mask_path, complete)
                    metadata["completed"][role] = int(complete.sum())
                    atomic_json(metadata_path, metadata)
                    print(f"{role}: {int(complete.sum())}/{shape[0]}", flush=True)
                if not complete.all() or not np.isfinite(matrix).all():
                    raise RuntimeError(f"Embedding array is incomplete or nonfinite: {role}")
                matrix.flush()
                del matrix
                if partial.exists():
                    os.replace(partial, final)
                metadata["completed"][role] = shape[0]
                atomic_json(metadata_path, metadata)
        except BaseException as exc:
            metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            atomic_json(metadata_path, metadata)
            raise
        metadata["status"] = "complete"
        atomic_json(metadata_path, metadata)
        print(f"Complete: {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT.parent / "Sequence/sequences.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model-path", type=Path, default=LOCAL_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum padded token count per batch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_tokens < 3:
        parser.error("--batch-size must be positive and --max-tokens at least 3")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted. Rerun the identical command to resume completed batches.", file=sys.stderr)
        raise SystemExit(130)
