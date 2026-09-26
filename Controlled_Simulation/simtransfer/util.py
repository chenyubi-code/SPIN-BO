from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED_VERSION = "sha256-json-uint32-philox-v1"


def seed(master: int, *keys) -> int:
    """Role names are mandatory at call sites; never use Python's salted hash."""
    blob = json.dumps([SEED_VERSION, int(master), *keys], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def rng(master: int, *keys) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(seed(master, *keys)))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value.relative_to(ROOT)) if value.is_relative_to(ROOT) else value.name
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(jsonable(row))
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


