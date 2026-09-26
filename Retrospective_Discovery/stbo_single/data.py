from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "pair_id",
    "landscape_id",
    "mutant_parent",
    "mutation",
    "mutation_pos",
    "wt_aa",
    "mut_aa",
    "mutant_seq",
    "partner",
    "partner_seq",
    "y",
)


@dataclass(frozen=True)
class DatasetBundle:
    filesystem_id: str
    landscape_id: str
    dataset_dir: Path
    csv_path: Path
    target: str
    sources: tuple[str, ...]
    pair_ids: np.ndarray
    candidate_ids: np.ndarray
    mutations: np.ndarray
    mutation_positions: np.ndarray
    wt_amino_acids: np.ndarray
    mutant_amino_acids: np.ndarray
    mutant_sequences: np.ndarray
    target_partner_sequence: str
    source_partner_sequences: tuple[str, ...]
    source_outcomes: np.ndarray
    feasible: np.ndarray

    @property
    def candidate_count(self) -> int:
        return int(self.pair_ids.size)

    def model_metadata(self) -> dict[str, object]:
        return {
            "filesystem_id": self.filesystem_id,
            "landscape_id": self.landscape_id,
            "dataset_id": self.filesystem_id,
            "csv_file": self.csv_path.name,
            "target": self.target,
            "sources": list(self.sources),
            "candidate_count": self.candidate_count,
            "feasible_count": int(self.feasible.sum()),
            "response_direction": "maximize",
        }


@dataclass(frozen=True)
class MeanBasis:
    raw_design: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    raw_from_basis_transform: np.ndarray
    coordinate_system: str
    rank: int
    relative_tolerance: float

    def metadata(self) -> dict[str, object]:
        return {
            "raw_shape": list(self.raw_design.shape),
            "basis_shape": list(self.basis.shape),
            "rank": self.rank,
            "relative_tolerance": self.relative_tolerance,
            "singular_values": self.singular_values.tolist(),
            "raw_from_basis_transform": self.raw_from_basis_transform.tolist(),
            "coordinate_system": self.coordinate_system,
        }


class RetrospectiveOracle:
    """Query-only access to target truth until the complete budget is reached."""

    def __init__(self, truth: np.ndarray, feasible: np.ndarray, required_budget: int):
        values = np.asarray(truth, dtype=np.float64)
        mask = np.asarray(feasible, dtype=bool)
        if values.ndim != 1 or mask.shape != values.shape:
            raise ValueError("truth and feasibility must be aligned one-dimensional arrays")
        if not np.all(np.isfinite(values)):
            raise ValueError("target truth contains nonfinite values")
        if required_budget <= 0 or required_budget > int(mask.sum()):
            raise ValueError("required budget is incompatible with feasible candidates")
        self.__truth = values.copy()
        self.__feasible = mask.copy()
        self.__queried = np.zeros(values.size, dtype=bool)
        self.__required_budget = int(required_budget)

    @property
    def query_count(self) -> int:
        return int(self.__queried.sum())

    def query(self, indices: Iterable[int]) -> np.ndarray:
        requested = np.asarray(list(indices), dtype=int)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("an oracle query must be a nonempty index vector")
        if np.unique(requested).size != requested.size:
            raise ValueError("an oracle query contains duplicate indices")
        if np.any(requested < 0) or np.any(requested >= self.__truth.size):
            raise IndexError("oracle query index is outside the candidate library")
        if np.any(~self.__feasible[requested]):
            raise ValueError("oracle query contains an infeasible candidate")
        if np.any(self.__queried[requested]):
            raise ValueError("oracle query repeats a previously queried candidate")
        if self.query_count + requested.size > self.__required_budget:
            raise ValueError("oracle query would exceed the frozen total budget")
        self.__queried[requested] = True
        return self.__truth[requested].copy()

    def truth_for_evaluation(self) -> np.ndarray:
        if self.query_count != self.__required_budget:
            raise RuntimeError(
                "complete target truth remains sealed until the total budget is exhausted"
            )
        return self.__truth.copy()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _one_value(frame: pd.DataFrame, column: str, context: str) -> str:
    values = frame[column].drop_duplicates().astype(str).tolist()
    _require(len(values) == 1, f"{context} must have exactly one {column}")
    return values[0]


def discover_dataset_csv(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir).expanduser().resolve()
    candidates = sorted(path for path in root.glob("*.csv") if path.is_file())
    if len(candidates) != 1:
        raise ValueError(
            f"{root} must contain exactly one dataset CSV; found {len(candidates)}"
        )
    return candidates[0]


def load_dataset(
    dataset_dir: str | Path,
    target: str,
    *,
    csv_path: str | Path | None = None,
    sources: Iterable[str] | None = None,
) -> DatasetBundle:
    """Load a published panel and preserve the supplied source order."""

    root = Path(dataset_dir).expanduser().resolve()
    resolved_csv = (
        discover_dataset_csv(root)
        if csv_path is None
        else Path(csv_path).expanduser().resolve()
    )
    _require(resolved_csv.is_file(), f"dataset CSV does not exist: {resolved_csv}")
    frame = pd.read_csv(resolved_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    _require(not missing, f"dataset CSV is missing required columns: {missing}")
    _require(len(frame) > 0, "dataset CSV is empty")
    _require(not frame[list(REQUIRED_COLUMNS)].isna().any().any(), "required fields contain NA")

    landscape_id = _one_value(frame, "landscape_id", "dataset")
    partners = sorted(frame["partner"].astype(str).unique().tolist())
    _require(target in partners, f"target {target!r} is not present in the dataset")
    _require(len(partners) >= 2, "single-target transfer requires at least one source")

    frame = frame.copy()
    frame["partner"] = frame["partner"].astype(str)
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame["mutation"] = frame["mutation"].astype(str)
    frame["mutant_seq"] = frame["mutant_seq"].astype(str)
    frame["partner_seq"] = frame["partner_seq"].astype(str)

    _require(frame["pair_id"].is_unique, "pair_id must be globally unique")
    target_rows = frame.loc[frame["partner"].eq(target)].copy()
    target_rows = target_rows.sort_values("pair_id", kind="stable").reset_index(drop=True)
    _require(len(target_rows) > 0, "target candidate library is empty")
    _require(target_rows["mutation"].is_unique, "target mutation IDs are not unique")
    _require(target_rows["mutant_seq"].is_unique, "target mutant sequences are not unique")
    _require(target_rows["pair_id"].is_unique, "target pair IDs are not unique")

    mutations = target_rows["mutation"].to_numpy(dtype=str)
    mutant_sequences = target_rows["mutant_seq"].to_numpy(dtype=str)
    positions = pd.to_numeric(target_rows["mutation_pos"], errors="raise").to_numpy(dtype=int)
    wt = target_rows["wt_aa"].astype(str).to_numpy()
    mut = target_rows["mut_aa"].astype(str).to_numpy()
    available_sources = tuple(partner for partner in partners if partner != target)
    if sources is None:
        selected_sources = available_sources
    else:
        selected_sources = tuple(str(source) for source in sources)
        _require(len(selected_sources) > 0, "source selection must not be empty")
        _require(
            len(set(selected_sources)) == len(selected_sources),
            "source selection contains duplicate partners",
        )
        unknown_sources = sorted(set(selected_sources).difference(available_sources))
        _require(not unknown_sources, f"source selection is invalid: {unknown_sources}")
    sources = selected_sources
    source_outcomes: list[np.ndarray] = []
    source_partner_sequences: list[str] = []
    for source in sources:
        rows = frame.loc[frame["partner"].eq(source)].copy()
        _require(len(rows) == len(target_rows), f"source {source} is incomplete")
        _require(rows["mutation"].is_unique, f"source {source} repeats a mutation")
        rows = rows.set_index("mutation").reindex(mutations)
        _require(not rows.isna().any().any(), f"source {source} cannot align to target mutations")
        _require(
            np.array_equal(rows["mutant_seq"].to_numpy(dtype=str), mutant_sequences),
            f"source {source} mutant sequence alignment failed",
        )
        source_y = pd.to_numeric(rows["y"], errors="raise").to_numpy(dtype=np.float64)
        _require(np.all(np.isfinite(source_y)), f"source {source} y contains NaN or Inf")
        source_outcomes.append(source_y)
        source_partner_sequences.append(_one_value(rows.reset_index(), "partner_seq", source))

    target_partner_sequence = _one_value(target_rows, "partner_seq", target)
    source_matrix = np.vstack(source_outcomes).astype(np.float64, copy=False)
    feasible = np.ones(len(target_rows), dtype=bool)
    candidate_ids = np.asarray(
        [f"{landscape_id}__{mutation}" for mutation in mutations], dtype=str
    )
    return DatasetBundle(
        filesystem_id=root.name,
        landscape_id=landscape_id,
        dataset_dir=root,
        csv_path=resolved_csv,
        target=str(target),
        sources=sources,
        pair_ids=target_rows["pair_id"].to_numpy(dtype=str),
        candidate_ids=candidate_ids,
        mutations=mutations,
        mutation_positions=positions,
        wt_amino_acids=wt,
        mutant_amino_acids=mut,
        mutant_sequences=mutant_sequences,
        target_partner_sequence=target_partner_sequence,
        source_partner_sequences=tuple(source_partner_sequences),
        source_outcomes=source_matrix,
        feasible=feasible,
    )


def make_retrospective_oracle(
    dataset: DatasetBundle, required_budget: int
) -> RetrospectiveOracle:
    """Load target truth into an opaque query-only object for one campaign.

    The truth vector is deliberately not a field on :class:`DatasetBundle`, so
    feature builders, the encoder and surrogate constructors cannot receive it
    accidentally.  This is the sole target-outcome loading boundary.
    """

    frame = pd.read_csv(dataset.csv_path, usecols=["pair_id", "partner", "y"])
    rows = frame.loc[frame["partner"].astype(str).eq(dataset.target)].copy()
    _require(len(rows) == dataset.candidate_count, "target truth row count changed")
    _require(rows["pair_id"].astype(str).is_unique, "target truth pair IDs repeat")
    rows["pair_id"] = rows["pair_id"].astype(str)
    rows = rows.set_index("pair_id").reindex(dataset.pair_ids)
    _require(not rows.isna().any().any(), "target truth cannot align to frozen candidates")
    truth = pd.to_numeric(rows["y"], errors="raise").to_numpy(dtype=np.float64)
    _require(np.all(np.isfinite(truth)), "target truth contains NaN or Inf")
    return RetrospectiveOracle(truth, dataset.feasible, required_budget)


def build_mean_basis(
    source_outcomes: np.ndarray, relative_tolerance: float = 1.0e-10
) -> MeanBasis:
    values = np.asarray(source_outcomes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("source outcomes must have shape (sources, candidates)")
    if not np.all(np.isfinite(values)):
        raise ValueError("source outcomes contain nonfinite values")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative rank tolerance must lie in (0,1)")
    raw = np.column_stack((np.ones(values.shape[1]), values.T))
    left, singular, right_t = np.linalg.svd(raw, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("source mean design has zero rank")
    rank = int(np.sum(singular > singular[0] * relative_tolerance))
    if rank < 1:
        raise ValueError("source mean rank reduction retained no columns")
    if rank == raw.shape[1]:
        # Preserve the historical/raw coefficient coordinates whenever there is
        # no redundancy.  Rank reduction is activated only when required.
        basis = raw.copy()
        transform = np.eye(raw.shape[1], dtype=np.float64)
        coordinate_system = "raw_intercept_plus_source_landscapes"
    else:
        basis = left[:, :rank].copy()
        for column in range(rank):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0.0:
                basis[:, column] *= -1.0
                right_t[column, :] *= -1.0
        transform = singular[:rank, None] * right_t[:rank, :]
        coordinate_system = "left_singular_vector_basis_with_deterministic_sign"
    return MeanBasis(
        raw_design=raw,
        basis=basis,
        singular_values=singular,
        right_singular_vectors=right_t,
        raw_from_basis_transform=transform,
        coordinate_system=coordinate_system,
        rank=rank,
        relative_tolerance=float(relative_tolerance),
    )
