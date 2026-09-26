"""Paper metrics; evaluator truth never enters model fitting or selection."""
from __future__ import annotations
import numpy as np


def top_order(values, ids):
    return np.lexsort((np.asarray(ids, dtype=str), -np.asarray(values)))


def discovery(truth, assay_values, observed, ids, ks=(20,)):
    observed = np.asarray(observed, dtype=int)
    best, worst = float(np.max(truth)), float(np.min(truth))
    span = best - worst
    if span <= 0:
        raise ValueError("Degenerate target truth: normalized regret undefined")
    order = top_order(truth, ids)
    row = {"budget": len(observed), "latent_best": float(np.max(truth[observed])),
           "observed_best": float(np.max(assay_values[observed])),
           "simple_regret": best-float(np.max(truth[observed])),
           "normalized_regret": (best-float(np.max(truth[observed])))/span,
           "top1_success": int(np.any(truth[observed] == best))}
    observed_set = set(observed)
    for k in ks:
        row[f"recall_{k}"] = len(observed_set.intersection(order[:k])) / k
    return row


def terminal_summary(rows, total_budget, intermediate_budgets=(200,)):
    if not rows:
        return {}
    completed = rows[-1]["budget"] == total_budget
    x = np.asarray([r["budget"] for r in rows])
    y = np.asarray([r["normalized_regret"] for r in rows])
    integral = float(np.sum(np.diff(x)*(y[1:]+y[:-1])/2)/(total_budget-x[0])) if len(x) > 1 else None
    out = dict(rows[-1])
    out.update(INR=integral if completed else None, INR_start_budget=int(x[0]), INR_end_budget=int(total_budget))
    for horizon in intermediate_budgets:
        prefix = [r for r in rows if r["budget"] <= horizon]
        reached = bool(prefix) and prefix[-1]["budget"] == horizon
        out[f"recall_20_at_{horizon}"] = prefix[-1]["recall_20"] if reached else None
    return out
