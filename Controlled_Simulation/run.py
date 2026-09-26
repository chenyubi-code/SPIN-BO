#!/usr/bin/env python3
"""Run or summarize the paper's controlled simulations. No figure/table generation."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from simtransfer.design import FOCAL, GP_METHODS, METHODS, load_config, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "run", "summarize"))
    parser.add_argument("--config", help="Optional JSON file; defaults to Configs/paper.json")
    parser.add_argument("--condition", help="A grid condition ID, e.g. main_r0p6_g0p9")
    parser.add_argument("--focal", action="store_true", help="Select the focal condition and all seven methods")
    parser.add_argument("--landscapes", help="Comma-separated subset of 17001,...,17005")
    parser.add_argument("--campaigns", help="Comma-separated subset of 17101,17102")
    parser.add_argument("--methods", help="Comma-separated method IDs; defaults to the four GP variants, or all seven with --focal")
    parser.add_argument("--embedding-dir", help="Location of generated embeddings; resolved relative to this folder")
    args = parser.parse_args()
    if args.focal and args.condition and args.condition != FOCAL:
        parser.error("--focal conflicts with --condition")
    config = load_config(args.config)
    if args.embedding_dir:
        config["embedding_dir"] = args.embedding_dir
    chosen = select(condition_id=FOCAL if args.focal else args.condition)
    methods = args.methods.split(",") if args.methods else (METHODS if args.focal else GP_METHODS)
    if set(methods)-set(METHODS):
        parser.error("Unknown method ID")
    landscapes = [int(x) for x in args.landscapes.split(",")] if args.landscapes else config["landscape_seeds"]
    campaigns = [int(x) for x in args.campaigns.split(",")] if args.campaigns else config["campaign_seeds"]
    if args.command == "plan":
        print(json.dumps({"conditions": [c["id"] for c in chosen], "methods": methods,
                          "landscape_seeds": landscapes, "campaign_seeds": campaigns,
                          "method_campaigns": len(chosen)*len(methods)*len(landscapes)*len(campaigns),
                          "initial_assays": config["n0"], "batch_size": config["batch_size"],
                          "total_assays": config["budget"]}, indent=2))
        return 0
    if args.command == "summarize":
        from simtransfer.reporting import report
        report(config)
        return 0
    from simtransfer.runner import execute
    statuses = execute(config, chosen, landscapes, campaigns, methods,
                       prepare_only=args.command == "prepare")
    failed = sum(s["state"] == "failed" for s in statuses)
    print(json.dumps({"completed_or_skipped": sum(s["state"] in ("complete", "skipped_complete") for s in statuses),
                      "failed": failed}, indent=2))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
