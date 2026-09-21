#!/usr/bin/env python3
"""Run single-GPU private E46 inference from verified saved artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag import final_private_p05_e46 as p05  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "prepare", "generate", "finalize"))
    for name in ("project-root", "p00-artifact", "e00-artifact", "e45-artifact",
                 "e46-artifact", "private", "output", "config"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--model-cache", type=Path)
    for name in ("inventory-reviewed", "experiment-plan-reviewed",
                 "private-inference-authorized", "submission-creation-authorized",
                 "e45-selector-reviewed", "e46-adapter-reviewed", "unified-clean-authorized"):
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    approvals = (args.inventory_reviewed, args.experiment_plan_reviewed,
                 args.private_inference_authorized, args.submission_creation_authorized,
                 args.e45_selector_reviewed, args.e46_adapter_reviewed,
                 args.unified_clean_authorized)
    if not all(approvals):
        raise p05.PrivateE46Error("P05 private inference approvals are incomplete.")
    config = p05.load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    shared = {"root": args.project_root, "p00": args.p00_artifact,
              "e00": args.e00_artifact, "e45": args.e45_artifact,
              "e46": args.e46_artifact, "private": args.private,
              "output": args.output, "config": config}
    if args.phase == "preflight":
        result = p05.preflight(**shared)
    elif args.phase == "prepare":
        result = p05.prepare(**shared)
    elif args.phase == "generate":
        if args.model_cache is None:
            raise p05.PrivateE46Error("Generation requires --model-cache.")
        result = p05.generate(**shared, model_cache=args.model_cache)
    else:
        result = p05.finalize(**shared)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
