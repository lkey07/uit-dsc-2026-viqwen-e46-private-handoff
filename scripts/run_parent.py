"""Run P06: the separate E46/top12-v2 + parent one-GPU candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
CODE = ROOT / "code-v95/uit-dsc-fixed-rag-code"
SCRIPT = CODE / "scripts/run_final_private_p06_e46_parent.py"
CONFIG = CODE / "configs/final-private-p06-e46-top12v2-parent-max1536-unified-v1.json"
INPUTS = ROOT / "inputs"
OUTPUT = ROOT / "output/final-private-p06-e46-top12v2-parent-max1536-unified-v1"
FLAGS = [
    "--inventory-reviewed", "--experiment-plan-reviewed",
    "--private-inference-authorized", "--submission-creation-authorized",
    "--e45-selector-reviewed", "--e46-adapter-reviewed",
    "--parent-variant-authorized", "--unified-clean-authorized",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path,
                        help="Pinned base-model snapshot; downloads exact revision if omitted")
    parser.add_argument("--gpu", default="0", help="Physical CUDA GPU to expose (default: 0)")
    args = parser.parse_args()
    for path in (SCRIPT, CONFIG, ROOT / "private-official.json"):
        if not path.is_file():
            raise RuntimeError(f"Missing handoff file: {path}")
    source = {
        "p00": INPUTS / MANIFEST["archive_roots"]["assets/P00_dataset.zip"],
        "e00": INPUTS / MANIFEST["archive_roots"]["assets/E00-v2-dataset.zip"],
        "e45": INPUTS / MANIFEST["archive_roots"]["assets/E45_dataset.zip"],
        "e46": INPUTS / MANIFEST["archive_roots"]["assets/E46_dataset.zip"],
    }
    if not all(path.is_dir() for path in source.values()):
        raise RuntimeError("Run scripts/prepare_assets.py first")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    base = [
        "--project-root", str(CODE), "--p00-artifact", str(source["p00"]),
        "--e00-artifact", str(source["e00"]), "--e45-artifact", str(source["e45"]),
        "--e46-artifact", str(source["e46"]),
        "--private", str(ROOT / "private-official.json"),
        "--output", str(OUTPUT), "--config", str(CONFIG), *FLAGS,
    ]
    for phase in ("preflight", "prepare"):
        subprocess.run([sys.executable, "-u", str(SCRIPT), phase, *base],
                       check=True, env=environment)
    if args.model_cache is None:
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(repo_id=MANIFEST["base_model"]["id"],
                                       revision=MANIFEST["base_model"]["revision"]))
    else:
        model = args.model_cache
    if model.resolve().name != MANIFEST["base_model"]["revision"]:
        raise RuntimeError("Base-model snapshot revision differs from E46")
    subprocess.run([sys.executable, "-u", str(SCRIPT), "generate", *base,
                    "--model-cache", str(model)], check=True, env=environment)
    subprocess.run([sys.executable, "-u", str(SCRIPT), "finalize", *base],
                   check=True, env=environment)
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": "PARENT VARIANT READY FOR MANUAL REVIEW",
                      "questions": report["sample_size"],
                      "submission_zip": str(OUTPUT / "submission.zip"),
                      "parent_expanded": report["diagnostics"]["questions_with_parent_expansion"],
                      "seed_budget_fallbacks": report["diagnostics"]["seed_budget_fallback_questions"],
                      "eos": report["diagnostics"]["eos_questions"],
                      "length": report["diagnostics"]["length_questions"],
                      "cleaned": report["diagnostics"]["changed_questions"]}, indent=2))


if __name__ == "__main__":
    main()
