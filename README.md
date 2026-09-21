# E46 Vi-Qwen private handoff (one GPU)

This is a **new, independent public repository** for the owner's E46 private
inference handoff. It is not the older E44 code-only repository. The owner
explicitly chose to publish the 1,918 private questions and their saved P00
retrieval output here. Public clones, forks and cached copies cannot be
recalled by deleting this repository later. Do not add API keys or `.env`.

## What's included

| Path | Purpose |
| --- | --- |
| `code-v95/uit-dsc-fixed-rag-code/` | Complete pinned P05 source, config and Kaggle notebook. |
| `kaggle-upload-v95-final-private-p05-e46.zip` | Same code package for Kaggle upload. |
| `assets/E00-v2-dataset.zip` | Original E00-v2 corpus, chunk/document metadata and BM25 index. |
| `assets/E45_dataset.zip` | Complete 7,000-record top12-v2 retrieval/training-data output. |
| `assets/E46_dataset.zip` | Complete fresh E46 Vi-Qwen adapter output, including final adapter and checkpoints. |
| `assets/P00_dataset.zip` | Complete shared retrieval output for the private questions. |
| `private-official.json` | Exactly 1,918 question-only private inputs; no reference answers. |
| `MANIFEST.json` | SHA-256 pins for every handoff file. |

The four `assets/*.zip` files use **Git LFS**. A clone without LFS gives only
small pointer files, not the datasets. The combined ZIPs are about 1.2 GB and
the full E00 archive expands to over 4 GB. Ensure sufficient disk space.

## Clone, verify and prepare

```bash
git lfs install
git clone https://github.com/lkey07/uit-dsc-2026-viqwen-e46-private-handoff.git
cd uit-dsc-2026-viqwen-e46-private-handoff
git lfs pull
python scripts/audit_assets.py
python scripts/prepare_assets.py
```

The audit verifies archive SHA-256 values, the E00 metadata pins, E45's 7,000
training records, E46's completed adapter and E45 lineage, and P00's exact
private-question hash. The preparation script extracts only files P05 needs to
`inputs/`; original full ZIPs remain available for further work. Extraction is
idempotent and never silently overwrites an existing file.

## Run private inference locally

Use Python 3.12 and one CUDA GPU with enough memory for the full FP16 Vi-Qwen
3B model plus up to 8,192 input and 1,536 output tokens. The E46 runtime pins
are `torch==2.10.0+cu128`, `transformers==5.16.1`, `peft==0.19.1` and
`accelerate==1.13.0`; install them in an isolated environment together with
`safetensors`, `huggingface_hub`, `tqdm` and `nltk`. Confirm CUDA and the exact
versions before running. No GPU training or retrieval rebuild is required.

```bash
python scripts/run_private.py --gpu 0
```

To use an already downloaded base model, pass `--model-cache PATH_TO_SNAPSHOT`.
Otherwise the runner downloads only revision
`eaf427c24d86066a2b35828c499b7db3af321227` of
`AITeamVN/Vi-Qwen2-3B-RAG`. The runner chooses physical GPU 0 by default;
`--gpu 1` exposes a different single device. Failed/incomplete runs resume
from per-question records in `output/`, after identity validation.

The only submission candidate is
`output/final-private-p05-e46-top12v2-max1536-unified-v1/submission.zip`.
**Review the report and ZIP before manually submitting.** This repository
does not submit to the competition or read private reference answers.

## Kaggle route

The notebook is at
`code-v95/uit-dsc-fixed-rag-code/notebooks/FINAL-private-p05-e46-top12v2-max1536-one-gpu-kaggle.ipynb`.
For Kaggle, attach the code package and extracted E00/E45/E46/P00/private
inputs as datasets; the notebook discovers them by experiment ID and hash.
Use one visible CUDA GPU. `scripts/prepare_assets.py` can create the minimal
extracted directories before packaging those inputs.

## Pipeline and limitations

P00's saved BM25/dense RRF top20 is reused; no retrieval or reranker runs.
The E45 selector keeps six RRF anchors and selects six legal-priority contexts,
with a final unique-chunk fill for duplicate-heavy pools. Before private
generation, P05 audits its selector against **all 7,000 saved E45 training
context selections** and stops if any differ. E18 source metadata is prefixed
exactly as in E45 training; P00 parent expansion is not used. The fresh E46
rank-8 adapter is loaded on the pinned Vi-Qwen base in FP16. Each private
question is generated once, greedily, from the original prompt at max1536;
there is no max1024 pass or continuation. E43 then E44 Unified Clean is
applied to each raw answer. Every QID is checkpointed; finalization emits
only a UTF-8 `submission.json` inside `submission.zip`.

No private score is claimed. E46 was trained on all 7,000 official training
records with answer-level supervision; those answers are not retrieval labels.
The total pinned embedding + generator + adapter parameter ceiling is below
the competition's 4-billion exclusive limit. This is fixed deterministic RAG,
not an agentic workflow.
