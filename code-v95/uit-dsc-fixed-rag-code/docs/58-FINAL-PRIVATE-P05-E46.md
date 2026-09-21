# P05 — E46 private one-GPU direct max1536

Inputs: code-v95, completed P00 output, E00-v2 chunks/documents, completed E45 output,
completed E46 adapter, and the private question file. Do not attach private answers.

The notebook reconstructs the full P00 RRF top20 from the E00 source, applies the
E45 top12-v2 selector with its audited final fill for duplicate-heavy pools, and
adds the same E18 source metadata prefixes used in E45 training. It does not use
P00 parent-context expansion. It loads the fresh E46 adapter over the pinned
Vi-Qwen base in FP16 on one CUDA GPU. Every question is generated once from the
original prompt with greedy max1536; there is no max1024 pass or continuation.

Each QID is written atomically and resume validates run identity and record hashes.
Finalization applies E43 then E44 Unified Clean to every raw answer, writes
`results.jsonl`, review diagnostics, and a ZIP containing only `submission.json`.
Review report and output before manually submitting. No private score is known.

The ZIP made by `scripts/package_final_private_p05_e46_kaggle.py` contains code
and config only; it does not contain E00, E45, E46, P00 or private questions.
Public GitHub handoff must be reviewed for secrets and private data before upload.
