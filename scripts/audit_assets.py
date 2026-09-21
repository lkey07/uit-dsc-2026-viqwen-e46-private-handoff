"""Verify the exact E00/E45/E46/P00/private handoff without extracting it."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def sha_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def member_sha(archive: ZipFile, name: str) -> str:
    with archive.open(name) as stream:
        return sha_stream(stream)


def member_json(archive: ZipFile, name: str) -> dict:
    with archive.open(name) as stream:
        return json.load(stream)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        check(path.is_file(), f"Missing handoff file: {relative}")
        with path.open("rb") as stream:
            check(sha_stream(stream) == expected, f"SHA-256 mismatch: {relative}")
    private = json.loads((ROOT / "private-official.json").read_text(encoding="utf-8"))
    check(isinstance(private, dict) and len(private) == 1918, "Private question count changed")
    check(all(isinstance(row, dict) and isinstance(row.get("question"), str)
              and row["question"].strip() and row.get("answer") is None
              for row in private.values()), "Private input contains invalid or reference answers")

    archives = {key: ZipFile(ROOT / key) for key in manifest["archive_roots"]}
    code = ZipFile(ROOT / "kaggle-upload-v95-final-private-p05-e46.zip")
    try:
        for relative, archive in archives.items():
            for item in archive.infolist():
                parts = item.filename.split("/")
                check(not item.filename.startswith("/") and "\\" not in item.filename
                      and ".." not in parts and not stat.S_ISLNK(item.external_attr >> 16),
                      f"Unsafe archive member: {relative}:{item.filename}")

        prefix = "uit-dsc-fixed-rag-code/"
        e45_config = member_json(code, prefix + "configs/e45-top12v2-fulltrain7000-v1.json")
        e46_config = member_json(code, prefix + "configs/e46-viqwen-top12v2-fulltrain7000-v1.json")
        p00_config = member_json(code, prefix + "configs/final-private-p00-retrieval-parent-v1.json")
        check(e45_config["metadata_source"] == p00_config["metadata_source"],
              "E45/P00 metadata source differs")
        metadata = e45_config["metadata_source"]
        e00 = archives["assets/E00-v2-dataset.zip"]
        e00_root = manifest["archive_roots"]["assets/E00-v2-dataset.zip"] + "/"
        check(member_sha(e00, e00_root + "manifest.json") == e45_config["source_e00"]["manifest_sha256"],
              "E00 manifest SHA mismatch")
        for name, expected in (("chunks.jsonl", metadata["chunks_sha256"]),
                               ("documents.jsonl", metadata["documents_sha256"]),
                               ("bm25.sqlite3", e45_config["source_e00"]["bm25_sha256"])):
            check(member_sha(e00, e00_root + name) == expected, f"E00 {name} SHA mismatch")

        e45 = archives["assets/E45_dataset.zip"]
        e45_root = manifest["archive_roots"]["assets/E45_dataset.zip"] + "/"
        e45_report = member_json(e45, e45_root + "report.json")
        e45_summary = member_json(e45, e45_root + "training-data/summary.json")
        e45_report_sha = member_sha(e45, e45_root + "report.json")
        check(e45_report.get("experiment_id") == "E45-top12v2-fulltrain7000-v1"
              and e45_report.get("sample_size") == 7000
              and e45_report.get("answers_used_by_retrieval_or_selector") is False
              and e45_report.get("private_read") is False,
              "E45 report contract mismatch")
        for name in ("retrieval/raw-results.jsonl", "retrieval/candidate-pool-top20.jsonl",
                     "training-data/records.jsonl", "training-data/summary.json"):
            expected = e45_report["files"][name]["sha256"]
            check(member_sha(e45, e45_root + name) == expected, f"E45 {name} SHA mismatch")
        check(e45_summary["records_sha256"] == e45_report["files"]["training-data/records.jsonl"]["sha256"]
              and e45_summary["record_count"] == 7000, "E45 training records mismatch")

        e46 = archives["assets/E46_dataset.zip"]
        e46_root = manifest["archive_roots"]["assets/E46_dataset.zip"] + "/"
        complete = member_json(e46, e46_root + "adapter-final/complete.json")
        identity = member_json(e46, e46_root + "training-identity.json")
        check(complete.get("experiment_id") == "E46-viqwen-top12v2-fulltrain7000-v1"
              and complete.get("config_sha256") == member_sha(code, prefix + "configs/e46-viqwen-top12v2-fulltrain7000-v1.json")
              and complete.get("model_id") == manifest["base_model"]["id"]
              and complete.get("model_revision") == manifest["base_model"]["revision"]
              and complete.get("identity_sha256") == identity.get("identity_sha256")
              and identity.get("e45_report_sha256") == e45_report_sha
              and complete.get("training_records_sha256") == e45_summary["records_sha256"]
              and complete.get("official_answer_truncation_count") == 0
              and complete.get("e38_adapter_loaded") is False
              and complete.get("e19_adapter_loaded") is False,
              "E46 adapter/training lineage mismatch")
        check(complete.get("model_parameters") == e46_config["model"]["checkpoint_parameters"],
              "E46 model parameter count mismatch")
        check(member_sha(e46, e46_root + "adapter-final/adapter_model.safetensors")
              == complete["adapter_sha256"], "E46 adapter SHA mismatch")

        p00 = archives["assets/P00_dataset.zip"]
        p00_root = manifest["archive_roots"]["assets/P00_dataset.zip"] + "/"
        p00_report = member_json(p00, p00_root + "report.json")
        check(p00_report.get("experiment_id") == "FINAL-private-p00-retrieval-parent-v1"
              and p00_report.get("evidence", {}).get("config_sha256")
              == member_sha(code, prefix + "configs/final-private-p00-retrieval-parent-v1.json")
              and p00_report.get("sample_size") == len(private)
              and p00_report.get("private_questions_sha256") == manifest["files"]["private-official.json"]
              and p00_report.get("answers_used") is False
              and p00_report.get("private_reference_answers_read") is False,
              "P00/private identity mismatch")
        for name in ("retrieval/candidate-pool-top20.jsonl", "retrieval/raw-results.jsonl",
                     "prepared/results.jsonl"):
            check(member_sha(p00, p00_root + name) == p00_report["files"][name]["sha256"],
                  f"P00 {name} SHA mismatch")
    finally:
        code.close()
        for archive in archives.values():
            archive.close()
    print(json.dumps({"status": "VERIFIED", "private_questions": len(private),
                      "e45_training_records": 7000, "e46_adapter_sha256": complete["adapter_sha256"],
                      "e45_report_sha256": e45_report_sha,
                      "p00_private_sha256": p00_report["private_questions_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
