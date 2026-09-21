"""Check P05 top12-v2 logic against all saved E45 selections, ZIP-to-ZIP."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code-v95/uit-dsc-fixed-rag-code/src"))

from uit_dsc_fixed_rag.e18_source_metadata import enrich_context  # noqa: E402
from uit_dsc_fixed_rag.e45_top12v2_fulltrain import select_top12_v2  # noqa: E402


def main() -> None:
    with ZipFile(ROOT / "assets/E45_dataset.zip") as e45, ZipFile(ROOT / "assets/E00-v2-dataset.zip") as e00:
        raw_name = "e45-top12v2-fulltrain7000-v1/retrieval/raw-results.jsonl"
        train_name = "e45-top12v2-fulltrain7000-v1/training-data/records.jsonl"
        wanted: set[str] = set()
        with e45.open(raw_name) as source:
            for line in source:
                wanted.update(item["chunk_id"] for item in json.loads(line)["fused_pool"])
        print(f"E45 candidate chunk IDs: {len(wanted)}", flush=True)
        chunks: dict[str, dict] = {}
        with e00.open("e00-v2/chunks.jsonl") as source:
            for line in source:
                row = json.loads(line)
                if row["chunk_id"] in wanted:
                    chunks[row["chunk_id"]] = row
        if set(chunks) != wanted:
            raise RuntimeError("E45 candidate chunks missing from E00")
        wanted_documents = {chunk["document_id"] for chunk in chunks.values()}
        print(f"E45 source documents: {len(wanted_documents)}", flush=True)
        documents: dict[str, dict] = {}
        with e00.open("e00-v2/documents.jsonl") as source:
            for line in source:
                row = json.loads(line)
                if row["document_id"] in wanted_documents:
                    documents[row["document_id"]] = row
        if set(documents) != wanted_documents:
            raise RuntimeError("E45 source documents missing from E00")
        selector = json.loads((ROOT / "code-v95/uit-dsc-fixed-rag-code/configs/e45-top12v2-fulltrain7000-v1.json")
                              .read_text(encoding="utf-8"))["selector"]
        count, mismatch, first = 0, 0, []
        with e45.open(raw_name) as raw_source, e45.open(train_name) as train_source:
            for raw_line, train_line in itertools.zip_longest(raw_source, train_source):
                if raw_line is None or train_line is None:
                    raise RuntimeError("E45 raw and training row counts differ")
                raw, trained = json.loads(raw_line), json.loads(train_line)
                if raw["question_id"] != trained["question_id"] or raw["sample_index"] != count:
                    raise RuntimeError(f"E45 row identity mismatch at {count}")
                candidates = []
                for rank, context in enumerate(raw["candidate_contexts"]):
                    chunk = chunks[context["chunk_id"]]
                    enriched, evidence = enrich_context(dict(context), chunk, documents[chunk["document_id"]])
                    candidates.append({"context": enriched, "body": context["text"],
                                       "evidence": evidence, "rrf_rank": rank,
                                       "rrf_score": raw["fused_pool"][rank]["rrf_score"]})
                selected, _ = select_top12_v2(trained["question"], candidates, selector,
                                               allow_content_relaxed_fill=True)
                actual = [item["context"]["chunk_id"] for item in selected]
                if actual != trained["selected_chunk_ids"]:
                    mismatch += 1
                    if len(first) < 5:
                        first.append({"sample_index": count, "question_id": trained["question_id"],
                                      "actual_ranks": [item["rrf_rank"] for item in selected],
                                      "saved_ranks": trained["selected_rrf_ranks"]})
                count += 1
                if count % 1000 == 0:
                    print(f"Selector audit: {count}/7000, mismatches={mismatch}", flush=True)
        print(json.dumps({"audited": count, "mismatches": mismatch, "first": first}, indent=2), flush=True)
        if count != 7000 or mismatch:
            raise RuntimeError("P05 selector does not reproduce saved E45 training contexts")


if __name__ == "__main__":
    main()
