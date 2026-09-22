"""Read-only audit of all 1,918 top12-v2 parent units inside the ZIP assets."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code-v95/uit-dsc-fixed-rag-code"
sys.path.insert(0, str(CODE / "src"))

from uit_dsc_fixed_rag.e18_source_metadata import enrich_context  # noqa: E402
from uit_dsc_fixed_rag.e21_parent_context import article_blocks, seed_unit  # noqa: E402
from uit_dsc_fixed_rag.e45_top12v2_fulltrain import select_top12_v2  # noqa: E402
from uit_dsc_fixed_rag.e46_viqwen_top12v2_fulltrain import load_config  # noqa: E402
from uit_dsc_fixed_rag.final_private_p00 import load_private_questions  # noqa: E402


def main() -> None:
    ec = load_config(CODE, CODE / "configs/e46-viqwen-top12v2-fulltrain7000-v1.json")
    questions, ids, identity = load_private_questions(ROOT / "private-official.json")
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    p00_root = manifest["archive_roots"]["assets/P00_dataset.zip"]
    e00_root = manifest["archive_roots"]["assets/E00-v2-dataset.zip"]
    with ZipFile(ROOT / "assets/P00_dataset.zip") as archive:
        report = json.loads(archive.read(f"{p00_root}/report.json"))
        if (report.get("sample_size") != len(ids)
                or report.get("private_questions_sha256") != identity["private_sha256"]):
            raise RuntimeError("P00/private identity differs.")
        pool = [json.loads(line) for line in archive.open(f"{p00_root}/retrieval/candidate-pool-top20.jsonl")]
    wanted = {item["chunk_id"] for row in pool for item in row["fused_pool"]}
    metadata = ec.e45.section("metadata_source")
    with ZipFile(ROOT / "assets/E00-v2-dataset.zip") as archive:
        chunks, digest = {}, hashlib.sha256()
        with archive.open(f"{e00_root}/chunks.jsonl") as stream:
            for line in stream:
                digest.update(line)
                chunk = json.loads(line)
                if chunk["chunk_id"] in wanted:
                    chunks[chunk["chunk_id"]] = chunk
        if digest.hexdigest() != metadata["chunks_sha256"] or set(chunks) != wanted:
            raise RuntimeError("E00 chunks changed or candidate chunk missing.")
        document_ids = {chunk["document_id"] for chunk in chunks.values()}
        documents, digest = {}, hashlib.sha256()
        with archive.open(f"{e00_root}/documents.jsonl") as stream:
            for line in stream:
                digest.update(line)
                document = json.loads(line)
                if document["document_id"] in document_ids:
                    documents[document["document_id"]] = document
        if digest.hexdigest() != metadata["documents_sha256"] or set(documents) != document_ids:
            raise RuntimeError("E00 documents changed or document missing.")
        selected, changed = [], 0
        for index, (qid, row) in enumerate(zip(ids, pool)):
            if row.get("question_id") != qid or row.get("sample_index") != index:
                raise RuntimeError(f"P00 pool order differs: {index}")
            candidates = []
            for rank, fused in enumerate(row["fused_pool"]):
                chunk = chunks[fused["chunk_id"]]
                context = {key: chunk[key] for key in ("chunk_id", "document_id", "article_number", "text")}
                enriched, evidence = enrich_context(context, chunk, documents[chunk["document_id"]])
                candidates.append({"context": enriched, "body": context["text"],
                                   "evidence": evidence, "rrf_rank": rank,
                                   "rrf_score": fused["rrf_score"]})
            top12, _ = select_top12_v2(questions[qid], candidates, ec.e45.section("selector"),
                                       allow_content_relaxed_fill=True)
            selected.append([item["context"]["chunk_id"] for item in top12])
            changed += [item["rrf_rank"] for item in top12] != list(range(12))
        selected_docs = {chunks[cid]["document_id"] for row in selected for cid in row}
        by_document = {doc_id: [] for doc_id in selected_docs}
        digest = hashlib.sha256()
        with archive.open(f"{e00_root}/chunks.jsonl") as stream:
            for line in stream:
                digest.update(line)
                chunk = json.loads(line)
                if chunk["document_id"] in by_document:
                    by_document[chunk["document_id"]].append(chunk)
        if digest.hexdigest() != metadata["chunks_sha256"]:
            raise RuntimeError("E00 chunks changed during parent scan.")
    selected_ids = {cid for row in selected for cid in row}
    blocks = {}
    for doc_id, items in by_document.items():
        for block in article_blocks(items, documents[doc_id]):
            for chunk in block:
                if chunk["chunk_id"] in selected_ids:
                    blocks[chunk["chunk_id"]] = block
    if set(blocks) != selected_ids:
        raise RuntimeError("Top12-v2 parent block missing.")
    policy = report["context_policy"]
    expandable = 0
    for chosen in selected:
        units = [seed_unit(chunks[cid], rank, blocks[cid],
                           documents[chunks[cid]["document_id"]], policy)
                 for rank, cid in enumerate(chosen)]
        expandable += any(unit["expansions"] for unit in units)
    print(json.dumps({"status": "VERIFIED", "questions": len(ids),
                      "changed_from_rrf12": changed,
                      "questions_with_available_parent": expandable,
                      "selector": ec.e45.section("selector")["name"],
                      "answers_read": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
