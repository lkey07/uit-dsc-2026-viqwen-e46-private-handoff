"""Materialize only the verified files required by P05 from the full ZIPs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

from audit_assets import main as audit_assets

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"

WANTED = {
    "assets/E00-v2-dataset.zip": (
        "manifest.json", "chunks.jsonl", "documents.jsonl",
    ),
    "assets/E45_dataset.zip": (
        "report.json", "retrieval/raw-results.jsonl",
        "training-data/summary.json", "training-data/records.jsonl",
    ),
    "assets/E46_dataset.zip": (
        "training-identity.json", "adapter-final/complete.json",
        "adapter-final/adapter_config.json", "adapter-final/adapter_model.safetensors",
    ),
    "assets/P00_dataset.zip": (
        "report.json", "identity.json", "retrieval/candidate-pool-top20.jsonl",
        "prepared/results.jsonl",
    ),
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    audit_assets()
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    # Check only files still missing; a prior P05 extraction is expected here.
    required_bytes = 0
    for relative, members in WANTED.items():
        prefix = manifest["archive_roots"][relative] + "/"
        destination_root = INPUTS / manifest["archive_roots"][relative]
        with ZipFile(ROOT / relative) as archive:
            for name in members:
                expected_size = archive.getinfo(prefix + name).file_size
                destination = destination_root / name
                if destination.is_file():
                    if destination.stat().st_size != expected_size:
                        raise RuntimeError(f"Existing extracted file differs: {destination}")
                else:
                    required_bytes += expected_size
    if shutil.disk_usage(ROOT).free < required_bytes + 512 * 1024 * 1024:
        raise RuntimeError(f"Need about {required_bytes / 2**30:.1f} GiB plus 0.5 GiB free to extract inputs")
    for relative, members in WANTED.items():
        prefix = manifest["archive_roots"][relative] + "/"
        destination_root = INPUTS / manifest["archive_roots"][relative]
        with ZipFile(ROOT / relative) as archive:
            for name in members:
                source = prefix + name
                destination = destination_root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file():
                    if destination.stat().st_size != archive.getinfo(source).file_size:
                        raise RuntimeError(f"Existing extracted file differs: {destination}")
                    print(f"Already present: {destination}", flush=True)
                    continue
                temporary = destination.with_name(destination.name + ".part")
                if temporary.exists():
                    raise RuntimeError(f"Incomplete prior extraction; remove only this .part file: {temporary}")
                with archive.open(source) as input_stream, temporary.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                if temporary.stat().st_size != archive.getinfo(source).file_size:
                    raise RuntimeError(f"Extracted size differs: {source}")
                os.replace(temporary, destination)
                print(f"Extracted: {destination}", flush=True)
    print("Inputs ready. Run scripts/run_private.py on one CUDA GPU.", flush=True)


if __name__ == "__main__":
    main()
