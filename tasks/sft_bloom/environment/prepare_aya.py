# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bake CohereForAI/aya_dataset into ``/data`` as instruction/response pairs.

``bloom`` is the multilingual model in the matrix (46 languages), and the four
hand-written English rows it used to train on exercised none of that. Aya is 202k
human-annotated prompt/completion pairs across 65 languages, so the tokenizer sees the
byte-level pressure the model was actually pretrained for. It runs at BUILD time because
the agent image is run with no network and no host mounts -- nothing can be fetched once
the container starts.

The mapping is direct: ``inputs`` -> ``instruction``, ``targets`` -> ``output``. The
``language``/``annotation_type``/``user_id`` columns are dropped -- nothing downstream
reads them, and they would only bloat the image.

Usage:  python prepare_aya.py --out /data
"""

import argparse
import json
import os
import sys

REPO_ID = "CohereForAI/aya_dataset"
REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
FILENAME = "data/train-00000-of-00001.parquet"

DATASET_KEY = "aya"
OUT_NAME = "aya.jsonl"

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
}


def convert(row: dict) -> dict | None:
    """One aya row -> one alpaca example, or None if it does not fit."""
    instruction = (row.get("inputs") or "").strip()
    output = (row.get("targets") or "").strip()
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": "", "output": output}


def write_dataset_info(out_dir: str) -> None:
    """Merge our entry into ``{out}/dataset_info.json`` rather than clobbering it."""
    path = os.path.join(out_dir, "dataset_info.json")
    info = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            info = json.load(handle)
    info.update(DATASET_INFO)
    with open(path, "w", encoding="utf-8") as out:
        json.dump(info, out, indent=2)
        out.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/data", help="output directory (default: /data)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N converted rows")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    # hf_hub_download + pyarrow, not datasets.load_dataset: no Arrow cache to build and
    # purge, and we read row group by row group so peak memory stays flat.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    handle = pq.ParquetFile(src)
    total = kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for batch in handle.iter_batches(batch_size=1024, columns=["inputs", "targets"]):
            for row in batch.to_pylist():
                total += 1
                example = convert(row)
                if example is None:
                    continue
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept += 1
                if args.limit is not None and kept >= args.limit:
                    break
            if args.limit is not None and kept >= args.limit:
                break

    write_dataset_info(args.out)

    dropped = total - kept
    print(
        f"wrote {out_path}: {kept} examples "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB), dropped {dropped}/{total} "
        f"({100 * dropped / max(total, 1):.2f}%)",
        flush=True,
    )
    if kept == 0:
        print("ERROR: no examples survived conversion", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
