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

"""Bake bigcode/commitpackft into ``/data`` as code-edit instruction pairs.

``sft_starcoder2`` is the code model in the matrix, and the four hand-written English
sentences it used to train on told it nothing about code. CommitPackFT is filtered commit
data across 277 languages: the commit message is the instruction, the pre-image is the
input and the post-image is the target, which is exactly the edit-following task a code
model is finetuned for. It runs at BUILD time because the agent image is run with no
network and no host mounts -- nothing can be fetched once the container starts.

This is the largest output of the twelve converters -- 702k rows, ~1.3 GB of JSONL --
and it is JSONL anyway. ``FILEEXT2TYPE`` in ``torch_orig/data/constants.py`` advertises
``parquet``, but nothing implements it: both readers behind it are minimal vendored ones
(``torch_orig/_dataset_utils.load_dataset`` and ``ref_jax_repo/datasets/dataset.py``)
that handle json/jsonl/csv/txt and raise on anything else. Teaching them parquet would
mean editing the contract in 57 files to save ~9 s of container start, so it stays JSONL.

The 277 language configs are separate ``data/{lang}/data.jsonl`` files, pulled in one
``snapshot_download`` with an ``allow_patterns`` filter -- ``load_dataset`` would want a
config name and would fetch them one config at a time.

Usage:  python prepare_commitpackft.py --out /data
"""

import argparse
import glob
import json
import os
import sys

REPO_ID = "bigcode/commitpackft"
REVISION = "fc56fe33c030c6daa414c2b112c932b8eed085e6"
ALLOW_PATTERNS = ["data/*/data.jsonl"]

DATASET_KEY = "commitpackft"
OUT_NAME = "commitpackft.jsonl"

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
}


def convert(row: dict) -> dict | None:
    """One CommitPackFT row -> one alpaca example, or None if it does not fit."""
    instruction = (row.get("message") or "").strip()
    old = row.get("old_contents") or ""
    new = row.get("new_contents") or ""
    if not instruction or not new.strip():
        return None
    if old == new:
        return None  # a no-op edit teaches the model to echo its input
    return {"instruction": instruction, "input": old, "output": new}


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

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=REVISION, allow_patterns=ALLOW_PATTERNS
    )
    shards = sorted(glob.glob(os.path.join(root, "data", "*", "data.jsonl")))
    print(
        f"downloaded {REPO_ID} @ {REVISION[:8]} -> {root} ({len(shards)} language shards)",
        flush=True,
    )
    if not shards:
        print(f"ERROR: {ALLOW_PATTERNS} matched nothing in {root}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    total = kept = 0
    # Streamed shard by shard: 277 files in, one file out, never more than a line in memory.
    with open(out_path, "w", encoding="utf-8") as out:
        for shard in shards:
            with open(shard, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    example = convert(json.loads(line))
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
