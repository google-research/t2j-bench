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

"""Bake google/IFEval into ``/data`` as PPO rollout prompts.

The PPO tasks used to roll out from four hand-written prompts. IFEval is 541 real
instruction-following prompts ("Write a resume ... in all lowercase"), which is what the
policy should be generating against. It runs at BUILD time because the agent image is
run with no network and no host mounts -- nothing can be fetched once the container
starts.

IFEval is **prompt-only** by construction: it ships no reference answers, because the
benchmark grades a completion by machine-checking the constraint (``instruction_id_list``
+ ``kwargs``) rather than by comparing to a target. That is exactly the shape PPO wants,
so ``output`` is emitted empty -- the unsupervised processor encodes the prompt and
ignores the response, and the reward comes from the reward endpoint, not the file.

Usage:  python prepare_ifeval.py --out /data
"""

import argparse
import json
import os
import sys

REPO_ID = "google/IFEval"
REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
FILENAME = "ifeval_input_data.jsonl"

DATASET_KEY = "ifeval"
OUT_NAME = "ifeval.jsonl"

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
}


def convert(row: dict) -> dict | None:
    """One IFEval row -> one prompt-only alpaca example, or None if it does not fit."""
    prompt = (row.get("prompt") or "").strip()
    if not prompt:
        return None
    return {"instruction": prompt, "input": "", "output": ""}


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

    from huggingface_hub import hf_hub_download

    # hf_hub_download, not datasets.load_dataset: one 0.2 MB fetch with no Arrow cache to
    # build and purge.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    total = kept = 0
    with open(src, encoding="utf-8") as handle, open(out_path, "w", encoding="utf-8") as out:
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
