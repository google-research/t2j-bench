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

"""Bake openai/openai_humaneval into ``/data`` as PPO rollout prompts.

``ppo_starcoder2`` is the code model in the matrix, and the four hand-written English
sentences it used to roll out from told it nothing about code. HumanEval is 164 Python
function signatures with docstrings -- the canonical completion prompts for a code
policy. It runs at BUILD time because the agent image is run with no network and no host
mounts -- nothing can be fetched once the container starts.

HumanEval ships only a ``test`` split (it is an eval benchmark), which is fine here: the
rollout prompt set does not have to be disjoint from anything, and 164 prompts is
comfortably more than ``max_samples``. ``canonical_solution`` is carried through as
``output`` -- PPO's unsupervised processor reads only the prompt, but a real reference
keeps the file useful for a supervised sanity check.

Usage:  python prepare_humaneval.py --out /data
"""

import argparse
import json
import os
import sys

REPO_ID = "openai/openai_humaneval"
REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
FILENAME = "openai_humaneval/test-00000-of-00001.parquet"

DATASET_KEY = "humaneval"
OUT_NAME = "humaneval.jsonl"

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
}


def convert(row: dict) -> dict | None:
    """One HumanEval row -> one alpaca example, or None if it does not fit."""
    prompt = row.get("prompt") or ""
    if not prompt.strip():
        return None
    # Not .strip()ed: the signature's trailing newline and the docstring's indentation
    # are part of the prompt a code model completes from.
    return {
        "instruction": prompt,
        "input": "",
        "output": row.get("canonical_solution") or "",
    }


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
    # purge.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    rows = pq.read_table(src, columns=["prompt", "canonical_solution"]).to_pylist()
    total = kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            total += 1
            example = convert(row)
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
