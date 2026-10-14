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

"""Bake CyberNative/Code_Vulnerability_Security_DPO into ``/data`` as preference pairs.

``dpo_starcoder2`` used to train on four hand-written rows whose ``chosen``/``rejected``
strings were the same in every example, so the DPO objective saw a constant preference
signal. This is 4,656 real pairs where ``chosen`` is a secure implementation and
``rejected`` is the same task written with the vulnerability the row names -- a genuine,
code-shaped preference axis for a code model. It runs at BUILD time because the agent
image is run with no network and no host mounts -- nothing can be fetched once the
container starts.

The row already carries a single-turn question plus both completions, so no transcript
splitting is needed: ``question`` becomes the one human turn, ``chosen``/``rejected``
become the pair, and the row's ``system`` prompt is passed through the ``system`` column
(``_convert_sharegpt`` reads it when the messages list has no system turn). ``lang`` and
``vulnerability`` are dropped -- nothing downstream reads them.

Usage:  python prepare_code_vuln_dpo.py --out /data
"""

import argparse
import json
import os
import sys

REPO_ID = "CyberNative/Code_Vulnerability_Security_DPO"
REVISION = "81aeacf06cf43b16d7278a3a01f019a496a53c51"
FILENAME = "secure_programming_dpo.json"

DATASET_KEY = "code_vuln_dpo"
OUT_NAME = "code_vuln_dpo.jsonl"

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "system": "system",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
}


def convert(row: dict) -> dict | None:
    """One CyberNative row -> one sharegpt-pairwise example, or None if it does not fit."""
    question = (row.get("question") or "").strip()
    chosen = (row.get("chosen") or "").strip()
    rejected = (row.get("rejected") or "").strip()
    if not question or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None  # no preference signal, which is the whole point of the swap

    return {
        "conversations": [{"from": "human", "value": question}],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
        "system": (row.get("system") or "").strip(),
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

    from huggingface_hub import hf_hub_download

    # hf_hub_download, not datasets.load_dataset: one 6.9 MB fetch with no Arrow cache to
    # build and purge.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    total = kept = 0
    # Despite the .json extension upstream, the file is JSON Lines -- one object per line,
    # no enclosing array. json.load on the whole file raises "Extra data" at line 2.
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
