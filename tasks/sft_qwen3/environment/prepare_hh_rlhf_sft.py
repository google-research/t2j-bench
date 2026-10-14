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

"""Bake Anthropic/hh-rlhf (helpful-base) into ``/data`` as SFT conversations.

The SFT tasks used to train on four hand-written rows that shared a single output
string, so the loss saw almost no signal. This downloads the real thing instead. It runs
at BUILD time because the agent image is run with no network and no host mounts --
nothing can be fetched once the container starts.

Only ``helpful-base`` is taken. The other three subsets include ``harmless-base``, which
is red-teaming data containing explicit slurs by design and has no business being baked
into a benchmark image.

This is the supervised half of the same source ``prepare_hh_rlhf.py`` converts for DPO:
there we keep the prompt plus a chosen/rejected pair, here we keep the **chosen**
transcript whole and hand it over as multi-turn sharegpt. ``_convert_sharegpt`` checks
turn parity (even index -> user tag, odd -> assistant tag) and on violation sets
``broken = True`` and returns an empty ``_prompt`` *silently*, so rows that do not fit
the shape are dropped here, loudly, instead.

Usage:  python prepare_hh_rlhf_sft.py --out /data
"""

import argparse
import gzip
import json
import os
import re
import sys

REPO_ID = "Anthropic/hh-rlhf"
# The dataset has been static since 2023-05-26; pinning keeps the image reproducible.
REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
FILENAME = "helpful-base/train.jsonl.gz"

DATASET_KEY = "hh_rlhf_sft"
OUT_NAME = "hh_rlhf_sft.jsonl"

# Turn boundaries. hh-rlhf transcripts open with "\n\nHuman: " and alternate from there.
TURN_RE = re.compile(r"\n\n(Human|Assistant): ")
ROLE_TAG = {"Human": "human", "Assistant": "gpt"}

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
}


def split_turns(transcript: str) -> list[dict[str, str]]:
    """Flat transcript -> ``[{"from": "human"|"gpt", "value": ...}, ...]``.

    Returns ``[]`` for anything that is not a strict Human/Assistant alternation
    starting with Human -- the caller drops those rather than letting the loader
    silently turn them into empty prompts.
    """
    matches = list(TURN_RE.finditer(transcript))
    if not matches or transcript[: matches[0].start()].strip():
        return []  # no turns, or stray text before the first one

    turns = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
        expected = "Human" if i % 2 == 0 else "Assistant"
        if match.group(1) != expected:
            return []
        turns.append({"from": ROLE_TAG[match.group(1)], "value": transcript[match.end() : end]})

    # Must be complete Human/Assistant pairs: SFT trains on the assistant turns.
    if len(turns) % 2 != 0:
        return []
    return turns


def convert(chosen: str) -> dict | None:
    """One hh-rlhf row -> one multi-turn sharegpt example, or None if it does not fit."""
    turns = split_turns(chosen)
    if not turns:
        return None
    if any(not turn["value"].strip() for turn in turns):
        return None  # an empty turn contributes no tokens and confuses the label mask
    return {"conversations": turns}


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

    # hf_hub_download, not datasets.load_dataset: one 16 MB fetch with no Arrow cache to
    # build and purge, and the hub exposes only a "default" config -- the subsets are
    # directories, reachable by path but not by config name.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    total = kept = 0
    with gzip.open(src, "rt", encoding="utf-8") as handle, open(
        out_path, "w", encoding="utf-8"
    ) as out:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            example = convert(json.loads(line).get("chosen", ""))
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
