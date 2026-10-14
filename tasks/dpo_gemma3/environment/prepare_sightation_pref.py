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

"""Bake sightation/SightationPreference into ``/data`` as image preference pairs.

The vision DPO tasks used to train on four hand-written rows whose ``chosen`` and
``rejected`` strings were identical in every example, over four procedurally drawn shapes
-- a constant preference signal on a constant image. SightationPreference is 16,319 real
pairs of scientific-diagram descriptions, sighted-annotator ranked, so the reward margin
is grounded in something the vision tower actually has to look at. It runs at BUILD time
because the agent image is run with no network and no host mounts -- nothing can be
fetched once the container starts.

Two things about the upstream shape need handling:

**There is no prompt column.** The rows are (image, chosen description, rejected
description) only. A fixed instruction is synthesised as the single human turn -- the same
text for every row, which is fine for DPO: the pair, not the prompt, carries the signal.

**Roughly sixteen rows share each diagram** (one per annotator pass). The image bytes are
written once per distinct ``image_fn`` and every row referencing that diagram points at
the same file -- the difference between ~3.6 GB and a few hundred MB of layer, times the
six tasks that bake this. Paths are stored **relative**, so ``_find_medias`` resolves them
against ``data_args.media_dir or data_args.dataset_dir``, which is ``/data``.

Usage:  python prepare_sightation_pref.py --out /data
"""

import argparse
import json
import os
import posixpath
import sys

REPO_ID = "sightation/SightationPreference"
REVISION = "5316510f71126c8f88e5d29cbba3ff4cdcd95cc7"
SHARDS = [f"data/train-{index:05d}-of-00008.parquet" for index in range(8)]
COLUMNS = ["image", "image_fn", "chosen", "rejected"]

DATASET_KEY = "sightation_pref"
OUT_NAME = "sightation_pref.jsonl"
IMAGE_SUBDIR = posixpath.join("images", DATASET_KEY)

# Synthesised because the upstream rows carry no prompt. Leading <image> because the
# plugins expand placeholders in order and assert the count matches ``images``.
PROMPT = "<image>Describe this scientific diagram in detail for a reader who cannot see it."

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "images": "images",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
}


def image_name(image_fn: str, index: int) -> str:
    """A flat, safe file name for one diagram.

    ``image_fn`` is a bare name like ``100.png`` upstream, but it is data, so anything
    with a separator in it falls back to the row index rather than escaping the tree.
    """
    base = os.path.basename((image_fn or "").strip())
    if not base or base in (".", "..") or "/" in base or "\\" in base:
        return f"{index:06d}.png"
    return base


def convert(row: dict, index: int, image_dir: str, written: set[str]) -> dict | None:
    """One SightationPreference row -> one sharegpt-pairwise example, or None.

    Writes the row's diagram into ``image_dir`` as a side effect, but only the first time
    that diagram is seen -- ``written`` carries the names already on disk.
    """
    chosen = (row.get("chosen") or "").strip()
    rejected = (row.get("rejected") or "").strip()
    image = row.get("image")
    payload = image.get("bytes") if isinstance(image, dict) else None
    if not chosen or not rejected or not payload:
        return None
    if chosen == rejected:
        return None  # no preference signal, which is the whole point of the swap

    name = image_name(row.get("image_fn"), index)
    if name not in written:
        with open(os.path.join(image_dir, name), "wb") as out:
            out.write(payload)
        written.add(name)

    return {
        "conversations": [{"from": "human", "value": PROMPT}],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
        "images": [posixpath.join(IMAGE_SUBDIR, name)],
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

    image_dir = os.path.join(args.out, IMAGE_SUBDIR)
    os.makedirs(image_dir, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    written: set[str] = set()
    total = kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for shard in SHARDS:
            # One shard at a time, deleted as we go: all eight together are ~898 MB of
            # cache, and the RUN layer would keep whatever is still on disk at the end.
            src = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", filename=shard, revision=REVISION
            )
            print(f"downloaded {REPO_ID}/{shard} @ {REVISION[:8]} -> {src}", flush=True)

            handle = pq.ParquetFile(src)
            for batch in handle.iter_batches(batch_size=64, columns=COLUMNS):
                for row in batch.to_pylist():
                    index = total
                    total += 1
                    example = convert(row, index, image_dir, written)
                    if example is None:
                        continue
                    out.write(json.dumps(example, ensure_ascii=False) + "\n")
                    kept += 1
                    if args.limit is not None and kept >= args.limit:
                        break
                if args.limit is not None and kept >= args.limit:
                    break
            del handle
            if args.limit is not None and kept >= args.limit:
                break

    write_dataset_info(args.out)

    dropped = total - kept
    print(
        f"wrote {out_path}: {kept} examples "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB), dropped {dropped}/{total} "
        f"({100 * dropped / max(total, 1):.2f}%); "
        f"{len(written)} distinct images under {image_dir} "
        f"({kept / max(len(written), 1):.1f} rows per image)",
        flush=True,
    )
    if kept == 0:
        print("ERROR: no examples survived conversion", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
