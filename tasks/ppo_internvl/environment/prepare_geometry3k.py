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

"""Bake hiyouga/geometry3k into ``/data`` as image+instruction pairs.

The vision-language tasks used to train on four procedurally drawn shapes (a red circle,
some diagonal stripes) rendered at a fixed size, so the vision tower saw one aspect ratio
and one trivial layout. Geometry3K is 2,101 real geometry diagrams with worked questions:
varied aspect ratios, dense line art, and text the model has to read off the figure. It
runs at BUILD time because the agent image is run with no network and no host mounts --
nothing can be fetched once the container starts.

The parquet embeds each diagram as raw bytes, so the images are written out beside the
records and referenced by **relative** path -- ``_find_medias`` joins them against
``data_args.media_dir or data_args.dataset_dir``, which is ``/data``.

``problem`` already carries its own ``<image>`` placeholder, and the plugins assert that
placeholder count equals image count, so rows whose two counts disagree are dropped here,
loudly, rather than raising mid-epoch.

Usage:  python prepare_geometry3k.py --out /data
"""

import argparse
import json
import os
import sys

REPO_ID = "hiyouga/geometry3k"
REVISION = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"
FILENAME = "data/train-00000-of-00001.parquet"

DATASET_KEY = "geometry3k"
OUT_NAME = "geometry3k.jsonl"
IMAGE_SUBDIR = os.path.join("images", DATASET_KEY)

IMAGE_PLACEHOLDER = "<image>"

# Magic bytes -> extension. The loader opens by content, but a correct suffix keeps the
# baked tree inspectable and stops PIL from guessing on a truncated payload.
MAGIC = ((b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"GIF8", ".gif"))

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "images": "images",
        },
    }
}


def image_ext(payload: bytes) -> str:
    for magic, ext in MAGIC:
        if payload.startswith(magic):
            return ext
    return ".png"


def convert(row: dict, index: int, image_dir: str) -> dict | None:
    """One Geometry3K row -> one alpaca example, or None if it does not fit.

    Writes the row's diagrams into ``image_dir`` as a side effect and returns paths
    relative to the dataset dir.
    """
    problem = (row.get("problem") or "").strip()
    answer = (row.get("answer") or "").strip()
    images = row.get("images") or []
    if not problem or not answer or not images:
        return None
    if problem.count(IMAGE_PLACEHOLDER) != len(images):
        return None  # the plugin asserts these match; a mismatch would fail mid-epoch

    relatives = []
    for position, image in enumerate(images):
        payload = image.get("bytes") if isinstance(image, dict) else None
        if not payload:
            return None
        name = f"{index:06d}_{position}{image_ext(payload)}"
        with open(os.path.join(image_dir, name), "wb") as out:
            out.write(payload)
        relatives.append(os.path.join(IMAGE_SUBDIR, name))

    return {"instruction": problem, "input": "", "output": answer, "images": relatives}


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
    # purge, and we read row group by row group so the embedded images never all sit in
    # memory at once.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    image_dir = os.path.join(args.out, IMAGE_SUBDIR)
    os.makedirs(image_dir, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    handle = pq.ParquetFile(src)
    total = kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for batch in handle.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                index = total
                total += 1
                example = convert(row, index, image_dir)
                if example is None:
                    continue
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept += 1
                if args.limit is not None and kept >= args.limit:
                    break
            if args.limit is not None and kept >= args.limit:
                break

    write_dataset_info(args.out)

    written = len(os.listdir(image_dir))
    dropped = total - kept
    print(
        f"wrote {out_path}: {kept} examples "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB), dropped {dropped}/{total} "
        f"({100 * dropped / max(total, 1):.2f}%); {written} images under {image_dir}",
        flush=True,
    )
    if kept == 0:
        print("ERROR: no examples survived conversion", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
