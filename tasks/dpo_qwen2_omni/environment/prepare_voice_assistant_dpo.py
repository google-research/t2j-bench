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

"""Bake gpt-omni/VoiceAssistant-400K into ``/data`` as spoken-prompt preference pairs.

``qwen2_omni`` is the only model in the matrix with an audio tower, and the four
hand-written text rows this task used to train on never touched it -- the whole audio
path was dead weight in every run, and the ``chosen``/``rejected`` strings were identical
in all four, so the DPO objective saw a constant preference signal on top of that.

**The negatives are synthesised.** There is no spoken-prompt preference dataset upstream,
so ``chosen`` is this row's own ``answer`` and ``rejected`` is another row's answer: a
reply that is fluent and well-formed but answers a different question, which is exactly
the failure mode a speech-in model has to be pushed away from. The pairing is a single
n-cycle over a ``random.Random(0)`` shuffle -- deterministic across builds, and a cycle
rather than a plain shuffle so no row can ever be paired with itself.

Everything else matches the SFT converter: one shard (all 325 are 219.5 GB), and the
22.05 kHz float32 clips are downmixed and resampled to 16 kHz mono PCM_16 **here**, so
the training path only does ``soundfile.read`` plus a hard sample-rate assert. See
``prepare_voice_assistant.py`` for the full reasoning.

Usage:  python prepare_voice_assistant_dpo.py --out /data
"""

import argparse
import io
import json
import os
import posixpath
import random
import sys

REPO_ID = "gpt-omni/VoiceAssistant-400K"
REVISION = "65dab707fcdd3d43dcfb834398aa9fed4116be3a"
FILENAME = "data/train-00000-of-00325.parquet"
COLUMNS = ["question_audio", "answer"]

DATASET_KEY = "voice_assistant_dpo"
OUT_NAME = "voice_assistant_dpo.jsonl"
AUDIO_SUBDIR = posixpath.join("audio", DATASET_KEY)

TARGET_RATE = 16000
MAX_SECONDS = 30.0
PAIRING_SEED = 0

# The audio counterpart of "<image>"; see AUDIO_PLACEHOLDER in torch_orig/data/constants.py.
PROMPT = "<audio>"

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "audios": "audios",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
}


def write_wav_16k(payload: bytes, path: str) -> float:
    """Decode one embedded WAV, downmix + resample to 16 kHz mono, write PCM_16.

    Returns the output duration in seconds, or 0.0 if the clip could not be used.
    """
    import numpy as np
    import soundfile as sf
    import soxr

    data, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if data.size == 0:
        return 0.0
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != TARGET_RATE:
        mono = soxr.resample(mono, rate, TARGET_RATE)
    seconds = len(mono) / TARGET_RATE
    if seconds <= 0 or seconds > MAX_SECONDS:
        return 0.0
    # subtype="PCM_16" lets libsndfile do the scale-and-clip from float [-1, 1].
    sf.write(path, np.asarray(mono, dtype="float32"), TARGET_RATE, subtype="PCM_16")
    return seconds


def derange(count: int, seed: int = PAIRING_SEED) -> list[int]:
    """A fixed-point-free permutation of ``range(count)``.

    A shuffled single cycle: every index maps to the next one around the shuffled ring, so
    ``perm[i] != i`` holds by construction rather than by rejection sampling.
    """
    if count < 2:
        return list(range(count))
    order = list(range(count))
    random.Random(seed).shuffle(order)
    perm = [0] * count
    for position, index in enumerate(order):
        perm[index] = order[(position + 1) % count]
    return perm


def convert(row: dict, index: int, audio_dir: str) -> dict | None:
    """One VoiceAssistant row -> one half-built example, or None if it does not fit.

    ``rejected`` is filled in by ``main`` once every surviving row is known -- it comes
    from a different row, so it cannot be decided here.
    """
    answer = (row.get("answer") or "").strip()
    audio = row.get("question_audio")
    payload = audio.get("bytes") if isinstance(audio, dict) else None
    if not answer or not payload:
        return None

    name = f"{index:06d}.wav"
    path = os.path.join(audio_dir, name)
    if write_wav_16k(payload, path) == 0.0:
        if os.path.exists(path):
            os.remove(path)
        return None

    return {
        "conversations": [{"from": "human", "value": PROMPT}],
        "chosen": {"from": "gpt", "value": answer},
        "audios": [posixpath.join(AUDIO_SUBDIR, name)],
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

    # hf_hub_download + pyarrow, not datasets.load_dataset: load_dataset would want the
    # whole 219.5 GB repo listing, and we read row group by row group so the embedded
    # audio never all sits in memory at once.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    audio_dir = os.path.join(args.out, AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    # The clips stream to disk as they are decoded; only the text records are held, so
    # peak memory is a few MB of strings rather than the shard.
    handle = pq.ParquetFile(src)
    examples: list[dict] = []
    total = 0
    for batch in handle.iter_batches(batch_size=16, columns=COLUMNS):
        for row in batch.to_pylist():
            index = total
            total += 1
            example = convert(row, index, audio_dir)
            if example is None:
                continue
            examples.append(example)
            if args.limit is not None and len(examples) >= args.limit:
                break
        if args.limit is not None and len(examples) >= args.limit:
            break

    kept = len(examples)
    if kept < 2:
        print(
            f"ERROR: {kept} usable rows, need at least 2 to pair negatives",
            file=sys.stderr,
        )
        return 1

    perm = derange(kept)
    with open(out_path, "w", encoding="utf-8") as out:
        for position, example in enumerate(examples):
            example["rejected"] = dict(examples[perm[position]]["chosen"])
            out.write(json.dumps(example, ensure_ascii=False) + "\n")

    write_dataset_info(args.out)

    dropped = total - kept
    audio_bytes = sum(
        os.path.getsize(os.path.join(audio_dir, name)) for name in os.listdir(audio_dir)
    )
    print(
        f"wrote {out_path}: {kept} examples "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB), dropped {dropped}/{total} "
        f"({100 * dropped / max(total, 1):.2f}%); "
        f"{kept} clips at {TARGET_RATE} Hz mono PCM_16 under {audio_dir} "
        f"({audio_bytes / 1e6:.1f} MB); negatives paired by seed {PAIRING_SEED}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
