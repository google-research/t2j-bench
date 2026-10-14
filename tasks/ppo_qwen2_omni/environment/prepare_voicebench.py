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

"""Bake hlt-lab/voicebench (alpacaeval) into ``/data`` as PPO rollout prompts.

``qwen2_omni`` is the only model in the matrix with an audio tower, and the four
hand-written text rows this task used to roll out from never touched it -- the whole
audio path was dead weight in every run. VoiceBench's ``alpacaeval`` subset is 199 spoken
open-ended instructions, which is what a speech-in policy should be rolled out on. It
runs at BUILD time because the agent image is run with no network and no host mounts --
nothing can be fetched once the container starts.

Prompt-only by design: PPO generates the response and scores it with the reward model, so
``output`` is empty and the ``prompt`` text column is dropped -- the instruction lives in
the audio, and keeping the transcript around would only tempt something downstream into
reading it.

Unlike VoiceAssistant-400K, this one already ships 16 kHz mono int16, so the resample is
a no-op -- but the clips still go through the same normalize-and-write path, because
"upstream happens to match" is not something to leave unasserted in a build step.

Usage:  python prepare_voicebench.py --out /data
"""

import argparse
import io
import json
import os
import posixpath
import sys

REPO_ID = "hlt-lab/voicebench"
REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
FILENAME = "alpacaeval/test-00000-of-00001.parquet"
COLUMNS = ["audio"]

DATASET_KEY = "voicebench"
OUT_NAME = "voicebench.jsonl"
AUDIO_SUBDIR = posixpath.join("audio", DATASET_KEY)

TARGET_RATE = 16000
MAX_SECONDS = 30.0

# The audio counterpart of "<image>"; see AUDIO_PLACEHOLDER in torch_orig/data/constants.py.
PROMPT = "<audio>"

DATASET_INFO = {
    DATASET_KEY: {
        "file_name": OUT_NAME,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "audios": "audios",
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


def convert(row: dict, index: int, audio_dir: str) -> dict | None:
    """One VoiceBench row -> one prompt-only alpaca example, or None if it does not fit.

    Writes the row's clip into ``audio_dir`` as a side effect and returns a path relative
    to the dataset dir, which is where ``_find_medias`` resolves it from.
    """
    audio = row.get("audio")
    payload = audio.get("bytes") if isinstance(audio, dict) else None
    if not payload:
        return None

    name = f"{index:06d}.wav"
    path = os.path.join(audio_dir, name)
    if write_wav_16k(payload, path) == 0.0:
        if os.path.exists(path):
            os.remove(path)
        return None

    return {
        "instruction": PROMPT,
        "input": "",
        "output": "",
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

    # hf_hub_download + pyarrow, not datasets.load_dataset: no Arrow cache to build and
    # purge, and no need to name a config out of the eight VoiceBench subsets.
    src = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=FILENAME, revision=REVISION
    )
    print(f"downloaded {REPO_ID}/{FILENAME} @ {REVISION[:8]} -> {src}", flush=True)

    audio_dir = os.path.join(args.out, AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)
    out_path = os.path.join(args.out, OUT_NAME)

    handle = pq.ParquetFile(src)
    total = kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for batch in handle.iter_batches(batch_size=16, columns=COLUMNS):
            for row in batch.to_pylist():
                index = total
                total += 1
                example = convert(row, index, audio_dir)
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
    audio_bytes = sum(
        os.path.getsize(os.path.join(audio_dir, name)) for name in os.listdir(audio_dir)
    )
    print(
        f"wrote {out_path}: {kept} examples "
        f"({os.path.getsize(out_path) / 1e6:.1f} MB), dropped {dropped}/{total} "
        f"({100 * dropped / max(total, 1):.2f}%); "
        f"{kept} clips at {TARGET_RATE} Hz mono PCM_16 under {audio_dir} "
        f"({audio_bytes / 1e6:.1f} MB)",
        flush=True,
    )
    if kept == 0:
        print("ERROR: no examples survived conversion", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
