#!/usr/bin/env python3
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

"""git-lfs standalone transfer agent backed by a Hugging Face Storage Bucket.

git-lfs keeps pointer files in git; this agent moves the object bytes to and from the
bucket. Each object is stored once, keyed by its sha256 oid:

    hf://buckets/{T2J_LFS_BUCKET}/lfs/objects/{oid[0:2]}/{oid[2:4]}/{oid}

Objects are content-addressed and never overwritten, so the (unversioned) bucket is safe
to use as LFS storage -- git history decides which version a checkout gets. Never run
`hf sync --delete` / `hf buckets rm` against the lfs/ prefix: bucket deletes are permanent.

Registered by tools/setup_lfs.sh. Auth uses the normal huggingface_hub token lookup
(HF_TOKEN, or `hf auth login`); downloads from a public bucket need no token.

Also usable directly, to upload objects without a git push:
    python tools/lfs_hf_bucket.py upload FILE [FILE ...]
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import batch_bucket_files, download_bucket_files, get_bucket_paths_info  # noqa: E402

BUCKET = os.environ.get("T2J_LFS_BUCKET", "jiefengcgg/t2j-bench")
PREFIX = "lfs/objects"
UPLOAD_BATCH = 64


def object_key(oid):
    return f"{PREFIX}/{oid[0:2]}/{oid[2:4]}/{oid}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_sizes(keys):
    """{key: size} for the keys already present in the bucket."""
    return {f.path: f.size for f in get_bucket_paths_info(BUCKET, keys)}


def lfs_tmp_dir():
    git_dir = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()
    path = os.path.join(git_dir, "lfs", "tmp")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------- transfer agent


def serve():
    # The protocol owns stdout; route anything a library prints there to stderr instead.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    def reply(msg):
        proto.write(json.dumps(msg) + "\n")

    def complete(oid, path=None, error=None):
        msg = {"event": "complete", "oid": oid}
        if path:
            msg["path"] = path
        if error:
            msg["error"] = {"code": 1, "message": error}
        reply(msg)

    tmp_dir = None
    for line in sys.stdin:
        req = json.loads(line)
        event = req["event"]
        if event == "init":
            reply({})
        elif event == "terminate":
            break
        elif event == "upload":
            oid, size, key = req["oid"], req["size"], object_key(req["oid"])
            try:
                if existing_sizes([key]).get(key) != size:
                    batch_bucket_files(BUCKET, add=[(req["path"], key)])
                reply({"event": "progress", "oid": oid, "bytesSoFar": size, "bytesSinceLast": size})
                complete(oid)
            except Exception as e:  # report per-object; git-lfs retries / surfaces it
                complete(oid, error=f"upload to hf://buckets/{BUCKET}/{key} failed: {e}")
        elif event == "download":
            oid, size, key = req["oid"], req["size"], object_key(req["oid"])
            dest = None
            try:
                tmp_dir = tmp_dir or lfs_tmp_dir()
                fd, dest = tempfile.mkstemp(dir=tmp_dir, prefix="hf-bucket-")
                os.close(fd)
                download_bucket_files(BUCKET, [(key, dest)], raise_on_missing_files=True)
                if sha256_file(dest) != oid:
                    os.unlink(dest)
                    raise RuntimeError("sha256 mismatch after download")
                reply({"event": "progress", "oid": oid, "bytesSoFar": size, "bytesSinceLast": size})
                complete(oid, path=dest)
            except Exception as e:
                # A failed transfer must not leave partial objects behind for every retry.
                if dest is not None and os.path.exists(dest):
                    os.unlink(dest)
                complete(oid, error=f"download of hf://buckets/{BUCKET}/{key} failed: {e}")


# ---------------------------------------------------------------- direct upload


def upload(paths):
    by_key = {}
    for p in paths:
        by_key[object_key(sha256_file(p))] = p
    present = existing_sizes(list(by_key))
    todo = [(p, k) for k, p in by_key.items() if present.get(k) != os.path.getsize(p)]
    print(f"{len(by_key)} objects, {len(by_key) - len(todo)} already in {BUCKET}, uploading {len(todo)}",
          file=sys.stderr)
    for i in range(0, len(todo), UPLOAD_BATCH):
        batch_bucket_files(BUCKET, add=todo[i:i + UPLOAD_BATCH])
        print(f"  {min(i + UPLOAD_BATCH, len(todo))}/{len(todo)}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload(sys.argv[2:])
    else:
        serve()
