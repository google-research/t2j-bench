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

"""Bake one model's weights into ``/data/{repo_id}`` at image build time.

Task code loads weights from ``/data/{repo_id}`` and never downloads. This script
fetches only the files the loader opens (the same selection as
``model/loader.py::_iter_checkpoint_files`` -- keep the two in sync), and rewrites a
``.bin``-only checkpoint (falcon) as ``.safetensors``, since the JAX side reads
safetensors only.

Usage:
    python prepare_model.py --repo bigscience/bloom-560m --out /data
    python prepare_model.py --repo bigscience/bloom-560m --out /data --aux-only
    python prepare_model.py --repo bigscience/bloom-560m --out /data --from-cache
    python prepare_model.py --repo tiiuae/falcon-rw-1b  --out /data --normalize-only
"""

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Weight/export formats; only the checkpoint the loader opens is kept. `.pt` is not listed
# because Qwen2.5-Omni needs `spk_dict.pt`.
_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".gguf",
    ".msgpack",
    ".h5",
    ".onnx",
    ".onnx_data",
    ".pth",
    ".tflite",
}

_SAFE_INDEX = "model.safetensors.index.json"
_PT_INDEX = "pytorch_model.bin.index.json"

# Record of a `.bin` -> `.safetensors` rewrite, written only when one happens.
_NORMALIZED_RECORD = ".t2j_normalized.json"


def select_weight_files(files: list[str], read_json) -> list[str]:
    """The checkpoint files ``_iter_checkpoint_files`` would open, in its order.

    ``read_json`` fetches and parses one repo-relative file, so the same selection runs
    against the hub (``--repo``) or against a cached snapshot (``--from-cache``).
    """
    present = set(files)
    for index_name in (_SAFE_INDEX, _PT_INDEX):
        if index_name in present:
            return sorted(set(read_json(index_name)["weight_map"].values()))
    for file_name in ("model.safetensors", "pytorch_model.bin"):
        if file_name in present:
            return [file_name]
    raise SystemExit(
        f"ERROR: no checkpoint found among {sorted(present)[:20]} -- "
        "model/loader.py::_iter_checkpoint_files would raise here too."
    )


def select_aux_files(files: list[str], weights: list[str]) -> list[str]:
    """Config/tokenizer/processor files: every top-level non-weight file."""
    chosen = set(weights)
    return sorted(
        name
        for name in files
        if name not in chosen
        and "/" not in name
        and Path(name).suffix not in _WEIGHT_SUFFIXES
    )


def safetensors_name(bin_name: str) -> str:
    """``pytorch_model-00002-of-00003.bin`` -> ``model-00002-of-00003.safetensors``."""
    stem = Path(bin_name).stem
    if not stem.startswith("pytorch_model"):
        raise SystemExit(
            f"ERROR: unexpected .bin checkpoint name {bin_name!r} -- prepare_model.py only "
            "knows how to rename pytorch_model*.bin. Add the rule rather than guessing."
        )
    return f"model{stem[len('pytorch_model'):]}.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_one(src: Path, dst: Path) -> dict:
    """Rewrite one ``.bin`` shard as ``.safetensors``, preserving the state dict exactly.

    torch is imported here so the other code paths work without it.
    """
    import torch
    from safetensors.torch import save_file

    try:
        state = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, ValueError):
        state = torch.load(src, map_location="cpu", weights_only=True)  # pre-zip format

    tensors: dict[str, "torch.Tensor"] = {}
    owner: dict[int, str] = {}
    ties: list[str] = []
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(
                f"ERROR: {src.name}[{key!r}] is a {type(tensor).__name__}, not a tensor -- "
                "safetensors carries tensors only."
            )
        storage_id = tensor.untyped_storage().data_ptr()
        if storage_id in owner:
            # safetensors rejects shared storage (e.g. tied lm_head/embedding); copy the
            # tensor so the key set stays the same.
            tensor = tensor.clone()
            ties.append(f"{key} <- {owner[storage_id]}")
        else:
            owner[storage_id] = key
        tensors[key] = tensor.contiguous()

    # `format: pt` is what the hub's own safetensors carry; transformers looks for it.
    save_file(tensors, str(dst), metadata={"format": "pt"})
    dtypes = collections.Counter(str(t.dtype).removeprefix("torch.") for t in tensors.values())
    return {
        "source": src.name,
        "source_sha256": _sha256(src),
        "target": dst.name,
        "target_sha256": _sha256(dst),
        "tensors": len(tensors),
        "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
        "dtypes": dict(sorted(dtypes.items())),
        "materialized_ties": ties,
    }


def normalize_to_safetensors(
    dest: Path, weights: list[str], aux: list[str]
) -> tuple[list[str], list[str]]:
    """Rewrite a ``.bin`` checkpoint in ``dest`` as ``.safetensors``; return the new lists.

    A no-op when there is no ``.bin`` checkpoint.
    """
    binaries = [name for name in weights if name.endswith(".bin")]
    if not binaries:
        return weights, aux

    renamed = {name: safetensors_name(name) for name in binaries}
    records = []
    for name, out_name in renamed.items():
        print(f"  normalize {name} -> {out_name}  (no safetensors upstream)", flush=True)
        record = _convert_one(dest / name, dest / out_name)
        for tie in record["materialized_ties"]:
            print(f"             materialized tied weight {tie}", flush=True)
        records.append(record)
        # Under --from-cache this only removes the hardlink, not the cached file.
        (dest / name).unlink()
    weights = sorted(renamed.get(name, name) for name in weights)

    # Sharded checkpoints: rewrite the index to point at the new shard names.
    pt_index = dest / _PT_INDEX
    if pt_index.is_file():
        with open(pt_index) as handle:
            payload = json.load(handle)
        payload["weight_map"] = {
            key: renamed[shard] for key, shard in payload["weight_map"].items()
        }
        # Recomputed: copied tied weights add bytes.
        payload.setdefault("metadata", {})["total_size"] = sum(r["bytes"] for r in records)
        with open(dest / _SAFE_INDEX, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        pt_index.unlink()
        aux = sorted((set(aux) - {_PT_INDEX}) | {_SAFE_INDEX})

    with open(dest / _NORMALIZED_RECORD, "w") as handle:
        json.dump({"tool": "base_image/prepare_model.py", "converted": records}, handle, indent=2)
        handle.write("\n")
    return weights, aux


def _local_listing(dest: Path) -> list[str]:
    """Top-level file names in ``dest`` (all the loader looks at)."""
    if not dest.is_dir():
        raise SystemExit(f"ERROR: {dest} does not exist -- nothing to inspect.")
    return sorted(path.name for path in dest.iterdir() if path.is_file())


def normalize_in_place(dest: Path) -> tuple[list[str], list[str]]:
    """``--normalize-only``: run the rewrite over a ``/data/{repo_id}`` already on disk.

    Repairs an existing image without re-downloading. Idempotent.
    """
    files = _local_listing(dest)

    def read_json(name: str) -> dict:
        with open(dest / name) as handle:
            return json.load(handle)

    weights = select_weight_files(files, read_json)
    aux = select_aux_files(files, weights)
    return normalize_to_safetensors(dest, weights, aux)


def assert_resolves_to_safetensors(dest: Path) -> None:
    """Fail unless the checkpoint on disk resolves to safetensors (the JAX loader has no
    ``.bin`` fallback)."""

    def read_json(name: str) -> dict:
        with open(dest / name) as handle:
            return json.load(handle)

    stragglers = [
        name
        for name in select_weight_files(_local_listing(dest), read_json)
        if not name.endswith(".safetensors")
    ]
    if stragglers:
        raise SystemExit(
            f"ERROR: {dest} still resolves to {stragglers}. normalize_to_safetensors did "
            "not fire, and the JAX loader reads safetensors only -- do not ship this image."
        )


def _fmt(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.2f} GB" if num_bytes >= 1e9 else f"{num_bytes / 1e6:.1f} MB"


def _report(dest: Path, weights: list[str], aux: list[str]) -> None:
    def total(names):
        return sum(os.path.getsize(dest / n) for n in names if (dest / n).is_file())

    for label, names in (("weights", weights), ("aux", aux)):
        listing = ", ".join(names) if len(names) <= 6 else f"{', '.join(names[:6])}, +{len(names) - 6} more"
        print(f"  {label:8s} {len(names):3d} file(s), {_fmt(total(names)):>9s}  [{listing}]", flush=True)


def download(repo: str, revision: str, dest: Path, aux_only: bool) -> tuple[list[str], list[str]]:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    files = HfApi().list_repo_files(repo, revision=revision)

    def read_json(name: str) -> dict:
        with open(hf_hub_download(repo_id=repo, filename=name, revision=revision)) as handle:
            return json.load(handle)

    weights = [] if aux_only else select_weight_files(files, read_json)
    aux = select_aux_files(files, weights)
    snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=weights + aux,
        local_dir=str(dest),
    )
    # Drop huggingface_hub's download metadata so it doesn't end up in the image.
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    return weights, aux


def seed_from_cache(repo: str, revision: str, dest: Path, aux_only: bool) -> tuple[list[str], list[str]]:
    """Populate ``dest`` from the local HF cache, hardlinking where possible (``--host-seed``)."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=repo, revision=revision, local_files_only=True))
    files = sorted(
        str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file()
    )

    def read_json(name: str) -> dict:
        with open(snapshot / name) as handle:
            return json.load(handle)

    weights = [] if aux_only else select_weight_files(files, read_json)
    aux = select_aux_files(files, weights)
    for name in weights + aux:
        src = (snapshot / name).resolve()  # cache entries are symlinks into blobs/
        dst = dest / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)  # different device
    return weights, aux


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="hub repo id, e.g. bigscience/bloom-560m")
    parser.add_argument("--out", default="/data", help="root to write {repo_id} under (default: /data)")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--aux-only",
        action="store_true",
        help="config/tokenizer/processor only, no checkpoint (verifier images)",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="hardlink from the local HF cache instead of downloading (host seeding)",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="fetch nothing; rewrite an existing {out}/{repo} .bin checkpoint as safetensors",
    )
    args = parser.parse_args()
    if args.normalize_only and (args.aux_only or args.from_cache):
        parser.error("--normalize-only fetches nothing, so --aux-only/--from-cache mean nothing")

    dest = Path(args.out) / args.repo
    if args.normalize_only:
        print(f"{args.repo} -> {dest} (normalize-only, no fetch)", flush=True)
        weights, aux = normalize_in_place(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        kind = "aux-only" if args.aux_only else "full"
        source = "cache" if args.from_cache else "hub"
        print(f"{args.repo} @ {args.revision} -> {dest} ({kind}, from {source})", flush=True)
        fetch = seed_from_cache if args.from_cache else download
        weights, aux = fetch(args.repo, args.revision, dest, args.aux_only)
        weights, aux = normalize_to_safetensors(dest, weights, aux)

    if not args.aux_only:
        assert_resolves_to_safetensors(dest)
    _report(dest, weights, aux)

    missing = [n for n in weights + aux if not (dest / n).is_file()]
    if missing:
        print(f"ERROR: selected but not present after fetch: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
