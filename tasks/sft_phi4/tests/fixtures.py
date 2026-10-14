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

"""Golden + candidate loaders for the standalone grader (generated; generic).

Hermetic: imports only the vendored ``_t2j`` helpers, ``numpy``, and the standard
library. Per-instance specifics come entirely from ``golden/`` (data) and
``golden/meta.json`` (profile, thresholds, repo_config) — this file is identical
across instances.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _t2j.candidate import Candidate

# A Git LFS pointer file starts with this; detect a forgotten `git lfs pull`
# (or, in the no-LFS setup, a golden that was never regenerated) and fail loudly
# instead of comparing against a 130-byte text stub.
_LFS_POINTER_PREFIX = b"version https://git-lfs"


def _guard(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"missing golden artifact: {path}. Fetch the frozen fixtures by running "
            "`git lfs pull` (they are git-ignored)."
        )
    with path.open("rb") as handle:
        if handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
            raise RuntimeError(f"{path} is a Git LFS pointer, not real data — run `git lfs pull` or regenerate.")
    return path


def _load_json(path: Path) -> Any:
    return json.loads(_guard(path).read_text())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return {k: v for k, v in np.load(_guard(path)).items()}


@dataclass
class Golden:
    """Frozen reference artifacts + materialized thresholds for one instance."""

    dir: Path
    meta: dict
    inputs: dict
    forward: dict
    loss_artifact: dict
    gradients: dict
    lr_schedule: dict
    loss_trajectory: dict
    grad_authenticity: dict
    batch_schema: dict
    params_spec: dict
    param_norms: dict
    generation: dict | None
    train_trajectory: dict | None
    fp32_forward: dict | None
    # input-space battery: {variant: {"inputs": {...}, "forward": {...},
    # "fp32_forward": {...}}}. None when the golden froze no battery (multimodal / older
    # golden); the battery tests deselect on it (conftest ``has_battery``).
    battery: dict | None
    # spec goldens: resolved training config, tokenizer facts, and held-out
    # validation-loss result. None when absent -- each is produced by a future freezer that
    # needs the H100 checkpoint / L6 sweep (NOT the torch-free toolbox), so their tests are
    # presence-gated (conftest ``has_config_spec`` / ``has_loader_spec`` / ``has_validation``)
    # and deselect until the golden exists.
    config_spec: dict | None
    loader_spec: dict | None
    validation_loss: dict | None
    # T3 end-to-end train-run reference (torch train/eval/pre-eval + per-task bands).
    # None until the out-of-band train-run sweep freezes it; test_train_run_end_to_end is
    # marker-gated (conftest ``train_run``) and deselects until then.
    train_run: dict | None

    @property
    def training_method(self) -> str:
        return self.meta["training_method"]

    @property
    def finetuning_type(self) -> str:
        return str(self.meta.get("repo_config", {}).get("finetuning_type", "full"))

    @property
    def is_lora(self) -> bool:
        """LoRA-trained instance. Forward/loss parity stays value-exact (adapters init to
        a no-op: B=0), but per-parameter gradients and the loss trajectory depend on the
        random adapter init, which differs across torch-PEFT and JAX RNG -- those are
        graded structurally (see test_numeric / test_behavioral)."""
        return self.finetuning_type == "lora"

    def hidden_states(self) -> list:
        """Ordered per-layer hidden states frozen in forward.npz (hidden_0..hidden_N).

        Empty when the instance did not freeze hidden states. Index 0 is the
        embedding output; the final index is the post-final-norm state (HF /
        ref_jax_repo convention: length num_hidden_layers + 1).
        """
        keys = [k for k in self.forward if k.startswith("hidden_")]
        keys.sort(key=lambda k: int(k.split("_", 1)[1]))
        return [self.forward[k] for k in keys]


def _load_battery(golden_dir: Path) -> dict | None:
    """Load the input battery (``golden/battery/{variant}/``), or None if absent.

    Each variant dir carries ``inputs.npz`` (input_ids/attention_mask/labels), ``forward.npz``
    (bf16-native logits [+ SFT loss]) and ``fp32_forward.npz`` (fp32 logits [+ SFT loss])."""
    root = golden_dir / "battery"
    if not root.is_dir():
        return None
    battery: dict = {}
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        inputs_path = variant_dir / "inputs.npz"
        if not inputs_path.exists():
            continue
        entry = {"inputs": _load_npz(inputs_path)}
        for kind in ("forward", "fp32_forward"):
            path = variant_dir / f"{kind}.npz"
            if path.exists():
                entry[kind] = _load_npz(path)
        battery[variant_dir.name] = entry
    return battery or None


def load_golden(golden_dir: str | Path) -> Golden:
    d = Path(golden_dir)
    generation_path = d / "generation.npz"
    train_traj_path = d / "train_trajectory.json"
    fp32_forward_path = d / "fp32_forward.npz"
    config_spec_path = d / "config_spec.json"
    loader_spec_path = d / "loader_spec.json"
    validation_loss_path = d / "validation_loss.json"
    train_run_path = d / "train_run.json"
    return Golden(
        dir=d,
        meta=_load_json(d / "meta.json"),
        inputs=_load_npz(d / "inputs.npz"),
        forward=_load_npz(d / "forward.npz"),
        loss_artifact=_load_json(d / "loss_artifact.json"),
        gradients=_load_json(d / "gradients.json"),
        lr_schedule=_load_json(d / "lr_schedule.json"),
        loss_trajectory=_load_json(d / "loss_trajectory.json"),
        grad_authenticity=_load_json(d / "grad_authenticity.json"),
        batch_schema=_load_json(d / "batch_schema.json"),
        params_spec=_load_json(d / "params_spec.json"),
        param_norms=_load_json(d / "param_norms.json"),
        generation=_load_npz(generation_path) if generation_path.exists() else None,
        train_trajectory=_load_json(train_traj_path) if train_traj_path.exists() else None,
        fp32_forward=_load_npz(fp32_forward_path) if fp32_forward_path.exists() else None,
        battery=_load_battery(d),
        config_spec=_load_json(config_spec_path) if config_spec_path.exists() else None,
        loader_spec=_load_json(loader_spec_path) if loader_spec_path.exists() else None,
        validation_loss=_load_json(validation_loss_path) if validation_loss_path.exists() else None,
        train_run=_load_json(train_run_path) if train_run_path.exists() else None,
    )


def build_candidate(test_dir: str | Path, golden: Golden) -> Candidate:
    """Construct the candidate driver against the repo under test.

    Defaults to the committed golden ``../solution/ref_jax_repo`` so ``pytest tests/``
    self-validates the benchmark's reference. Under Harbor (and any attempt) the
    grader is pointed at the agent's produced repo by setting ``T2J_JAX_REPO`` (the
    verifier's ``test.sh`` sets it to ``/workspace/jax_repo``).

    No weights are handed over. The candidate's own
    ``model/loader.py::convert_checkpoint`` reads the HF checkpoint that
    ``repo_config["model_name_or_path"]`` names, which every loader resolves to
    ``/data/{repo_id}`` — baked into the verifier image, identical to the agent's.
    """
    test_dir = Path(test_dir).resolve()
    meta = golden.meta
    repo_config = dict(meta["repo_config"])
    repo_dir = os.environ.get("T2J_JAX_REPO") or str((test_dir / ".." / "solution" / "ref_jax_repo").resolve())
    # Reference's frozen generated sequence (prompt + greedy continuation) for the
    # teacher-forced continuation-logit probe; None for DPO/PPO (no generation).
    generation_sequences = (
        golden.generation.get("sequences")
        if golden.generation is not None and "sequences" in golden.generation
        else None
    )
    # Per-variant frozen inputs for the input-space battery; None when the golden
    # froze no battery (multimodal / older golden). The candidate re-forwards these.
    battery_inputs = (
        {name: entry["inputs"] for name, entry in golden.battery.items()}
        if golden.battery else None
    )
    return Candidate(
        str(Path(repo_dir).resolve()),
        inputs=golden.inputs,
        repo_config=repo_config,
        training_method=meta["training_method"],
        seed=int(meta["seed"]),
        loss_trajectory_steps=int(meta["loss_trajectory_steps"]),
        generation_max_new_tokens=int(meta["generation_max_new_tokens"]),
        generation_sequences=generation_sequences,
        battery_inputs=battery_inputs,
        compare_hidden_states=bool(meta["compare_hidden_states"]),
        # Hard wall-clock budget for the repo's own training loop.
        # Per instance via instance.yaml -> meta; a generous default covers
        # pre-feature goldens whose meta predates the knob.
        train_timeout_s=float(meta.get("train_timeout_s") or 1800.0),
        # Hard wall-clock hang-guard for the equivalence-probe worker (_ensure_run:
        # forward / fp32 / hidden / loss / gradient / battery). Graded one task per GPU
        # with ample CPU, every reference -- incl. the multi-billion-param ones
        # (mistral-7B, phi4, starcoder2) -- completes well under 1800s.
        # Per instance via meta ("sandbox_timeout_s"); env override T2J_SANDBOX_TIMEOUT_S
        # for ad-hoc grade runs; default covers goldens whose meta predates the knob.
        timeout=float(
            os.environ.get("T2J_SANDBOX_TIMEOUT_S")
            or meta.get("sandbox_timeout_s")
            or 1800.0
        ),
    )
