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

"""Static, method-keyed expected-golden manifest.

Applicability deselection (conftest) is driven by golden-file presence: a check whose
golden the freeze never produced is deselected (kept out of numerator AND denominator).
That is correct for a golden the method genuinely has no analogue for -- but it means a
freeze BUG that silently drops a required golden reads as "not applicable" instead of
"coverage regression", and the continuous pass rate never notices.

This manifest is the ground truth for what a CORRECT freeze must emit per method. It is
static (no torch, no candidate run), derived from the intersection of what every instance
of each method actually froze. ``test_golden_manifest`` hard-fails, listing any
required-but-missing token, so a dropped golden is a loud FAIL rather than a silent
deselect; the golden-presence ``_APPLICABLE`` predicates read this instead of raw
existence, so a required-but-missing golden is COLLECTED (and fails) rather than skipped.

A token is either a filename (``forward.npz``) or ``<npz-stem>:<array>`` /
``<json-stem>:<key>`` for a within-file requirement (``fp32_forward:loss``).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

# Present in every instance of every method (15 files).
_COMMON: frozenset[str] = frozenset({
    "batch_schema.json",
    "config_spec.json",
    "forward.npz",
    "fp32_forward.npz",
    "grad_authenticity.json",
    "gradients.json",
    "inputs.npz",
    "interface.json",
    "loss_artifact.json",
    "loss_trajectory.json",
    "lr_schedule.json",
    "meta.json",
    "param_norms.json",
    "params_spec.json",
    # within-file: the fp32 forward always carries logits; the grad summary always
    # carries per-tensor grads. Both gate a check that applies to all three methods.
    "fp32_forward:logits",
    "gradients:grads",
})

# Method-specific additions, measured from the current 45 goldens:
#   sft: full generation + train loop + forward/fp32 scalar losses.
#   dpo: train loop + forward scalar loss (fp32 loss is NOT frozen for dpo/ppo).
#   ppo: neither train_trajectory nor generation nor a frozen scalar loss.
REQUIRED_GOLDENS: dict[str, frozenset[str]] = {
    "sft": _COMMON | frozenset({
        "generation.npz",
        "train_trajectory.json",
        "forward:loss",
        "fp32_forward:loss",
    }),
    "dpo": _COMMON | frozenset({
        "train_trajectory.json",
        "forward:loss",
    }),
    "ppo": _COMMON,
}

# Goldens that legitimately vary by instance (NOT required); listed so the coverage
# vector can report which optional artifacts an instance actually froze.
OPTIONAL_GOLDENS: frozenset[str] = frozenset({
    "generation.npz",          # sft-only, but harmless to probe everywhere
    "train_trajectory.json",
    "reference_units.json",    # bloom per-function suite
    "reference_units.npz",
    "train_run.json",          # bloom end-to-end suite
    "ppo_units.json",          # PPO update-mechanics reference units (ppo-only)
    "ppo_units.npz",
    # spec goldens still produced by a FUTURE freezer (need the H100 checkpoint / L6
    # sweep), presence-gated -> their checks deselect until frozen. Listed so the coverage
    # vector reports them once they exist; NOT required. (config_spec.json is now REQUIRED
    # instead -- config-DEFAULT parity needs no checkpoint and is frozen for all 45 by
    # the freezer; see _COMMON above.)
    "loader_spec.json",
    "validation_loss.json",
})


def _npz_arrays(path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            return {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
    except (OSError, zipfile.BadZipFile):
        return set()


def _has_token(golden_dir: Path, token: str) -> bool:
    """True if a manifest token (file, ``npz:array`` or ``json:key``) is present."""
    if ":" not in token:
        return (golden_dir / token).exists()
    stem, key = token.split(":", 1)
    npz = golden_dir / f"{stem}.npz"
    if npz.exists():
        return key in _npz_arrays(npz)
    js = golden_dir / f"{stem}.json"
    if js.exists():
        try:
            data = json.loads(js.read_text())
        except (OSError, ValueError):
            return False
        return bool(data.get(key))
    return False


def missing_required(golden_dir: Path, method: str) -> list[str]:
    """Required manifest tokens (for ``method``) absent from ``golden_dir``, sorted.

    Unknown method -> empty list (nothing asserted): the manifest only speaks for the
    three benchmark methods, and an unrecognised method is a config error the method-name
    gate elsewhere already surfaces.
    """
    required = REQUIRED_GOLDENS.get(str(method).lower())
    if not required:
        return []
    return sorted(tok for tok in required if not _has_token(golden_dir, tok))


def coverage_vector(golden_dir: Path, method: str) -> dict:
    """Per-instance coverage summary for the eval summary (coverage vector)."""
    required = REQUIRED_GOLDENS.get(str(method).lower(), frozenset())
    return {
        "method": method,
        "required_total": len(required),
        "required_missing": missing_required(golden_dir, method),
        "optional_present": sorted(t for t in OPTIONAL_GOLDENS if _has_token(golden_dir, t)),
    }


def consistency_problems(golden_dir: Path) -> list[str]:
    """Cross-artifact contradictions the per-file presence list cannot express.

    Currently one: a ``meta.json`` saying ``finetuning_type: "lora"`` beside a
    ``gradients.json`` frozen from an unwrapped model. That is a stale artifact, not a
    port bug, but it surfaces as an opaque adapter live/dead mismatch inside
    ``test_gradient_per_param``. Empty list means consistent.
    """
    problems: list[str] = []
    try:
        meta = json.loads((golden_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return problems

    finetuning_type = str(meta.get("repo_config", {}).get("finetuning_type", "")).lower()
    if finetuning_type == "lora":
        try:
            grads = json.loads((golden_dir / "gradients.json").read_text()).get("grads", {})
        except (OSError, ValueError):
            return problems
        if grads and not any("lora_" in name for name in grads):
            problems.append(
                f"meta.finetuning_type is 'lora' but gradients.json holds {len(grads)} "
                "base leaves and no lora_A/lora_B keys -- these goldens predate the "
                "all-LoRA re-baseline; re-freeze from the reference implementation"
            )
    return problems
