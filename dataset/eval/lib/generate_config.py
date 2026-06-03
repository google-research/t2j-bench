# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automatic evaluator config generation and optional execution."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import EvalConfig, load_eval_config
from .discovery import repo_imports
from .runner import run_evaluation
from .utils import ensure_dir, jsonable


MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "gemma3": {
        "model_name_or_path": "google/gemma-3-4b-it",
        "template": "gemma3",
    },
    "qwen3_vl": {
        "model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
        "template": "qwen3_vl_nothink",
    },
    "llava_next": {
        "model_name_or_path": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "template": "llava_next",
    },
}


def _infer_model_family(reference_repo: Path, candidate_repo: Path) -> str:
    haystack = "/".join(part.lower() for part in (*reference_repo.parts, *candidate_repo.parts))
    for family in MODEL_REGISTRY:
        if family in haystack:
            return family
    raise ValueError(
        "Could not infer model family from repo paths. "
        "Use a repo path containing one of: gemma3, qwen3_vl, llava_next."
    )


def _infer_framework(repo_dir: Path, fallback: str) -> str:
    haystack = "/".join(part.lower() for part in repo_dir.parts)
    if "torch" in haystack:
        return "pytorch"
    if "jax" in haystack or "flax" in haystack:
        return "flax"
    loader_path = repo_dir / "model" / "loader.py"
    if loader_path.exists():
        text = loader_path.read_text()
        lowered = text.lower()
        if "import jax" in lowered or "flax" in lowered or ".apply(" in text:
            return "flax"
        if "import torch" in lowered:
            return "pytorch"
    return fallback


def _repo_defaults(repo_dir: Path) -> dict[str, Any]:
    with repo_imports(str(repo_dir)) as modules:
        args = modules.config.get_train_args({})
    result: dict[str, Any] = {}
    if not isinstance(args, tuple):
        return result
    for item in args:
        if is_dataclass(item):
            result.update(asdict(item))
    return result


def _default_precision(defaults: dict[str, Any]) -> str:
    if defaults.get("bf16"):
        return "bf16_compare"
    if defaults.get("fp16"):
        return "fp16_compare"
    return "bf16_compare"


def _default_eval_examples(training_method: str) -> list[dict[str, Any]]:
    if training_method == "sft":
        return [
            {
                "_prompt": [{"role": "user", "content": "Describe the sky in three words."}],
                "_response": [{"role": "assistant", "content": "Blue, bright, calm."}],
                "_system": "",
                "_tools": "",
                "_images": None,
                "_videos": None,
                "_audios": None,
            }
        ]
    if training_method == "dpo":
        return [
            {
                "_prompt": [{"role": "user", "content": "Say hello in one short sentence."}],
                "_response": [
                    {"role": "assistant", "content": "Hello there!"},
                    {"role": "assistant", "content": "Goodbye forever."},
                ],
                "_system": "",
                "_tools": "",
                "_images": None,
                "_videos": None,
                "_audios": None,
            }
        ]
    if training_method == "ppo":
        return [
            {
                "_prompt": [{"role": "user", "content": "Say hello in one short sentence."}],
                "_response": [{"role": "assistant", "content": "Hello."}],
                "_system": "",
                "_tools": "",
                "_images": None,
                "_videos": None,
                "_audios": None,
            }
        ]
    raise ValueError(f"Unsupported training method: {training_method}")


def _coerce_device(index: int | None, fallback: str) -> str:
    if index is None:
        return fallback
    return f"cuda:{index}"


def _detect_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def build_generated_config(
    training_method: str,
    reference_repo: str,
    candidate_repo: str,
    *,
    reference_name: str = "torch_orig",
    candidate_name: str = "jax_repo",
    precision_profile: str | None = None,
    model_name_or_path: str | None = None,
    output_dir: str | None = None,
    reference_device: str | None = None,
    candidate_device: str | None = None,
    backend_device: str | None = None,
    batch_size: int = 1,
    max_eval_examples: int = 1,
    loss_trajectory_steps: int = 2,
    generation_max_new_tokens: int = 8,
    compare_hidden_states: bool = False,
) -> dict[str, Any]:
    reference_path = Path(reference_repo).resolve()
    candidate_path = Path(candidate_repo).resolve()
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    if training_method not in {"sft", "dpo", "ppo"}:
        raise ValueError("training_method must be one of sft, dpo, ppo.")

    family = _infer_model_family(reference_path, candidate_path)
    registry = MODEL_REGISTRY[family]
    ref_defaults = _repo_defaults(reference_path)
    precision_profile = precision_profile or _default_precision(ref_defaults)

    gpu_count = _detect_gpu_count()
    if gpu_count > 0:
        reference_device = reference_device or _coerce_device(0, "cuda:0")
        candidate_device = candidate_device or _coerce_device(1 if gpu_count > 1 else 0, "cuda:0")
        if _infer_framework(candidate_path, "flax") == "flax":
            default_backend_index = 2 if gpu_count > 2 else (1 if gpu_count > 1 else 0)
            backend_device = backend_device or _coerce_device(default_backend_index, candidate_device)
    else:
        reference_device = reference_device or "cpu"
        candidate_device = candidate_device or "cpu"
        backend_device = backend_device or "cpu"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    config_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (Path.cwd() / "generated_outputs" / f"{training_method}_{family}_{timestamp}").resolve()
    )

    shared_config: dict[str, Any] = {
        "model_name_or_path": model_name_or_path or ref_defaults.get("model_name_or_path") or registry["model_name_or_path"],
        "template": ref_defaults.get("template") or registry["template"],
        "learning_rate": ref_defaults.get("learning_rate", 1.0e-6),
        "eval_examples": _default_eval_examples(training_method),
    }
    for key in (
        "finetuning_type",
        "freeze_vision_tower",
        "freeze_multi_modal_projector",
        "pref_loss",
        "pref_beta",
        "gamma",
        "lam",
        "cliprange",
        "cliprange_value",
        "vf_coef",
        "kl_coef",
    ):
        if key in ref_defaults and ref_defaults[key] is not None:
            shared_config[key] = ref_defaults[key]

    config = {
        "training_method": training_method,
        "reference_impl": reference_name,
        "candidate_impl": candidate_name,
        "precision_profile": precision_profile,
        "runtime_profile": "single_gpu",
        "output_dir": str(config_output_dir),
        "batch_size": batch_size,
        "max_eval_examples": max_eval_examples,
        "loss_trajectory_steps": loss_trajectory_steps,
        "generation_max_new_tokens": generation_max_new_tokens,
        "compare_hidden_states": compare_hidden_states,
        "shared_config": shared_config,
        "implementations": [
            {
                "name": reference_name,
                "framework": _infer_framework(reference_path, "pytorch"),
                "repo_dir": str(reference_path),
                "device": reference_device,
            },
            {
                "name": candidate_name,
                "framework": _infer_framework(candidate_path, "flax"),
                "repo_dir": str(candidate_path),
                "device": candidate_device,
                "backend_device": backend_device,
            },
        ],
    }
    return config


def write_generated_config(config: dict[str, Any], path: str | None = None) -> str:
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(str(output_dir))
    config_path = Path(path).resolve() if path is not None else output_dir / "generated_eval_config.json"
    config_path.write_text(json.dumps(jsonable(config), indent=2, sort_keys=True))
    return str(config_path)


def generate_and_run(
    training_method: str,
    reference_repo: str,
    candidate_repo: str,
    *,
    config_out: str | None = None,
    run: bool = True,
    **kwargs: Any,
) -> tuple[str, EvalConfig | None, dict[str, Any] | None]:
    raw_config = build_generated_config(training_method, reference_repo, candidate_repo, **kwargs)
    config_path = write_generated_config(raw_config, config_out)
    if not run:
        return config_path, None, None
    config = load_eval_config(config_path)
    report = run_evaluation(config)
    return config_path, config, report.summary()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an evaluator config from repo paths and optionally run a full evaluation."
    )
    parser.add_argument("training_method", choices=["sft", "dpo", "ppo"])
    parser.add_argument("reference_repo", help="Path to the PyTorch reference repo.")
    parser.add_argument("candidate_repo", help="Path to the JAX repo.")
    parser.add_argument("--config-out", help="Where to write the generated config.")
    parser.add_argument("--output-dir", help="Evaluation output directory.")
    parser.add_argument("--reference-name", default="torch_orig")
    parser.add_argument("--candidate-name", default="jax_repo")
    parser.add_argument("--precision-profile", choices=["fp16_compare", "bf16_compare"])
    parser.add_argument("--model-name-or-path", help="Override the inferred Hugging Face model name or local checkpoint path.")
    parser.add_argument("--reference-device")
    parser.add_argument("--candidate-device")
    parser.add_argument("--backend-device")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-examples", type=int, default=1)
    parser.add_argument("--loss-trajectory-steps", type=int, default=2)
    parser.add_argument("--generation-max-new-tokens", type=int, default=8)
    parser.add_argument("--compare-hidden-states", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="Only generate the config; do not run evaluation.")
    args = parser.parse_args()

    config_path, _, summary = generate_and_run(
        args.training_method,
        args.reference_repo,
        args.candidate_repo,
        config_out=args.config_out,
        run=not args.no_run,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        precision_profile=args.precision_profile,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        reference_device=args.reference_device,
        candidate_device=args.candidate_device,
        backend_device=args.backend_device,
        batch_size=args.batch_size,
        max_eval_examples=args.max_eval_examples,
        loss_trajectory_steps=args.loss_trajectory_steps,
        generation_max_new_tokens=args.generation_max_new_tokens,
        compare_hidden_states=args.compare_hidden_states,
    )
    payload: dict[str, Any] = {"config_path": config_path}
    if summary is not None:
        payload["summary"] = summary
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
