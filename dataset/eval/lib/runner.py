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

"""Top-level staged evaluator."""

from __future__ import annotations

import copy
import gc
import json
from dataclasses import asdict
from typing import Any

from .config import EvalConfig
from .metrics import compare_arrays, compare_logits, thresholds_for_profile
from .results import CheckResult, EvalReport
from .runtime import RepoRuntime
from .utils import (
    ensure_dir,
    failure_kind_from_exception,
    is_nonfinite,
    jsonable,
    summarize_tensor,
    to_numpy,
    with_contract_hint,
)


def _tagged_summary(summary: str, source: str | None) -> str:
    if source is None:
        return summary
    prefix = f"[{source}] "
    return summary if summary.startswith(prefix) else prefix + summary


def _pass(
    name: str,
    stage: str,
    summary: str,
    details: dict[str, Any] | None = None,
    source: str | None = None,
) -> CheckResult:
    return CheckResult(name=name, stage=stage, passed=True, summary=summary, details=details or {}, source=source)


def _fail(
    name: str,
    stage: str,
    summary: str,
    details: dict[str, Any] | None = None,
    failure_kind: str | None = None,
    source: str | None = None,
) -> CheckResult:
    return CheckResult(
        name=name,
        stage=stage,
        passed=False,
        summary=_tagged_summary(summary, source),
        details=details or {},
        failure_kind=failure_kind,
        source=source,
    )


def _blocked(stage: str, name: str, reason: str, source: str | None = None) -> CheckResult:
    return _fail(name=name, stage=stage, summary=f"blocked: {reason}", failure_kind="blocked", source=source)


_REPORT_SEQUENCE_LIMIT = 32
_REPORT_NDARRAY_SUMMARY_THRESHOLD = 256


def _report_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _report_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_report_safe_value(item) for item in value[:_REPORT_SEQUENCE_LIMIT]]
    if isinstance(value, list):
        if len(value) > _REPORT_SEQUENCE_LIMIT:
            return {
                "kind": "list",
                "length": len(value),
                "sample": [_report_safe_value(item) for item in value[:_REPORT_SEQUENCE_LIMIT]],
            }
        return [_report_safe_value(item) for item in value]
    try:
        arr = to_numpy(value)
    except Exception:
        return jsonable(value)
    if arr.size > _REPORT_NDARRAY_SUMMARY_THRESHOLD or arr.ndim >= 3:
        return {"kind": "tensor_summary", **summarize_tensor(arr)}
    return jsonable(arr)


def _report_safe_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _report_safe_value(value) for key, value in artifacts.items()}


def _leaf_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return tuple()
    return tuple(int(item) for item in shape)


def _release_runtime(runtime: RepoRuntime | None) -> None:
    if runtime is None:
        return
    for attribute in ("model", "ref_model", "batch", "params", "ref_params", "train_dataset", "dataset_module"):
        if hasattr(runtime, attribute):
            try:
                setattr(runtime, attribute, None)
            except Exception:
                pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _check_reference(runtime: RepoRuntime) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        params = runtime.flat_parameters()
        results.append(_pass("params_nonempty", "reference", f"{len(params)} parameter tensors exported"))
    except Exception as exc:
        return [_fail("params_nonempty", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    try:
        batch = runtime.batch_arrays()
        if not batch:
            raise ValueError("collator produced an empty batch")
        results.append(_pass("data_pipeline", "reference", f"batch keys: {sorted(batch)}"))
    except Exception as exc:
        return results + [_fail("data_pipeline", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    try:
        forward = runtime.forward_outputs()
        if is_nonfinite(forward["logits"]):
            raise ValueError("forward logits contain non-finite values")
        if forward["loss"] is not None and not float(forward["loss"]) == float(forward["loss"]):
            raise ValueError("forward loss is non-finite")
        results.append(_pass("forward", "reference", "forward pass is finite"))
    except Exception as exc:
        return results + [_fail("forward", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    try:
        runtime.derive_runtime_contract()
    except Exception as exc:
        return results + [_fail("runtime_contract", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    try:
        gradients = runtime.gradient_artifact()
        if not float(gradients["loss"]) == float(gradients["loss"]):
            raise ValueError("gradient loss is non-finite")
        results.append(_pass("gradients", "reference", "gradients are finite", {"grad_norm": gradients["grad_norm"]}))
    except Exception as exc:
        return results + [_fail("gradients", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    try:
        lr = runtime.lr_schedule_artifact()
        if any((value != value) for value in lr["learning_rates"]):
            raise ValueError("lr schedule contains non-finite values")
        results.append(_pass("lr_schedule", "reference", "lr schedule is finite"))
    except Exception as exc:
        return results + [_fail("lr_schedule", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    try:
        loss_artifact = runtime.loss_artifact()
        if not float(loss_artifact["loss"]) == float(loss_artifact["loss"]):
            raise ValueError("loss artifact is non-finite")
        results.append(_pass("method_loss", "reference", f"{runtime.eval_config.training_method} loss artifact is finite"))
    except Exception as exc:
        return results + [_fail("method_loss", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    if runtime.eval_config.training_method == "sft":
        try:
            generation = runtime.generation_artifact()
            if generation.get("supported"):
                results.append(_pass("generation", "reference", "generation succeeded"))
            else:
                results.append(_pass("generation", "reference", generation["reason"]))
        except Exception as exc:
            return results + [_fail("generation", "reference", with_contract_hint(str(exc), exc), failure_kind=failure_kind_from_exception(exc), source="torch")]
    if runtime.eval_config.training_method in {"dpo", "ppo"} and runtime.ref_model is None:
        results.append(_pass("reference_model", "reference", "reference model not required or adapter-disabled"))
    elif runtime.eval_config.training_method in {"dpo", "ppo"}:
        results.append(_pass("reference_model", "reference", "reference model loaded"))
    return results


def _compare_parameter_trees(reference: RepoRuntime, candidate: RepoRuntime) -> CheckResult:
    ref = reference.flat_parameters()
    cand = candidate.flat_parameters()
    ref_keys = set(ref)
    cand_keys = set(cand)
    missing = sorted(ref_keys - cand_keys)
    extra = sorted(cand_keys - ref_keys)
    shape_mismatches = {
        key: {"reference": list(_leaf_shape(ref[key])), "candidate": list(_leaf_shape(cand[key]))}
        for key in sorted(ref_keys & cand_keys)
        if _leaf_shape(ref[key]) != _leaf_shape(cand[key])
    }
    passed = not missing and not extra and not shape_mismatches
    summary = "parameter names and shapes match" if passed else "parameter trees diverged"
    return CheckResult(
        name="weight_loading",
        stage="spec",
        passed=passed,
        summary=_tagged_summary(summary, None if passed else "jax"),
        details={"missing": missing[:32], "extra": extra[:32], "shape_mismatches": shape_mismatches},
        failure_kind=None if passed else "shape_mismatch",
        source=None if passed else "jax",
    )


def _compare_batch_schema(reference: RepoRuntime, candidate: RepoRuntime) -> CheckResult:
    ref = reference.batch_arrays()
    cand = candidate.batch_arrays()
    details: dict[str, Any] = {"missing": [], "extra": [], "shape_mismatches": {}, "dtype_mismatches": {}}
    ref_keys = set(ref)
    cand_keys = set(cand)
    details["missing"] = sorted(ref_keys - cand_keys)
    details["extra"] = sorted(cand_keys - ref_keys)
    for key in sorted(ref_keys & cand_keys):
        if ref[key].shape != cand[key].shape:
            details["shape_mismatches"][key] = {"reference": list(ref[key].shape), "candidate": list(cand[key].shape)}
        if str(ref[key].dtype) != str(cand[key].dtype):
            details["dtype_mismatches"][key] = {"reference": str(ref[key].dtype), "candidate": str(cand[key].dtype)}
    passed = not any(details.values())
    return CheckResult(
        name="data_pipeline",
        stage="spec",
        passed=passed,
        summary=_tagged_summary("batch schema matches" if passed else "batch schema diverged", None if passed else "jax"),
        details=details,
        failure_kind=None if passed else "schema_mismatch",
        source=None if passed else "jax",
    )


def _numeric_check(name: str, metrics: dict[str, Any], config: EvalConfig, *, logits: bool = False) -> CheckResult:
    thresholds = thresholds_for_profile(config.precision_profile)
    same_shape = metrics.get("same_shape", True)
    finite = metrics.get("finite", True)
    if not same_shape:
        return _fail(name, "numeric", "tensor shapes differ", metrics, "shape_mismatch", source="jax")
    if not finite:
        return _fail(name, "numeric", "non-finite tensors", metrics, "nonfinite", source="jax")
    passed = (
        metrics.get("max_abs_diff", 0.0) <= thresholds.abs_tol
        and metrics.get("max_rel_diff", 0.0) <= thresholds.rel_tol
        and metrics.get("cosine_similarity", 1.0) >= thresholds.cosine_floor
    )
    if logits:
        passed = passed and metrics.get("token_kl", 0.0) <= thresholds.max_kl
    summary = "numeric parity within thresholds" if passed else "numeric parity exceeded thresholds"
    return CheckResult(
        name=name,
        stage="numeric",
        passed=passed,
        summary=_tagged_summary(summary, None if passed else "jax"),
        details=metrics,
        failure_kind=None if passed else "numeric_mismatch",
        source=None if passed else "jax",
    )


def _compare_forward(reference: RepoRuntime, candidate: RepoRuntime, report: EvalReport) -> list[CheckResult]:
    ref = reference.forward_outputs()
    cand = candidate.forward_outputs()
    report.artifacts["reference_forward"] = jsonable(ref)
    report.artifacts["candidate_forward"] = jsonable(cand)
    results = [_numeric_check("forward_logits", compare_logits(ref["logits"], cand["logits"]), reference.eval_config, logits=True)]
    if reference.eval_config.compare_hidden_states and ref.get("hidden_states") and cand.get("hidden_states"):
        layer_metrics = []
        for ref_layer, cand_layer in zip(ref["hidden_states"], cand["hidden_states"], strict=False):
            layer_metrics.append(compare_arrays(ref_layer, cand_layer))
        worst = max(layer_metrics, key=lambda item: item.get("max_abs_diff", 0.0)) if layer_metrics else {}
        results.append(_numeric_check("forward_hidden_states", worst, reference.eval_config))
    if ref["loss"] is not None and cand["loss"] is not None:
        results.append(_numeric_check("forward_loss", compare_arrays([ref["loss"]], [cand["loss"]]), reference.eval_config))
    return results


def _compare_loss_artifacts(reference: RepoRuntime, candidate: RepoRuntime, report: EvalReport) -> list[CheckResult]:
    ref = reference.loss_artifact()
    cand = candidate.loss_artifact()
    report.artifacts["reference_loss_artifact"] = jsonable(ref)
    report.artifacts["candidate_loss_artifact"] = jsonable(cand)
    checks = [_numeric_check("method_loss", compare_arrays([ref["loss"]], [cand["loss"]]), reference.eval_config)]
    if reference.eval_config.training_method == "dpo":
        checks.append(_numeric_check("log_probs", compare_arrays(ref["policy_logps"], cand["policy_logps"]), reference.eval_config))
        if ref.get("reference_logps") is not None and cand.get("reference_logps") is not None:
            checks.append(_numeric_check("ref_log_probs", compare_arrays(ref["reference_logps"], cand["reference_logps"]), reference.eval_config))
    if reference.eval_config.training_method == "ppo":
        checks.append(_numeric_check("token_logprobs", compare_arrays(ref["token_logprobs"], cand["token_logprobs"]), reference.eval_config))
        checks.append(_numeric_check("advantages", compare_arrays(ref["advantages"], cand["advantages"]), reference.eval_config))
        checks.append(_numeric_check("returns", compare_arrays(ref["returns"], cand["returns"]), reference.eval_config))
    return checks


def _compare_gradients(reference: RepoRuntime, candidate: RepoRuntime, report: EvalReport) -> list[CheckResult]:
    ref = reference.gradient_artifact()
    cand = candidate.gradient_artifact()
    report.artifacts["reference_gradients"] = jsonable(ref)
    report.artifacts["candidate_gradients"] = jsonable(cand)
    return [
        _numeric_check("gradient_loss", compare_arrays([ref["loss"]], [cand["loss"]]), reference.eval_config),
        _numeric_check("gradient_norm", compare_arrays([ref["grad_norm"]], [cand["grad_norm"]]), reference.eval_config),
    ]


def _compare_lr(reference: RepoRuntime, candidate: RepoRuntime, report: EvalReport) -> CheckResult:
    ref = reference.lr_schedule_artifact()
    cand = candidate.lr_schedule_artifact()
    report.artifacts["reference_lr_schedule"] = jsonable(ref)
    report.artifacts["candidate_lr_schedule"] = jsonable(cand)
    return _numeric_check("lr_schedule", compare_arrays(ref["learning_rates"], cand["learning_rates"]), reference.eval_config)


def _compare_invariants(reference: RepoRuntime, candidate: RepoRuntime, report: EvalReport) -> CheckResult:
    ref = reference.gradient_accumulation_invariant()
    cand = candidate.gradient_accumulation_invariant()
    report.artifacts["reference_invariant"] = jsonable(ref)
    report.artifacts["candidate_invariant"] = jsonable(cand)
    if not ref.get("supported") or not cand.get("supported"):
        return _pass("gradient_accumulation", "numeric", "gradient accumulation invariant not supported on this batch")
    metrics = compare_arrays(
        [ref["full_batch_loss"], ref["micro_batch_loss"]],
        [cand["full_batch_loss"], cand["micro_batch_loss"]],
    )
    return _numeric_check("gradient_accumulation", metrics, reference.eval_config)


def _compare_behavior(reference: RepoRuntime, candidate: RepoRuntime, report: EvalReport) -> list[CheckResult]:
    checks: list[CheckResult] = []
    ref = reference.loss_trajectory_artifact()
    cand = candidate.loss_trajectory_artifact()
    report.artifacts["reference_loss_trajectory"] = jsonable(ref)
    report.artifacts["candidate_loss_trajectory"] = jsonable(cand)
    if ref.get("supported") is False or cand.get("supported") is False:
        reason = ref.get("reason") or cand.get("reason") or "loss-trajectory replay failed"
        checks.append(_fail("loss_curve", "behavioral", reason, failure_kind="behavior_unavailable", source="jax"))
    else:
        ref_losses = [item["loss"] for item in ref["trajectory"]]
        cand_losses = [item["loss"] for item in cand["trajectory"]]
        checks.append(_numeric_check("loss_curve", compare_arrays(ref_losses, cand_losses), reference.eval_config))
    if reference.eval_config.training_method == "sft":
        ref_gen = reference.generation_artifact()
        cand_gen = candidate.generation_artifact()
        report.artifacts["reference_generation"] = jsonable(ref_gen)
        report.artifacts["candidate_generation"] = jsonable(cand_gen)
        if ref_gen.get("supported") and cand_gen.get("supported"):
            checks.append(_numeric_check("generation", compare_arrays(ref_gen["sequences"], cand_gen["sequences"]), reference.eval_config))
        elif not ref_gen.get("supported"):
            reason = ref_gen.get("reason") or "generation not supported by reference"
            checks.append(_pass("generation", "behavioral", reason))
        else:
            reason = ref_gen.get("reason") or cand_gen.get("reason") or "generation artifact failed"
            checks.append(_fail("generation", "behavioral", reason, failure_kind="behavior_unavailable", source="jax"))
    return checks


def run_evaluation(config: EvalConfig) -> EvalReport:
    import time, sys
    def _log(msg): print(f"[EVAL {time.time():.0f}] {msg}", flush=True, file=sys.stderr)
    ensure_dir(config.output_dir)
    report = EvalReport(config=asdict(config))
    impl_map = config.implementation_map()
    _log("Creating first reference runtime...")
    try:
        reference = RepoRuntime.create(config, impl_map[config.reference_impl])
    except Exception as exc:
        report.reference.append(
            _fail(
                "reference_runtime",
                "reference",
                with_contract_hint(str(exc), exc),
                failure_kind=failure_kind_from_exception(exc),
                source="torch",
            )
        )
        report.spec.append(_blocked("spec", "stage_gate", "reference validation failed", source="torch"))
        report.numeric.append(_blocked("numeric", "stage_gate", "reference validation failed", source="torch"))
        report.behavioral.append(_blocked("behavioral", "stage_gate", "reference validation failed", source="torch"))
        write_report(config.output_dir, report)
        return report
    _log("Running reference checks...")
    report.reference.extend(_check_reference(reference))
    _log(f"Reference checks done: {[c.name+':'+str(c.passed) for c in report.reference]}")
    if not all(item.passed for item in report.reference):
        report.spec.append(_blocked("spec", "stage_gate", "reference validation failed", source="torch"))
        report.numeric.append(_blocked("numeric", "stage_gate", "reference validation failed", source="torch"))
        report.behavioral.append(_blocked("behavioral", "stage_gate", "reference validation failed", source="torch"))
        write_report(config.output_dir, report)
        return report
    _log("Releasing first reference...")
    _release_runtime(reference)
    _log("Creating second reference runtime...")
    try:
        reference = RepoRuntime.create(config, impl_map[config.reference_impl])
    except Exception as exc:
        report.spec.append(
            _fail(
                "reference_reinit",
                "spec",
                with_contract_hint(str(exc), exc),
                failure_kind=failure_kind_from_exception(exc),
                source="torch",
            )
        )
        report.numeric.append(_blocked("numeric", "stage_gate", "reference reinitialization failed", source="torch"))
        report.behavioral.append(_blocked("behavioral", "stage_gate", "reference reinitialization failed", source="torch"))
        write_report(config.output_dir, report)
        return report
    _log("Creating candidate runtime...")
    try:
        candidate = RepoRuntime.create(config, impl_map[config.candidate_impl])
    except Exception as exc:
        failure = _fail(
            "candidate_init",
            "spec",
            with_contract_hint(str(exc), exc),
            failure_kind=failure_kind_from_exception(exc),
            source="jax",
        )
        report.spec.append(failure)
        report.numeric.append(_blocked("numeric", "stage_gate", "candidate initialization failed", source="jax"))
        report.behavioral.append(_blocked("behavioral", "stage_gate", "candidate initialization failed", source="jax"))
        write_report(config.output_dir, report)
        return report

    try:
        runtime_contract = reference.derive_runtime_contract()
        reference.runtime_contract = copy.deepcopy(runtime_contract)
        candidate.runtime_contract = copy.deepcopy(runtime_contract)
    except Exception as exc:
        report.spec.append(_fail("runtime_contract", "spec", str(exc), failure_kind=failure_kind_from_exception(exc), source="torch"))
        report.numeric.append(_blocked("numeric", "stage_gate", "runtime contract derivation failed", source="torch"))
        report.behavioral.append(_blocked("behavioral", "stage_gate", "runtime contract derivation failed", source="torch"))
        write_report(config.output_dir, report)
        return report

    _log("Candidate created. Running spec checks...")
    try:
        report.spec.append(_compare_parameter_trees(reference, candidate))
        report.spec.append(_compare_batch_schema(reference, candidate))
    except Exception as exc:
        report.spec.append(_fail("spec_runtime", "spec", str(exc), failure_kind=failure_kind_from_exception(exc), source="jax"))
    _log(f"Spec results: {[c.name+':'+str(c.passed) for c in report.spec]}")
    if not all(item.passed for item in report.spec):
        report.numeric.append(_blocked("numeric", "stage_gate", "spec stage failed", source="jax"))
        report.behavioral.append(_blocked("behavioral", "stage_gate", "spec stage failed", source="jax"))
        write_report(config.output_dir, report)
        return report

    step_names = ["forward", "loss_artifacts", "gradients", "lr", "invariants"]
    numeric_steps = [
        lambda: report.numeric.extend(_compare_forward(reference, candidate, report)),
        lambda: report.numeric.extend(_compare_loss_artifacts(reference, candidate, report)),
        lambda: report.numeric.extend(_compare_gradients(reference, candidate, report)),
        lambda: report.numeric.append(_compare_lr(reference, candidate, report)),
        lambda: report.numeric.append(_compare_invariants(reference, candidate, report)),
    ]
    for i, step in enumerate(numeric_steps):
        _log(f"Numeric step {i}: {step_names[i]}...")
        try:
            step()
        except Exception as exc:
            report.numeric.append(_fail("numeric_runtime", "numeric", str(exc), failure_kind=failure_kind_from_exception(exc), source="jax"))
            break
        _log(f"  -> results: {[c.name+':'+str(c.passed) for c in report.numeric]}")
    if not all(item.passed for item in report.numeric):
        report.behavioral.append(_blocked("behavioral", "stage_gate", "numeric stage failed", source="jax"))
        write_report(config.output_dir, report)
        return report
    try:
        report.behavioral.extend(_compare_behavior(reference, candidate, report))
    except Exception as exc:
        report.behavioral.append(_fail("behavior_runtime", "behavioral", str(exc), failure_kind=failure_kind_from_exception(exc), source="jax"))
    write_report(config.output_dir, report)
    return report


def write_report(output_dir: str, report: EvalReport) -> None:
    ensure_dir(output_dir)
    report_payload = report.to_dict()
    report_payload["artifacts"] = _report_safe_artifacts(report.artifacts)
    with open(f"{output_dir}/new_eval_report.json", "w") as handle:
        json.dump(jsonable(report_payload), handle, indent=2, sort_keys=True)
    with open(f"{output_dir}/new_eval_summary.json", "w") as handle:
        json.dump(jsonable(report.summary()), handle, indent=2, sort_keys=True)
