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

"""Static checks for framework independence in native-JAX candidates.

scan_repo_for_torch finds banned framework imports, recognizable dynamic-import
evasion, and exec/eval strings containing banned roots. jaxpr_uses_callbacks
checks captured JAX expressions for host-callback primitives. Source scanning
requires no PyTorch or JAX import and reports file/line diagnostics.
These checks complement runtime blocking; they are not a general security sandbox.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Top-level packages whose import implies a torch runtime dependency.
DEFAULT_BANNED_ROOTS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "functorch",
    "torch_xla",
)

# Directories excluded from the runtime import graph (offline build steps only).
DEFAULT_ALLOW_DIRS: tuple[str, ...] = ("convert",)

# Callback primitives that allow a "pure" JAX forward to call host/torch code.
CALLBACK_PRIMITIVES: tuple[str, ...] = ("pure_callback", "io_callback", "callback")


@dataclass(slots=True)
class PurityFinding:
    path: str
    lineno: int
    kind: str  # "import" | "from_import" | "dynamic_import" | "exec_eval" | "callback"
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "lineno": self.lineno, "kind": self.kind, "detail": self.detail}


@dataclass(slots=True)
class PurityReport:
    findings: list[PurityFinding] = field(default_factory=list)
    scanned_files: int = 0
    skipped_dirs: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "clean": self.clean,
            "scanned_files": self.scanned_files,
            "skipped_dirs": sorted(set(self.skipped_dirs)),
            "findings": [item.to_dict() for item in self.findings],
        }


def _root_of(module_name: str | None) -> str | None:
    if not module_name:
        return None
    return module_name.split(".", 1)[0]


def _is_banned_module(module_name: str | None, banned_roots: Sequence[str]) -> bool:
    if not module_name:
        return False
    for root in banned_roots:
        if module_name == root or module_name.startswith(root + "."):
            return True
    return False


def _string_mentions_banned(text: str, banned_roots: Sequence[str]) -> bool:
    # Catch both literal "torch" and trivial concatenations once joined.
    return any(root in text for root in banned_roots)


class _TorchImportVisitor(ast.NodeVisitor):
    def __init__(self, path: str, banned_roots: Sequence[str]) -> None:
        self.path = path
        self.banned_roots = banned_roots
        self.findings: list[PurityFinding] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _is_banned_module(alias.name, self.banned_roots):
                self.findings.append(
                    PurityFinding(self.path, node.lineno, "import", f"import {alias.name}")
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Relative imports (node.level > 0) cannot reach top-level torch.
        if node.level == 0 and _is_banned_module(node.module, self.banned_roots):
            self.findings.append(
                PurityFinding(self.path, node.lineno, "from_import", f"from {node.module} import ...")
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._callable_name(node.func)
        if func_name in {"import_module", "__import__"}:
            for arg in node.args:
                literal = self._const_str(arg)
                if literal is not None and _is_banned_module(_root_of(literal), self.banned_roots):
                    self.findings.append(
                        PurityFinding(
                            self.path, node.lineno, "dynamic_import", f"{func_name}({literal!r})"
                        )
                    )
                elif literal is None and self._has_banned_string_parts(arg):
                    self.findings.append(
                        PurityFinding(
                            self.path,
                            node.lineno,
                            "dynamic_import",
                            f"{func_name}(<computed torch name>)",
                        )
                    )
        elif func_name in {"exec", "eval"}:
            for arg in node.args:
                literal = self._const_str(arg)
                if literal is not None and _string_mentions_banned(literal, self.banned_roots):
                    self.findings.append(
                        PurityFinding(
                            self.path, node.lineno, "exec_eval", f"{func_name}(<torch in source>)"
                        )
                    )
        self.generic_visit(node)

    @staticmethod
    def _callable_name(func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    @staticmethod
    def _const_str(node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _has_banned_string_parts(self, node: ast.expr) -> bool:
        # Detect "tor" + "ch" style obfuscation: join all string literals in the
        # expression subtree and look for a banned root.
        parts: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                parts.append(child.value)
        return _string_mentions_banned("".join(parts), self.banned_roots)


def _is_under_allowed_dir(rel_parts: Sequence[str], allow_dirs: Sequence[str]) -> bool:
    return any(part in allow_dirs for part in rel_parts)


def scan_repo_for_torch(
    repo_dir: str,
    *,
    banned_roots: Iterable[str] | None = None,
    allow_dirs: Iterable[str] | None = None,
) -> PurityReport:
    """Scan every ``*.py`` under ``repo_dir`` for torch imports/evasion.

    Files under any directory named in ``allow_dirs`` (default: ``convert/``)
    are skipped, since the design permits torch in offline build steps that are
    not part of the runtime import graph (re-enforced at runtime by the sandbox).
    """

    roots = tuple(banned_roots) if banned_roots is not None else DEFAULT_BANNED_ROOTS
    allowed = tuple(allow_dirs) if allow_dirs is not None else DEFAULT_ALLOW_DIRS
    base = Path(repo_dir).resolve()
    report = PurityReport()
    for py_path in sorted(base.rglob("*.py")):
        rel_parts = py_path.relative_to(base).parts
        if _is_under_allowed_dir(rel_parts[:-1], allowed):
            report.skipped_dirs.append(str(Path(*rel_parts[:-1])) if len(rel_parts) > 1 else ".")
            continue
        report.scanned_files += 1
        rel = str(py_path.relative_to(base))
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            report.findings.append(PurityFinding(rel, 0, "parse_error", str(exc)))
            continue
        visitor = _TorchImportVisitor(rel, roots)
        visitor.visit(tree)
        report.findings.extend(visitor.findings)
    return report


def count_jaxpr_primitives(jaxpr: object) -> int:
    """Count equations in ``jaxpr`` (recursing into sub-jaxprs).

    A genuine transformer forward compiles to many primitives; a host-callback
    shim that merely calls torch produces a near-trivial jaxpr (§6.5).
    """

    return _count_eqns(jaxpr, seen=set())


def _count_eqns(obj: object, seen: set[int]) -> int:
    inner = getattr(obj, "jaxpr", None)
    if inner is not None and inner is not obj:
        return _count_eqns(inner, seen)
    eqns = getattr(obj, "eqns", None)
    if eqns is None or id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = 0
    for eqn in eqns:
        total += 1
        params = getattr(eqn, "params", {}) or {}
        for value in params.values():
            for sub in _iter_jaxprs(value):
                total += _count_eqns(sub, seen)
    return total


def jaxpr_uses_callbacks(jaxpr: object) -> list[str]:
    """Return host-callback primitive names found anywhere in ``jaxpr``.

    Accepts a ``jax.core.ClosedJaxpr`` / ``Jaxpr`` (or anything exposing
    ``.jaxpr``/``.eqns``). Recurses into sub-jaxprs carried in equation params.
    A non-empty result means the "pure" forward may be calling out to host
    (torch) code (§6.5).
    """

    found: list[str] = []
    _collect_callback_prims(jaxpr, found, seen=set())
    return sorted(set(found))


def _collect_callback_prims(obj: object, found: list[str], seen: set[int]) -> None:
    inner = getattr(obj, "jaxpr", None)
    if inner is not None and inner is not obj:
        _collect_callback_prims(inner, found, seen)
    eqns = getattr(obj, "eqns", None)
    if eqns is None:
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))
    for eqn in eqns:
        prim_name = str(getattr(getattr(eqn, "primitive", None), "name", ""))
        for needle in CALLBACK_PRIMITIVES:
            if needle in prim_name:
                found.append(prim_name)
        params = getattr(eqn, "params", {}) or {}
        for value in params.values():
            for sub in _iter_jaxprs(value):
                _collect_callback_prims(sub, found, seen)


def _iter_jaxprs(value: object) -> Iterable[object]:
    if hasattr(value, "eqns") or hasattr(value, "jaxpr"):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_jaxprs(item)
