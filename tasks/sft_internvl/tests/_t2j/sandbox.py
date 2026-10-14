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

"""Torch-blocking import sandbox for native-JAX candidates.

The native-JAX contract (see ``NATIVE_JAX_CONVERSION_DESIGN.md`` R1) requires
that a candidate ``jax_repo`` never imports PyTorch at runtime. This module
installs a :class:`importlib.abc.MetaPathFinder` at the very front of
``sys.meta_path`` that raises :class:`ImportError` for any module whose name is
in (or is a submodule of) a configurable denylist. Installing the blocker
*before* the candidate is imported makes torch usage structurally impossible
rather than merely discouraged.

It is intended to run inside a freshly spawned subprocess (see
``candidate_worker.py``) so that the reference process — which legitimately uses
torch — is unaffected.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Iterable, Sequence

# Default denylist of top-level packages that imply a torch runtime dependency.
DEFAULT_BANNED_MODULES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "functorch",
    "torch_xla",
)


class TorchBlockedError(ImportError):
    """Raised when banned (torch-family) modules are imported in the sandbox."""


def _is_banned(module_name: str, banned: Sequence[str]) -> bool:
    for name in banned:
        if module_name == name or module_name.startswith(name + "."):
            return True
    return False


class BannedModuleFinder(MetaPathFinder):
    """A meta-path finder that refuses to import banned module names."""

    def __init__(self, banned: Sequence[str]) -> None:
        # Normalize/freeze so later mutation of the caller's list is harmless.
        self.banned: tuple[str, ...] = tuple(dict.fromkeys(banned))

    def find_spec(self, fullname: str, path=None, target=None) -> ModuleSpec | None:  # noqa: ANN001
        if _is_banned(fullname, self.banned):
            raise TorchBlockedError(
                f"'{fullname}' is banned in the native-jax candidate sandbox "
                f"(banned roots: {', '.join(self.banned)})"
            )
        # Returning None lets the normal finders handle every other module.
        return None


def install_torch_blocker(banned: Iterable[str] | None = None) -> BannedModuleFinder:
    """Install the torch blocker at the front of ``sys.meta_path``.

    Idempotent: a previously installed blocker is replaced. Any banned module
    that is *already* in ``sys.modules`` (e.g. leaked in via the parent process)
    is evicted so subsequent attribute access cannot succeed.
    """

    banned_list = list(banned) if banned is not None else list(DEFAULT_BANNED_MODULES)
    # Remove any pre-existing blocker instances.
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, BannedModuleFinder)
    ]
    finder = BannedModuleFinder(banned_list)
    sys.meta_path.insert(0, finder)
    # Evict already-imported banned modules so a stale reference cannot be reused.
    for module_name in list(sys.modules):
        if _is_banned(module_name, finder.banned):
            sys.modules.pop(module_name, None)
    return finder


def banned_modules_in_sys(banned: Iterable[str] | None = None) -> list[str]:
    """Return any currently-loaded modules that violate the denylist.

    Used for the §6.2 module-table assertion after the candidate produces an
    artifact, to catch vendored/renamed torch that slipped past the finder.
    """

    banned_tuple = tuple(banned) if banned is not None else DEFAULT_BANNED_MODULES
    return sorted(name for name in sys.modules if _is_banned(name, banned_tuple))
