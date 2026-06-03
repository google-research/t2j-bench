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

"""Shared fixtures and path setup for Gemma3 SFT torch_orig vs jax_repo tests."""
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

# ── Repo roots ──────────────────────────────────────────────────────────────
GEMMA3_ROOT = Path(__file__).resolve().parent.parent
TORCH_ROOT = GEMMA3_ROOT / "torch_orig"
JAX_ROOT = GEMMA3_ROOT / "jax_repo"


def _load_package(root: Path, package_name: str) -> ModuleType:
    """Import a repo directory as a top-level package so relative imports work."""
    if package_name in sys.modules:
        return sys.modules[package_name]
    init_path = root / "__init__.py"
    if not init_path.exists():
        # Create a synthetic __init__.py-less package
        spec = importlib.util.spec_from_file_location(
            package_name,
            root / "__init__.py" if (root / "__init__.py").exists() else None,
            submodule_search_locations=[str(root)],
        )
        if spec is None:
            # Build a namespace package
            mod = ModuleType(package_name)
            mod.__path__ = [str(root)]
            mod.__package__ = package_name
            sys.modules[package_name] = mod
            return mod
        mod = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = mod
        if spec.loader:
            spec.loader.exec_module(mod)
        return mod
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(root)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_submodule(root: Path, package_name: str, submodule: str) -> ModuleType:
    """Import ``<package_name>.<submodule>`` from *root* on disk."""
    _load_package(root, package_name)
    full_name = f"{package_name}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    # Walk dotted path to file
    parts = submodule.split(".")
    file_path = root
    for part in parts[:-1]:
        file_path = file_path / part
        # Ensure intermediate packages exist
        pkg_name = f"{package_name}.{'.'.join(parts[:parts.index(part)+1])}"
        if pkg_name not in sys.modules:
            _load_package(file_path, pkg_name)

    file_path = file_path / (parts[-1] + ".py")
    if not file_path.exists():
        # Try as a package
        file_path = root
        for part in parts:
            file_path = file_path / part
        file_path = file_path / "__init__.py"

    spec = importlib.util.spec_from_file_location(full_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot find {full_name} at {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Convenience import helpers exposed as fixtures ──────────────────────────

@pytest.fixture(scope="session")
def torch_config():
    """Return the torch_orig.config module."""
    sys.path.insert(0, str(TORCH_ROOT))
    try:
        return _import_submodule(TORCH_ROOT, "_torch_orig", "config")
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="session")
def jax_config():
    """Return the jax_repo.config module."""
    sys.path.insert(0, str(JAX_ROOT))
    try:
        return _import_submodule(JAX_ROOT, "_jax_repo", "config")
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="session")
def torch_configuration():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "transformers.configuration")


@pytest.fixture(scope="session")
def jax_configuration():
    return _import_submodule(JAX_ROOT, "_jax_repo", "transformers.configuration")


@pytest.fixture(scope="session")
def torch_utils():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "transformers.utils")


@pytest.fixture(scope="session")
def jax_utils():
    return _import_submodule(JAX_ROOT, "_jax_repo", "transformers.utils")


@pytest.fixture(scope="session")
def torch_modeling():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "transformers.modeling")


@pytest.fixture(scope="session")
def jax_modeling():
    return _import_submodule(JAX_ROOT, "_jax_repo", "transformers.modeling")


@pytest.fixture(scope="session")
def torch_formatter():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "data.formatter")


@pytest.fixture(scope="session")
def jax_formatter():
    return _import_submodule(JAX_ROOT, "_jax_repo", "data.formatter")


@pytest.fixture(scope="session")
def torch_template():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "data.template")


@pytest.fixture(scope="session")
def jax_template():
    return _import_submodule(JAX_ROOT, "_jax_repo", "data.template")


@pytest.fixture(scope="session")
def torch_constants():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "data.constants")


@pytest.fixture(scope="session")
def jax_constants():
    return _import_submodule(JAX_ROOT, "_jax_repo", "data.constants")


@pytest.fixture(scope="session")
def torch_processor():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "data.processor")


@pytest.fixture(scope="session")
def jax_processor():
    return _import_submodule(JAX_ROOT, "_jax_repo", "data.processor")


@pytest.fixture(scope="session")
def torch_collator():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "data.collator")


@pytest.fixture(scope="session")
def jax_collator():
    return _import_submodule(JAX_ROOT, "_jax_repo", "data.collator")


@pytest.fixture(scope="session")
def torch_mm_plugin():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "data.mm_plugin")


@pytest.fixture(scope="session")
def jax_mm_plugin():
    return _import_submodule(JAX_ROOT, "_jax_repo", "data.mm_plugin")


@pytest.fixture(scope="session")
def torch_tool_utils():
    return _import_submodule(TORCH_ROOT, "_torch_orig", "data.tool_utils")


@pytest.fixture(scope="session")
def jax_tool_utils():
    return _import_submodule(JAX_ROOT, "_jax_repo", "data.tool_utils")


# ── Tiny model configs for fast unit tests ──────────────────────────────────

@pytest.fixture
def tiny_text_config_dict():
    """A minimal Gemma3TextConfig dictionary for fast tests."""
    return dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=8,
        sliding_window=16,
        sliding_window_pattern=6,
    )


@pytest.fixture
def tiny_vision_config_dict():
    """A minimal SiglipVisionConfig dictionary for fast tests."""
    return dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=28,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        vision_use_head=False,
    )


@pytest.fixture
def tiny_gemma3_config_dict(tiny_text_config_dict, tiny_vision_config_dict):
    """A minimal Gemma3Config dictionary for fast tests."""
    return dict(
        text_config=tiny_text_config_dict,
        vision_config=tiny_vision_config_dict,
        mm_tokens_per_image=4,
        boi_token_index=100,
        eoi_token_index=101,
        image_token_index=127,
    )
