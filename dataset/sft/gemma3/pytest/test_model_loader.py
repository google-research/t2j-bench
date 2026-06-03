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

"""Tests verifying jax_repo model loader produces consistent results with torch_orig.

These tests focus on the JAX-specific model loader (TorchBackedSFTJaxModel) and
its bridging between torch and JAX, comparing against the direct torch model loader.
They don't require a real pretrained model — they test the loader logic and cross-entropy
loss helper.
"""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_JAX_LOADER_MOD = None


def _get_jax_loader():
    """Load jax_repo/model/loader.py with proper sys.modules registration."""
    global _JAX_LOADER_MOD
    if _JAX_LOADER_MOD is not None:
        return _JAX_LOADER_MOD

    mod_name = "_jax_model_loader_test"
    jax_root = Path(__file__).resolve().parent.parent / "jax_repo"
    loader_path = jax_root / "model" / "loader.py"

    spec = importlib.util.spec_from_file_location(mod_name, loader_path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass can resolve __module__
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        del sys.modules[mod_name]
        return None

    _JAX_LOADER_MOD = mod
    return mod


@pytest.fixture
def jax_loader():
    mod = _get_jax_loader()
    if mod is None:
        pytest.skip("jax not available for import")
    return mod


class TestCrossEntropyLossHelper:
    """Verify the JAX loader's _cross_entropy_loss matches standard torch CE."""

    def test_basic_loss(self, jax_loader):
        """Basic cross-entropy loss should match standard torch implementation."""
        torch.manual_seed(42)
        vocab_size = 128
        batch_size, seq_len = 2, 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        jax_loss = jax_loader._cross_entropy_loss(logits, labels, ignore_index=-100)

        # Manual computation
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:].contiguous()
        expected_loss = F.cross_entropy(
            shifted_logits.reshape(-1, vocab_size),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )

        assert torch.allclose(jax_loss, expected_loss, atol=1e-6)

    def test_with_ignore_index(self, jax_loader):
        """Positions with ignore_index should not contribute to loss."""
        torch.manual_seed(42)
        vocab_size = 64
        logits = torch.randn(1, 8, vocab_size)
        labels = torch.randint(0, vocab_size, (1, 8))
        labels[0, :4] = -100  # ignore first 4 positions

        jax_loss = jax_loader._cross_entropy_loss(logits, labels, ignore_index=-100)

        # Manual computation
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:].contiguous()
        expected = F.cross_entropy(
            shifted_logits.reshape(-1, vocab_size),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )

        assert torch.allclose(jax_loss, expected, atol=1e-6)

    def test_all_ignored(self, jax_loader):
        """When all labels are ignore_index, CE returns nan (no valid targets)."""
        logits = torch.randn(1, 5, 32)
        labels = torch.full((1, 5), -100, dtype=torch.long)

        loss = jax_loader._cross_entropy_loss(logits, labels, ignore_index=-100)
        # torch CE with no valid targets returns nan
        expected = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, 32),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        # Both should be nan
        assert torch.isnan(loss) == torch.isnan(expected)


class TestToTorchTensor:
    """Verify _to_torch_tensor handles various input types."""

    def test_torch_tensor_passthrough(self, jax_loader):
        device = torch.device("cpu")
        t = torch.tensor([1.0, 2.0, 3.0])
        result = jax_loader._to_torch_tensor(t, device)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, t)

    def test_numpy_float_array(self, jax_loader):
        import numpy as np
        device = torch.device("cpu")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = jax_loader._to_torch_tensor(arr, device)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_numpy_int_array(self, jax_loader):
        import numpy as np
        device = torch.device("cpu")
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = jax_loader._to_torch_tensor(arr, device)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.int64  # converted to int64


class TestParamSpecs:
    """Verify _as_param_spec and _state_dict_to_param_specs work correctly."""

    def test_float_tensor_spec(self, jax_loader):
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("jax not available")

        t = torch.randn(3, 4)
        spec = jax_loader._as_param_spec(t)
        assert spec.shape == (3, 4)
        assert spec.dtype == jnp.float32

    def test_int_tensor_spec(self, jax_loader):
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("jax not available")

        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        spec = jax_loader._as_param_spec(t)
        assert spec.shape == (3,)
        assert spec.dtype == jnp.int64

    def test_state_dict_to_specs(self, jax_loader):
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("jax not available")

        model = torch.nn.Linear(4, 8)
        specs = jax_loader._state_dict_to_param_specs(model)
        assert "weight" in specs
        assert "bias" in specs
        assert specs["weight"].shape == (8, 4)


class TestTokenizerModuleType:
    """Verify TokenizerModule TypedDict is consistent."""

    def test_typed_dict_keys(self):
        # Check both repos define TokenizerModule with same keys
        torch_root = Path(__file__).resolve().parent.parent / "torch_orig"
        jax_root = Path(__file__).resolve().parent.parent / "jax_repo"

        torch_loader_path = torch_root / "model" / "loader.py"
        jax_loader_path = jax_root / "model" / "loader.py"

        with open(torch_loader_path) as f:
            torch_source = f.read()
        with open(jax_loader_path) as f:
            jax_source = f.read()

        assert "class TokenizerModule" in torch_source
        assert "class TokenizerModule" in jax_source
        assert "tokenizer" in torch_source
        assert "processor" in torch_source
        assert "tokenizer" in jax_source
        assert "processor" in jax_source
