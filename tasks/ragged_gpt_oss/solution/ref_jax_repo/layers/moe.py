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

"""The routed mixture-of-experts block, mirroring torch_orig/layers/moe.py.

Same two classes, same attribute names, same order of operations. The names are
vLLM's (`gpt_oss.py`: ``GptOssRoutedExperts`` at :192, ``MLPBlock`` at :318),
because ``test_reference_units`` grades a port on mirroring the TORCH side's
attribute names and the torch side chose them.

A RESTRUCTURING OF UPSTREAM, NOT A DE-VENDORING OF IT. tpu-inference's
`gpt_oss.py` is one monolithic `nnx.Module` (`class GptOss`, 555 lines) with an
inline `__call__` and no MLP / Attention / DecoderLayer split at all -- unlike
its `qwen2.py`, which has exactly this decomposition. So there is no upstream
JAX `MLPBlock` to follow; there is a torch `MLPBlock` this file mirrors, and the
arithmetic came from `transformers`' `GptOssExperts` via the torch side's own
differential test.

THE WEIGHTS STAY PACKED (the decision the family rests on) and the FLOP-density
trade are both argued at length in the torch twin. Two JAX-specific notes:

  * ``nnx.Param`` over a ``uint8`` array is fine and is what keeps the packed
    tensors inside the parameter pytree, so ``apply_params``' coverage rule
    reaches them and ``test_param_correspondence`` sees the same key set as
    torch's ``named_parameters()``.
  * the dense contraction is also what makes the layer JITTABLE. True
    FLOP-sparsity needs a grouped GEMM over a dynamic number of rows per
    expert, which has no static-shape spelling in `jnp`; the alternative would
    be capacity-based routing, which DROPS tokens and is not what the
    checkpoint was trained for. Being dense here is a cost paid for being
    exact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig
from layers.activation import ClampedSwiGLU
from layers.linear import ReplicatedLinear
from quant.mxfp4 import BYTES_PER_BLOCK, VALUES_PER_BLOCK, dequantize_mxfp4


def _blocks_shape(rows: int, contracted: int) -> tuple[int, int, int]:
    """``(rows, groups, bytes)`` for one expert's packed projection."""
    return (rows, contracted // VALUES_PER_BLOCK, BYTES_PER_BLOCK)


class GptOssRoutedExperts(nnx.Module):
    """The 32 experts of one layer, resident as MXFP4.

    Parameter names are the checkpoint's own -- ``gate_up_proj_blocks``,
    ``gate_up_proj_scales``, ``gate_up_proj_bias`` and the ``down_proj`` triple
    -- so ``model/loader.py`` converts tensor for tensor with no rename table.

    ``dtype`` governs the BIASES and the dequantised weights; the packed tensors
    are uint8 whatever it says, which is the point.
    """

    def __init__(self, config: ModelConfig, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        experts = config.num_local_experts
        hidden = config.hidden_size
        inter = config.intermediate_size
        for name, size in (("hidden_size", hidden), ("intermediate_size", inter)):
            if size % VALUES_PER_BLOCK:
                raise ValueError(
                    f"MXFP4 packs {VALUES_PER_BLOCK} values to a block, so the "
                    f"contracted axis must be a multiple of {VALUES_PER_BLOCK}; "
                    f"{name} is {size}")
        self.num_experts = experts
        self.hidden_size = hidden
        self.intermediate_size = inter

        # Zeros, mirroring torch_orig's `torch.empty`: a layer that has not been
        # through the loader holds nothing meaningful on either side, and JAX has
        # no uninitialised array. `apply_params` refuses to return a model with
        # any parameter left unwritten.
        self.gate_up_proj_blocks = nnx.Param(
            jnp.zeros((experts, *_blocks_shape(2 * inter, hidden)), jnp.uint8))
        self.gate_up_proj_scales = nnx.Param(
            jnp.zeros((experts, 2 * inter, hidden // VALUES_PER_BLOCK), jnp.uint8))
        self.gate_up_proj_bias = nnx.Param(jnp.zeros((experts, 2 * inter), dtype))
        self.down_proj_blocks = nnx.Param(
            jnp.zeros((experts, *_blocks_shape(hidden, inter)), jnp.uint8))
        self.down_proj_scales = nnx.Param(
            jnp.zeros((experts, hidden, inter // VALUES_PER_BLOCK), jnp.uint8))
        self.down_proj_bias = nnx.Param(jnp.zeros((experts, hidden), dtype))

        self.act_fn = ClampedSwiGLU(limit=config.swiglu_limit)

    # --- the dequant, never cached ----------------------------------------- #
    def dequantized_gate_up(self, dtype) -> jax.Array:
        """``[E, hidden, 2 * inter]``. Fresh every call, ON PURPOSE.

        Caching it would put 2.1 GB (fp32) of expert weights per layer back on
        the device permanently and defeat the packing. Inside jit, XLA's
        liveness analysis frees it as soon as the layer's two contractions are
        done, so the transient is one layer's worth and not twenty-four.
        """
        return dequantize_mxfp4(self.gate_up_proj_blocks[...],
                                self.gate_up_proj_scales[...], dtype=dtype)

    def dequantized_down(self, dtype) -> jax.Array:
        """``[E, inter, hidden]``. Both axes are 2880 on this checkpoint, which
        is exactly why `quant/mxfp4.py`'s ``swap_last2`` has its own test."""
        return dequantize_mxfp4(self.down_proj_blocks[...],
                                self.down_proj_scales[...], dtype=dtype)

    def _zero_unselected(self, expert_ids) -> None:
        """Make the named experts contribute nothing. Test affordance only.

        Code 0x0 is +0.0 at every scale, so zeroing the packed bytes zeroes the
        weights exactly; the biases go with them. In place -- `param[...] = ...`
        -- never a rebind, for the same reason `apply_params` is.

        UNDERSCORED so it stays OUT of the graded contract; the torch twin states
        the mechanism (``interface.json['methods']`` is the intersection of the two
        sides' public methods, so a public name here would make every candidate
        implement a test-only helper).
        """
        index = jnp.asarray(sorted(expert_ids), dtype=jnp.int32)
        for param in (self.gate_up_proj_blocks, self.gate_up_proj_scales,
                      self.gate_up_proj_bias, self.down_proj_blocks,
                      self.down_proj_scales, self.down_proj_bias):
            param[...] = param[...].at[index].set(0)

    def __call__(self, hidden: jax.Array, routing_weights: jax.Array) -> jax.Array:
        """``hidden [T, H]`` and ``routing_weights [T, E]`` -> ``[T, H]``.

        ``routing_weights`` is DENSE over the experts and zero off the top-k, so
        this signature carries the whole routing decision as data. That is what
        lets ``test_reference_units`` replay this module on a frozen input: an
        argument the capture recorded, not a decision re-made inside.
        """
        dtype = hidden.dtype
        gate_up = jnp.einsum(
            "th,ehf->tef", hidden, self.dequantized_gate_up(dtype)
        ) + self.gate_up_proj_bias[...]
        activated = self.act_fn(gate_up)
        out = jnp.einsum(
            "tei,eih->teh", activated, self.dequantized_down(dtype)
        ) + self.down_proj_bias[...]
        return jnp.einsum("te,teh->th", routing_weights.astype(out.dtype), out)


class MLPBlock(nnx.Module):
    """Router + experts, mirroring vLLM's ``MLPBlock``."""

    def __init__(self, config: ModelConfig, *, dtype=jnp.float32,
                 rngs: nnx.Rngs | None = None) -> None:
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        # ReplicatedLinear, WITH a bias. `modules_to_not_convert` keeps the whole
        # router out of the MXFP4 conversion -- 32 logits decide which 19.1B
        # parameters run, so the router is the one place in this layer where
        # 4-bit weights would be a bad trade.
        self.router = ReplicatedLinear(config.hidden_size, config.num_local_experts,
                                       bias=True, dtype=dtype, rngs=rngs)
        self.experts = GptOssRoutedExperts(config, dtype=dtype, rngs=rngs)

    def routing_weights(self, hidden: jax.Array):
        """``(weights [T, E] dense, index [T, k])`` from the router logits.

        THE SOFTMAX IS OVER THE SELECTED EXPERTS ONLY, which is vLLM's
        ``renormalize=True`` and `transformers`' ``softmax(router_top_value)``:
        each token's four weights sum to 1. Softmaxing the full 32-logit row and
        then gathering gives weights summing to less than 1 and an MLP that is
        uniformly attenuated -- a port defect that looks like a tolerance
        problem.

        AT FLOAT32 OR WIDER, the same rule `attention/paged.py` states for its
        own softmax, applied to the second softmax this model has.
        ``promote_types`` and not a hard ``float32``, so a float64 caller is not
        silently narrowed. The SELECTION is unaffected: top-k reads the logits.

        ``jax.lax.top_k`` and ``torch.topk`` agree on ties by taking the LOWEST
        index, which is what keeps the two sides' selections identical when two
        logits collide -- possible in bfloat16, never observed on this
        checkpoint.
        """
        logits = self.router(hidden)
        accum = jnp.promote_types(logits.dtype, jnp.float32)
        values, index = jax.lax.top_k(logits, self.experts_per_token)
        probs = jax.nn.softmax(values.astype(accum), axis=-1)
        weights = jnp.zeros(logits.shape, accum).at[
            jnp.arange(logits.shape[0])[:, None], index].set(probs)
        return weights, index

    def __call__(self, hidden: jax.Array) -> jax.Array:
        weights, _ = self.routing_weights(hidden)
        return self.experts(hidden, weights)
