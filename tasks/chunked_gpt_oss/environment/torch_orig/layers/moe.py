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

"""The routed mixture-of-experts block, de-vendored from vLLM's `gpt_oss.py`.

Two classes, both named after their vLLM originals
(`vllm/model_executor/models/gpt_oss.py`: ``GptOssRoutedExperts`` at :192,
``MLPBlock`` at :318), because ``test_reference_units`` grades a port on mirroring
the torch side's attribute names and the torch side is ours to author. See
`modeling.py`'s docstring for why this family follows vLLM's naming rather than
`core_llama3`'s ``{Family}MLP`` convention.

THE WEIGHTS STAY PACKED. This is the decision the whole family rests on, so it is
stated where the parameters are declared:
`openai/gpt-oss-20b` is 20.9B parameters, 19.1B of them in these experts. The
grader caches a native-dtype model AND an fp32 model, and
``test_serve_end_to_end`` drives ``serve.py::_build`` for a third -- against
``gpus = 1``. At bf16 that is 41.8 + 83.6 + 41.8 GB on an 80 GB H100, and the fp32
copy alone does not fit. Held as MXFP4 the resident cost is ~14 GB a copy and all
three coexist. So :class:`GptOssRoutedExperts` owns ``uint8`` blocks and scales
and dequantises INSIDE the forward, one layer at a time; nothing dequantised is
ever cached on the module.

WHAT THIS COSTS, STATED PLAINLY. The forward contracts over the whole expert axis
and applies the routing weights afterwards, rather than gathering the four
selected experts per token. That is dense in FLOPs -- 32 experts of work where 4
are needed -- and it is deliberate:

  * It is what the released reference does. `transformers`' own eager
    ``GptOssExperts`` visits every expert the batch hit, and its batched sibling
    contracts over all of them.
  * True FLOP-sparsity needs a grouped GEMM over a dynamic number of rows per
    expert. JAX has no static-shape spelling of that, so the two sides of this
    port could not be the same algorithm, and a reference whose two halves take
    different paths cannot be read as one definition.
  * The MEMORY decision -- the one that makes the family possible -- is
    unaffected: dequantising is per layer and transient either way, because a
    prefill of any real length routes at least one token to every expert.

`_zero_unselected` exists for the harness's MoE routing test, which proves the
routing is genuinely top-k by zeroing every expert no token picked and requiring
the output not to move by a single ulp.
"""

from __future__ import annotations

import torch
from torch import nn

from config import ModelConfig
from layers.activation import ClampedSwiGLU
from layers.linear import ReplicatedLinear
from quant.mxfp4 import BYTES_PER_BLOCK, VALUES_PER_BLOCK, dequantize_mxfp4


def _blocks_shape(rows: int, contracted: int) -> tuple[int, int, int]:
    """``(rows, groups, bytes)`` for one expert's packed projection."""
    return (rows, contracted // VALUES_PER_BLOCK, BYTES_PER_BLOCK)


class GptOssRoutedExperts(nn.Module):
    """The 32 experts of one layer, resident as MXFP4.

    Parameter names are the checkpoint's own -- ``gate_up_proj_blocks``,
    ``gate_up_proj_scales``, ``gate_up_proj_bias`` and the ``down_proj`` triple --
    so ``model/loader.py`` copies tensor for tensor with no rename table, and
    ``test_param_correspondence`` grades a key set a reader can find in the
    safetensors index.

    The two ``_blocks`` tensors are ``uint8`` and are NOT floating point, so they
    are invisible to `test_param_norms`' L2 reduction and to any ``.to(dtype)``
    the harness performs. Both are correct: an 8-bit code table has no norm to
    compare and no dtype to cast.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
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

        # gate_up contracts over `hidden` and emits 2 * inter rows (gate and up
        # INTERLEAVED -- see ClampedSwiGLU); down contracts over `inter`.
        self.gate_up_proj_blocks = nn.Parameter(
            torch.empty(experts, *_blocks_shape(2 * inter, hidden), dtype=torch.uint8),
            requires_grad=False)
        self.gate_up_proj_scales = nn.Parameter(
            torch.empty(experts, 2 * inter, hidden // VALUES_PER_BLOCK,
                        dtype=torch.uint8),
            requires_grad=False)
        self.gate_up_proj_bias = nn.Parameter(torch.empty(experts, 2 * inter))
        self.down_proj_blocks = nn.Parameter(
            torch.empty(experts, *_blocks_shape(hidden, inter), dtype=torch.uint8),
            requires_grad=False)
        self.down_proj_scales = nn.Parameter(
            torch.empty(experts, hidden, inter // VALUES_PER_BLOCK,
                        dtype=torch.uint8),
            requires_grad=False)
        self.down_proj_bias = nn.Parameter(torch.empty(experts, hidden))

        self.act_fn = ClampedSwiGLU(limit=config.swiglu_limit)

    # --- the dequant, never cached ----------------------------------------- #
    def dequantized_gate_up(self, dtype: torch.dtype) -> torch.Tensor:
        """``[E, hidden, 2 * inter]``. Fresh every call, ON PURPOSE.

        Caching it would put 2.1 GB (fp32) of expert weights per layer back on
        the device permanently and defeat the packing.
        """
        return dequantize_mxfp4(self.gate_up_proj_blocks, self.gate_up_proj_scales,
                                dtype=dtype)

    def dequantized_down(self, dtype: torch.dtype) -> torch.Tensor:
        """``[E, inter, hidden]``. Both axes are 2880 on this checkpoint, which is
        exactly why `quant/mxfp4.py`'s ``swap_last2`` has its own test."""
        return dequantize_mxfp4(self.down_proj_blocks, self.down_proj_scales,
                                dtype=dtype)

    def _zero_unselected(self, expert_ids) -> None:
        """Make the named experts contribute nothing. Test affordance only.

        Code 0x0 is +0.0 at every scale, so zeroing the packed bytes zeroes the
        weights exactly; the biases go with them.

        UNDERSCORED, AND THAT IS A GRADING DECISION RATHER THAN A STYLE ONE.
        ``compare.public_class_methods`` collects every non-underscore method in a
        class body, and ``interface.json['methods']`` is the INTERSECTION of the
        torch side's and the reference's -- so a public name here would land on
        both sides, enter the frozen contract, and ``test_interface`` would then
        require every candidate to implement a helper that exists only for a
        de-vendoring test nothing grades.
        """
        index = torch.as_tensor(sorted(expert_ids), dtype=torch.long)
        with torch.no_grad():
            for param in (self.gate_up_proj_blocks, self.gate_up_proj_scales,
                          self.gate_up_proj_bias, self.down_proj_blocks,
                          self.down_proj_scales, self.down_proj_bias):
                param[index] = 0

    def forward(self, hidden: torch.Tensor,
                routing_weights: torch.Tensor) -> torch.Tensor:
        """``hidden [T, H]`` and ``routing_weights [T, E]`` -> ``[T, H]``.

        ``routing_weights`` is DENSE over the experts and zero off the top-k, so
        this signature carries the whole routing decision as data. That is what
        lets ``test_reference_units`` replay this module on a frozen input: an
        argument the capture recorded, not a decision re-made inside.
        """
        dtype = hidden.dtype
        gate_up = torch.einsum(
            "th,ehf->tef", hidden, self.dequantized_gate_up(dtype)
        ) + self.gate_up_proj_bias
        activated = self.act_fn(gate_up)
        out = torch.einsum(
            "tei,eih->teh", activated, self.dequantized_down(dtype)
        ) + self.down_proj_bias
        return torch.einsum("te,teh->th", routing_weights.to(out.dtype), out)


class MLPBlock(nn.Module):
    """Router + experts, mirroring vLLM's ``MLPBlock``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        # ReplicatedLinear, WITH a bias. `modules_to_not_convert` keeps the whole
        # router out of the MXFP4 conversion -- 32 logits decide which 19.1B
        # parameters run, so the router is the one place in this layer where
        # 4-bit weights would be a bad trade.
        self.router = ReplicatedLinear(config.hidden_size, config.num_local_experts,
                                       bias=True)
        self.experts = GptOssRoutedExperts(config)

    def routing_weights(self, hidden: torch.Tensor):
        """``(weights [T, E] dense, index [T, k])`` from the router logits.

        THE SOFTMAX IS OVER THE SELECTED EXPERTS ONLY, which is vLLM's
        ``renormalize=True`` and `transformers`' ``softmax(router_top_value)``:
        each token's four weights sum to 1. Softmaxing the full 32-logit row and
        then gathering gives weights summing to less than 1 and an MLP that is
        uniformly attenuated -- a port defect that looks like a tolerance
        problem.

        AT FLOAT32 OR WIDER, and that IS a departure from `transformers`, which
        softmaxes in the model dtype. It is the same rule `attention/paged.py`
        states for its own softmax ("scores and softmax in fp32"), applied to the
        second softmax this model has; over four values in bfloat16 the
        renormalisation would otherwise carry 8 mantissa bits into a weight that
        multiplies the entire MLP output. ``promote_types`` and not a hard
        ``float32``, so a float64 caller is not silently narrowed -- which is how
        the harness's MoE test gets a band tight enough to discriminate.
        The SELECTION is unaffected: top-k reads the logits, not the
        probabilities.
        """
        logits = self.router(hidden)
        accum = torch.promote_types(logits.dtype, torch.float32)
        values, index = torch.topk(logits, self.experts_per_token, dim=-1)
        probs = torch.softmax(values.to(accum), dim=-1)
        weights = torch.zeros(logits.shape, dtype=accum,
                              device=logits.device).scatter(-1, index, probs)
        return weights, index

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weights, _ = self.routing_weights(hidden)
        return self.experts(hidden, weights)
