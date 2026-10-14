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

"""RMSNorm, de-vendored from vLLM's forward_native path.

vLLM dispatches to a fused CUDA kernel and keeps forward_native as the
non-CUDA fallback. We take forward_native: it IS the semantic definition, and
it makes torch_orig legible and device-portable. The fused kernel is a
performance detail, not part of the equivalence target.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, optionally fused with a residual add.

    The two-argument form returns ``(normed, new_residual)`` and is what the
    decoder layer uses, so the residual stream is threaded explicitly rather
    than recomputed -- exactly as vLLM does it.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            x = x + residual
            residual = x
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        return x if residual is None else (x, residual)
