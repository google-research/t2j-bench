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

"""Embedding + LM head, de-vendored from vLLM at TP=1.

VocabParallelEmbedding shards the vocab across ranks and pads it to a multiple
of the world size. At TP=1 there is no sharding and no padding, so this is a
plain lookup -- but the CLASS NAME and the `weight` parameter name are kept
because the mirror check and the loader both key off them.
"""

from __future__ import annotations

import torch
from torch import nn


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self.weight)


class ParallelLMHead(VocabParallelEmbedding):
    """The unembedding. Tied to the embedding on Llama-3.2-1B."""

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden, self.weight)
