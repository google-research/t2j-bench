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

"""Embedding + LM head, mirroring environment/torch_orig/layers/vocab_embed.py.

VocabParallelEmbedding shards the vocab across ranks and pads it to a multiple
of the world size. At TP=1 there is no sharding and no padding, so this is a
plain lookup -- but the CLASS NAME and the `weight` parameter name are kept
because the mirror check and the loader both key off them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class VocabParallelEmbedding(nnx.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 dtype=jnp.float32, rngs: nnx.Rngs | None = None) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        key = rngs.params() if rngs is not None else jax.random.key(0)
        self.weight = nnx.Param(
            nnx.initializers.zeros(key, (num_embeddings, embedding_dim), dtype)
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        return self.weight[...][input_ids]


class ParallelLMHead(VocabParallelEmbedding):
    """The unembedding. Tied to the embedding on Llama-3.2-1B."""

    def project(self, hidden: jax.Array) -> jax.Array:
        return hidden @ self.weight[...].T
