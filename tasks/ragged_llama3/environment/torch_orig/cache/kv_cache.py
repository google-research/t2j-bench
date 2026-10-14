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

"""The paged KV cache tensors and their write path.

THE CACHE IS A CONTAINER: a tuple holding one tensor per distinct KV geometry,
each laid out [n_slots, num_pages, page_size, 2, num_kv_heads, head_dim].

EVERY FAMILY SHIPPED TODAY IS THE ONE-GROUP CASE, and that case is the identity:
the container's single tensor is bit-for-bit the rectangular cache this module
used to return. A layer reaches its own tensor through
`layer_slot[layer_idx] -> (group, slot)`, which is `(0, layer_idx)` for all of
them (:func:`identity_layer_slot`).

Upstream tpu-inference uses
[P, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim] and asserts
head_dim % 128 == 0. The packing dimension and that assertion are TPU lane
alignment, not semantics -- an H100 has no such constraint. Dropping them keeps
the tensor legible and keeps head_dim=64 (Llama-3.2-1B's actual head_dim)
expressible at all. Both sides use the layout above, so it is the contract
test_kv_cache_spec grades.
"""

from __future__ import annotations

import torch


def make_kv_cache(groups, num_pages: int, page_size: int,
                  dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """One tensor per DISTINCT (num_kv_heads, head_dim) geometry.

    `groups` is `((n_slots, num_kv_heads, head_dim), ...)`. EVERY FAMILY SHIPPED
    TODAY PASSES EXACTLY ONE GROUP whose `n_slots` is `num_layers`, and the tensor
    it produces is bit-for-bit the rectangular cache this function used to
    return -- that identity is what keeps thirteen instances' goldens
    byte-identical across this change.

    More than one group exists for a family whose layers do not agree on their
    KV geometry. gemma4-E2B is the case this was written for: head_dim 256 on its
    sliding layers and 512 on its full ones, so one rectangular tensor cannot
    express it at all.

    `n_slots` IS NOT `num_layers` in general. A family that SHARES a KV slot
    between layers allocates fewer slots than it has layers, and `layer_slot`
    maps the one to the other; `gemma4-E2B` allocates 15 slots for 35 layers.
    """
    if not groups:
        raise ValueError(
            "make_kv_cache needs at least one group: a cache with no tensors "
            "threads through every layer without error and holds nothing, so "
            "'no groups' must not be spelled the same way as 'one group'.")
    return tuple(
        torch.zeros((n_slots, num_pages, page_size, 2, num_kv_heads, head_dim),
                    dtype=dtype)
        for n_slots, num_kv_heads, head_dim in groups
    )


def identity_layer_slot(num_layers: int) -> tuple[tuple[int, int], ...]:
    """`layer_idx -> (group, slot)` for a family with one geometry and no sharing.

    Every shipped family. Named rather than inlined at its call sites so that the
    thing gemma4 generalises is a VALUE, not a pattern each engine and each
    attention module re-derives.
    """
    return tuple((0, i) for i in range(num_layers))


def identity_layer_writes(num_layers: int) -> tuple[bool, ...]:
    """`True` for every layer: each writes the slot it reads. Every shipped family.

    A `False` entry belongs to a KV-SHARED layer, which attends against another
    layer's slot and must not overwrite it.
    """
    return (True,) * num_layers


def _require_slot(cache, group: int, slot: int) -> None:
    """Refuse a `(group, slot)` this container cannot hold -- ON BOTH FRAMEWORKS.

    THE ONE OUT-OF-RANGE ASYMMETRY BETWEEN THE TWO SIDES, and it is not a style
    difference. MEASURED, jax 0.11 / torch 2.10 on the same `(3, 4)` array:

        torch   t[7]                -> IndexError
        jax     a[7]                -> row 2, silently. No error.
        jax     a.at[7].set(99.0)   -> the write is DROPPED. No error.

    So a `layer_slot` that outruns its container CRASHES the torch side and hands
    the JAX side a plausible wrong tensor whose cache write also vanished. The
    same defect then reads as a torch bug on one framework and as a numeric
    parity failure on the other, and neither symptom points at the map.

    THE FAMILY THIS IS FOR IS THE ONE THE CONTAINER EXISTS FOR. gemma4-E2B
    allocates 15 slots for 35 layers, so an implementer who widens `engine.py`'s
    groups and leaves this module's `identity_layer_slot` in place asks for slot
    34 of a 15-slot group on the very first run. `engine.py`'s
    `validate_metadata` already names this clamp hazard for the block-table
    gather; this is the same hazard at the cache address, and the same answer.

    FREE UNDER JIT. `group` and `slot` are static Python ints and `len(cache)` /
    `.shape[0]` are static too, so this runs at TRACE time and emits nothing.
    """
    if not 0 <= group < len(cache):
        raise IndexError(
            f"KV cache group {group} is out of range: the container holds "
            f"{len(cache)} group(s). The layer -> (group, slot) map addresses a "
            f"group that make_kv_cache was never asked to allocate.")
    n_slots = int(cache[group].shape[0])
    if not 0 <= slot < n_slots:
        raise IndexError(
            f"KV cache slot {slot} is out of range: group {group} holds "
            f"{n_slots} slot(s). A family that shares KV slots allocates fewer "
            f"slots than it has layers, so the groups passed to make_kv_cache "
            f"and the layer -> (group, slot) map have to move together.")


def write_kv(cache_layer: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    """Scatter this step's K and V into one layer's pages, in place.

    cache_layer  [num_pages, page_size, 2, num_kv_heads, head_dim]
    key, value   [num_tokens, num_kv_heads, head_dim]
    slot_mapping [num_tokens]  flat slot = page_id * page_size + offset
    """
    num_pages, page_size = cache_layer.shape[0], cache_layer.shape[1]
    flat = cache_layer.view(num_pages * page_size, 2, *cache_layer.shape[3:])
    flat[slot_mapping, 0] = key.to(flat.dtype)
    flat[slot_mapping, 1] = value.to(flat.dtype)
