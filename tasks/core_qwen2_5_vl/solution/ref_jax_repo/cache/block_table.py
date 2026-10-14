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

"""Page pool, block table and slot mapping, mirroring torch_orig's.

Split in two on purpose:

  * The two INDEX functions are free functions here as well as methods, so
    traced code can build a slot mapping without constructing a pool. They
    differ from torch_orig's by exactly the return type and nothing else.
  * BlockPool -- page ALLOCATION -- is pure Python control flow over a free
    list. Its allocate/free/num_free bodies are IDENTICAL to torch_orig's, and
    that identity is load-bearing: the frozen scenario pins which pages a
    request gets, so if the two sides allocated differently their
    cache tensors would differ legitimately and a correct candidate would fail
    test_kv_cache_write. The harness compares the three
    bodies by AST and fails on any divergence.

The allocator is deliberately the simplest thing that is still a real
allocator: a free list popped in ascending order, and therefore DETERMINISTIC.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def slot_mapping(page_ids: list[int], start_pos: int, num_tokens: int,
                 page_size: int) -> jax.Array:
    """Flat cache slot for each of `num_tokens` tokens starting at `start_pos`."""
    pos = jnp.arange(start_pos, start_pos + num_tokens)
    table = jnp.asarray(page_ids, dtype=jnp.int32)
    return table[pos // page_size] * page_size + (pos % page_size)


def block_table_row(page_ids: list[int], pages_per_seq: int) -> jax.Array:
    """One request's page list, right-padded with 0 to the compiled width.

    Padding with 0 is safe because every read is masked by seq_lens; page 0 is
    never attended through a padded entry.
    """
    row = jnp.zeros(pages_per_seq, dtype=jnp.int32)
    return row.at[: len(page_ids)].set(jnp.asarray(page_ids, dtype=jnp.int32))


class BlockPool:
    """Deterministic free-list allocator.

    allocate/free/num_free are the same logic as
    torch_orig/cache/block_table.py::BlockPool, checked by AST equality in CI;
    only the two index methods differ, and only in return type.
    """

    def __init__(self, num_pages: int, page_size: int) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self._free = list(range(num_pages))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, num_tokens: int) -> list[int]:
        need = (num_tokens + self.page_size - 1) // self.page_size
        if need > len(self._free):
            raise RuntimeError(
                f"out of KV pages: need {need}, {len(self._free)} free "
                f"(pool has {self.num_pages} pages of {self.page_size})"
            )
        pages, self._free = self._free[:need], self._free[need:]
        return pages

    def free(self, pages: list[int]) -> None:
        """Return `pages` to the free list, which stays sorted ascending.

        Validated rather than trusted: an out-of-range page silently entering
        the free list is handed out later as an out-of-bounds cache index, and
        a double free hands the same page to two requests. Both surface as a
        wrong-logits mismatch a long way from the cause. The whole batch is
        checked before anything is appended, so a rejected call leaves the pool
        untouched.

        Plain Python on both sides -- no jnp, no array ops -- which is what
        makes the AST equality with torch_orig's version expressible at all.
        """
        for i, page in enumerate(pages):
            if page < 0 or page >= self.num_pages:
                raise ValueError(
                    f"page {page} out of range for a pool of {self.num_pages} pages"
                )
            if page in self._free or page in pages[:i]:
                raise ValueError(f"page {page} is already free (double free)")
        self._free = sorted(self._free + list(pages))

    def slot_mapping(self, page_ids: list[int], start_pos: int,
                     num_tokens: int) -> jax.Array:
        """Flat cache slot for each of `num_tokens` tokens starting at `start_pos`."""
        return slot_mapping(page_ids, start_pos, num_tokens, self.page_size)

    def block_table_row(self, page_ids: list[int], pages_per_seq: int) -> jax.Array:
        """One request's page list, right-padded with 0 to the compiled width."""
        return block_table_row(page_ids, pages_per_seq)
