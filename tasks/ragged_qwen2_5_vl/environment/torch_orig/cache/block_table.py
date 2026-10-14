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

"""Page pool, block table and slot mapping.

The allocator is deliberately the simplest thing that is still a real
allocator: a free list popped in ascending order. It has to be DETERMINISTIC,
because the frozen scenario pins which pages a request gets -- if
the reference and a candidate allocate differently their cache tensors differ
legitimately, and a correct candidate would fail test_kv_cache_write.
"""

from __future__ import annotations

import torch


class BlockPool:
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

        Deliberately plain Python (no torch, no array ops) so the JAX
        BlockPool can express this identically.
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
                     num_tokens: int) -> torch.Tensor:
        """Flat cache slot for each of `num_tokens` tokens starting at `start_pos`."""
        pos = torch.arange(start_pos, start_pos + num_tokens, dtype=torch.long)
        page_of = pos // self.page_size
        offset = pos % self.page_size
        table = torch.tensor(page_ids, dtype=torch.long)
        return table[page_of] * self.page_size + offset

    def block_table_row(self, page_ids: list[int], pages_per_seq: int) -> torch.Tensor:
        """One request's page list, right-padded with 0 to the compiled width.

        Padding with 0 is safe because every read is masked by seq_lens; page 0
        is never attended through a padded entry.
        """
        row = torch.zeros(pages_per_seq, dtype=torch.int32)
        row[: len(page_ids)] = torch.tensor(page_ids, dtype=torch.int32)
        return row
