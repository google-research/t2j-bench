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

"""Prefix cache: page sharing above an untouched BlockPool.

`BlockPool` (block_table.py) defends exactly one invariant -- a page is never
handed to two owners at once -- with a seeded control, allocator_reuses_live_
page, and the file is byte-identical across every instance in this class.
Ref-counting inside the pool was considered and rejected: that forks a shared
file per mode and either breaks that control or silently changes what it
asserts.

Instead this class sits ABOVE the pool as its single owner for shared pages:
it allocates a page from the pool once, refcounts it here, and lends the same
physical id into every request's block table that shares it. The pool never
sees more than one allocate/free per page per lifetime, so its invariant
holds literally and the control keeps applying unchanged. This is upstream's
shape too -- a manager above a pool.

Four rules:
  * Hash TOKEN IDS, never KV bytes -- a bf16 KV hash is not bit-reproducible
    across the two reference implementations, so a golden keyed on one could
    never be frozen on the other.
  * The hash is `hashlib.blake2b` over the ids packed little-endian, not the
    builtin `hash()`, whose per-process salt would make a frozen golden
    irreproducible from run to run.
  * PAGE GRANULARITY: only a full page is shared. A partial trailing page
    belongs to its own request alone.
  * Eviction is DETERMINISTIC: refcount-0 cached pages are dropped in
    insertion order, never by dict iteration order or timing, because the
    class freezes exact page ids in scenario.json -- a policy that depends on
    either presents as an unstable golden, not as an eviction bug.

No array library is needed here: token ids arrive as plain Python ints and
every value this class produces (page ids, counts) is a plain Python int, so
this module is identical on both sides of the port.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from cache.block_table import BlockPool


class PrefixCache:
    """Shares full pages of KV cache across requests with an identical prefix.

    A page is in exactly one of two states while it is resident in the pool:
    CACHED (it is a completed full page, keyed in `_by_hash`/`_page_hash`,
    reusable by any future request whose prefix matches) or PRIVATE (a
    partial trailing page, owned solely by the request that allocated it).
    `_refcount` tracks live holders of either kind; only a CACHED page can
    have a refcount of 0 while still resident, because eviction -- not
    release -- is what returns a refcount-0 cached page to the pool.
    """

    def __init__(self, pool: BlockPool, page_size: int) -> None:
        self.pool = pool
        self.page_size = page_size
        # full-page token hash -> the physical page id holding it
        self._by_hash: dict[bytes, int] = {}
        # the reverse of _by_hash, and how release() tells a cached (shared)
        # page apart from a private one: a page id is cached iff it is a key
        # here.
        self._page_hash: dict[int, bytes] = {}
        # live references to a page, cached or private
        self._refcount: dict[int, int] = {}
        # cached page ids in the order they were cached, oldest first --
        # eviction order.
        self._insertion: list[int] = []
        # this request's current page ids, so release() knows what to touch
        self._owners: dict[object, list[int]] = {}
        self._hits = 0
        self._misses = 0

    def _hash_page(self, page_token_ids: Sequence[int]) -> bytes:
        """Stable hash of one full page's token ids.

        `hashlib.blake2b`, not the builtin `hash()`: the builtin salts by
        process, which would make a frozen scenario's page ids irreproducible
        the next time the suite runs.
        """
        packed = b"".join(int(t).to_bytes(8, "little") for t in page_token_ids)
        return hashlib.blake2b(packed).digest()

    def acquire(self, token_ids: Sequence[int],
               request_id: object) -> tuple[list[int], int]:
        """Return this request's COMPLETE page list and its cached-token count.

        `page_ids` is the full block table for this request -- any shared
        (cache-hit) pages first, in prompt order, followed by the freshly
        allocated pages that cover the unmatched suffix -- not just the
        subset that hit. The caller (the engine's block table) needs the
        whole list either way, and a hits-only list would force it to
        reassemble the rest itself.

        Walks the prompt's full pages in order and stops at the first miss --
        a prefix is a prefix, so a hit past a miss is not reachable. Tokens
        past the matched prefix (the rest of the full pages, plus a partial
        trailing page if there is one) are freshly allocated from the pool;
        the newly completed full pages among them are cached for future
        requests, the partial trailing page never is.
        """
        page_size = self.page_size
        num_full_pages = len(token_ids) // page_size

        page_ids: list[int] = []
        n_cached_tokens = 0
        matched_pages = 0
        for i in range(num_full_pages):
            chunk = token_ids[i * page_size:(i + 1) * page_size]
            page_id = self._by_hash.get(self._hash_page(chunk))
            if page_id is None:
                self._misses += 1
                break
            self._hits += 1
            self._refcount[page_id] += 1
            page_ids.append(page_id)
            n_cached_tokens += page_size
            matched_pages += 1

        remaining = len(token_ids) - matched_pages * page_size
        if remaining:
            new_pages = self._allocate(remaining)
            num_new_full = remaining // page_size
            for j, page_id in enumerate(new_pages):
                page_ids.append(page_id)
                self._refcount[page_id] = 1
                if j < num_new_full:
                    start = (matched_pages + j) * page_size
                    chunk = token_ids[start:start + page_size]
                    page_hash = self._hash_page(chunk)
                    self._by_hash[page_hash] = page_id
                    self._page_hash[page_id] = page_hash
                    self._insertion.append(page_id)

        self._owners[request_id] = page_ids
        return page_ids, n_cached_tokens

    def release(self, request_id: object) -> None:
        """Drop this request's hold on its pages.

        A cached (shared) page is only ever decremented here -- it is freed
        to the pool solely by eviction, never by a holder's release, because
        another live request (or a future prefix hit) may still need it. A
        private (uncached, partial trailing) page has exactly one holder, so
        it goes straight back to the pool the moment that holder releases.
        """
        page_ids = self._owners.pop(request_id, [])
        to_free: list[int] = []
        for page_id in page_ids:
            self._refcount[page_id] -= 1
            if self._refcount[page_id] == 0 and page_id not in self._page_hash:
                del self._refcount[page_id]
                to_free.append(page_id)
        if to_free:
            self.pool.free(to_free)

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def _allocate(self, num_tokens: int) -> list[int]:
        """`pool.allocate`, evicting refcount-0 cached pages on exhaustion."""
        while True:
            try:
                return self.pool.allocate(num_tokens)
            except RuntimeError:
                if not self._evict_one():
                    raise

    def _evict_one(self) -> bool:
        """Free the oldest-inserted refcount-0 cached page. False if none."""
        for i, page_id in enumerate(self._insertion):
            if self._refcount.get(page_id, 0) == 0:
                del self._insertion[i]
                page_hash = self._page_hash.pop(page_id)
                del self._by_hash[page_hash]
                del self._refcount[page_id]
                self.pool.free([page_id])
                return True
        return False
