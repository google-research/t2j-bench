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

"""Paged KV cache: page pool, block table, slot mapping, cache tensor.

Exports exactly torch_orig's three names. block_table_row and slot_mapping also
exist as free functions in cache.block_table -- so traced code can build a slot
mapping without constructing a pool -- but they are reached through the module,
not re-exported here, because the package surface is part of what the two sides
mirror.
"""

from cache.block_table import BlockPool
from cache.kv_cache import make_kv_cache, write_kv

# `identity_layer_slot` / `identity_layer_writes` belong here even though nothing
# imports them `from cache` today: the module docstring calls the layer -> (group,
# slot) map the generalisation point, and a package surface that omits the two
# names that ARE that map tells the next reader the opposite.
__all__ = ["BlockPool", "identity_layer_slot", "identity_layer_writes",
           "make_kv_cache", "write_kv"]
