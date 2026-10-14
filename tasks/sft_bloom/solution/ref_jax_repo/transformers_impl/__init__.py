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

"""Vendored native-JAX (flax) model implementation for this repo.

Mirror of ``torch_orig/transformers_impl``: ``configuration_bloom`` (``BloomConfig``),
``modeling_bloom`` (the flax model), and the ``configuration``/``modeling`` re-export
shims. The only differences from the torch version are the JAX adjustments documented in
``modeling_bloom`` — a flax ``__call__`` over an external parameter pytree, and
``model``-prefixed parameter paths mirroring torch's module tree.
"""
