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

"""Master copies of the vendored, standalone grade-time helpers (``_t2j/``).

These modules are copied verbatim by the test-case generator into every
instance's ``test/_t2j/`` directory and version-stamped. They are deliberately
hermetic: at grade time they import only ``jax``, ``numpy`` (+ ``ml_dtypes``,
which ships with jax) and the Python standard library — **never torch, never
the authoring engine**. A finalized instance therefore does not
silently change when the engine evolves. These copies are kept aligned with
their engine sources bit-for-bit by a parity test.

The comparison math (``compare.py``) mirrors the engine's ``metrics.py``; the
candidate-driving math (``_formulas.py``) mirrors its jax-path ``runtime.py``
helpers; ``purity.py``/``sandbox.py`` mirror their engine namesakes;
``candidate.py`` drives a candidate ``jax_repo`` in a torch-blocked subprocess
against frozen inputs.
"""

VERSION = "t2j-helpers/0.1.0"
