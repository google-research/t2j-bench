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

"""Vendored, standalone grade-time helpers for the inference task class (``_t2j/``).

Deliberately hermetic: at grade time these import only ``jax``, ``numpy``
(+ ``ml_dtypes``, which ships with jax) and the Python standard library — **never
torch, never the authoring engine**. A finalized instance therefore does
not silently change when the reference trees evolve.

``sandbox.py``, ``purity.py``, ``_arrays.py`` and ``similarity.py`` are
**byte-identical copies** of ``tasks/dpo_bloom/tests/_t2j/``, and
the harness holds them to that with ``filecmp``.
The copy is the point: the same torch-blocked import sandbox, the same comparator
taxonomy at the same tolerances, the same continuous [0,1] similarity toolbox
(``score_logits``/``S_tensor``/``power_mean``/``score_scalar``...), none of it
re-derived. Do not edit them here — fix the source and re-copy. ``similarity.py``
joined this set when the numeric stage moved from pass/fail parity
gates to per-check similarity scores (the training grader's numeric shape); it is a
pure math module (``numpy`` + ``._arrays`` only) and carries no inference-specific
divergence, so unlike ``compare.py`` it is a plain copy.

``compare.py`` IS THE ONE EXCEPTION, and it must not be "restored" to a byte copy.
It carries exactly one deliberate divergence: ``candidate_callable_params`` resolves
``Class.method`` as well as a top-level function, via ``_bound_method_param_info``.
The training original scans only ``tree.body``, which is enough there — all four of
its graded entry points are module-level — but ten of the nineteen signature targets
this class freezes are methods (``Engine.prefill``, the ``BlockPool`` surface,
``ModelConfig.from_hf``), and the inherited lookup reported "callable not found" for
every one of them against the CORRECT reference. Re-copying the training file would
put those ten false failures back. The same defect exists upstream across 45
instances; fixing it there is out of scope here.

The divergence is machine-checked and confined to those two symbols: the harness's
copy check names them in its allowed-divergence list and compares
every other top-level statement of the two files, so a second, unrelated edit fails
there. Any further change to this file belongs upstream and gets re-copied.

Inference-specific and therefore NOT copies: ``golden_manifest.py`` (mode-keyed, with
a directory-token grammar for the L3 batteries), ``candidate.py`` and ``_formulas.py``
(no training-loss formulas exist on this side — there is no backward pass).
"""

VERSION = "t2j-helpers/0.1.0"
