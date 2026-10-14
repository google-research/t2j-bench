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

"""The reference math, graded against hand-computed answers.

Needs no checkpoint and no GPU: these are closed-form properties, so they run in
CI and in the torch-free verifier alike. Each one is written to FAIL against the
plausible wrong version of the function it covers -- an uncentered cosine, a mean
instead of a minimum, a KL with the arguments the other way round, a
``reference_paged_attention_numpy`` that forgets the causal term.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parents[3] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from _t2j import _formulas as F  # noqa: E402


# --- centered_cosine -------------------------------------------------------- #
def test_centered_cosine_ignores_the_per_token_offset_a_raw_cosine_would_see():
    """Logits are defined up to a per-row constant, so a shift must not move it.

    The discriminating half is the last assertion: on two rows that share a large
    offset an UNCENTERED cosine reads 0.999 even when their directions are exactly
    opposed, so a function that skipped the centring would pass the first two
    assertions and still be blind to the disagreement the third one names.
    """
    base = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32)
    assert F.centered_cosine(base, base + 50.0) == pytest.approx(1.0, abs=1e-6)

    ref = base + 50.0
    opposed = -base + 50.0
    assert F.centered_cosine(ref, opposed) == pytest.approx(-1.0, abs=1e-5)
    raw = float(np.dot(ref.ravel(), opposed.ravel())
                / (np.linalg.norm(ref) * np.linalg.norm(opposed)))
    assert raw > 0.99, "the uncentered cosine must be the thing this test rejects"


def test_centered_cosine_reports_the_WORST_row_not_the_average():
    """One broken position in a long prompt is the bug, and a mean hides it."""
    good = np.tile(np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32), (8, 1))
    cand = good.copy()
    cand[3] = -cand[3]
    assert F.centered_cosine(good, cand) == pytest.approx(-1.0, abs=1e-6)
    assert np.mean([F.centered_cosine(good[i], cand[i]) for i in range(8)]) > 0.7


def test_centered_cosine_refuses_a_shape_mismatch_rather_than_broadcasting():
    assert F.centered_cosine(np.zeros((2, 4)), np.zeros((2, 5))) == 0.0


def test_centered_cosine_scores_a_degenerate_candidate_ZERO_not_one():
    """A candidate with no direction must not score maximum agreement.

    THIS WAS A REAL FALSE NEGATIVE, not a hypothetical. The clause used to be
    ``np.where(denom > 0, ..., 1.0)``, so a zero centred norm on EITHER side
    returned 1.0. On core_qwen2 the absolute clause of ``test_hidden_states`` is
    vacuous at the first three layers, leaving this cosine as the only guard --
    and a port emitting ``zeros_like`` placeholders there passed all 19 numeric
    and behavioral checks, with both clauses maximally satisfied at once.

    Every degenerate candidate below centres to the zero row, which is exactly
    why the offset case matters: ``centered_cosine`` is deliberately blind to a
    per-token constant, so a candidate that emits ONLY a constant has no
    direction left for this function to compare and must score 0.0.
    """
    ref = np.array([[1.0, -1.0, 0.5, -0.5], [2.0, 0.0, -1.0, 3.0]], dtype=np.float32)
    assert F.centered_cosine(ref, np.zeros_like(ref)) == 0.0
    assert F.centered_cosine(ref, np.full_like(ref, 7.0)) == 0.0
    # ...and asymmetrically: one good row does not rescue one degenerate row,
    # because the function reports the WORST row.
    half = np.stack([ref[0], np.zeros(4, dtype=np.float32)])
    assert F.centered_cosine(ref, half) == 0.0
    # The reverse direction too -- a degenerate REFERENCE against a real
    # candidate is equally uninformative and must not read as agreement.
    assert F.centered_cosine(np.zeros_like(ref), ref) == 0.0


def test_centered_cosine_still_calls_two_flat_rows_agreement():
    """The case the old `1.0` fallback was actually written for.

    Two rows that are both constant after centring genuinely carry the same (no)
    direction, and their VALUES are still graded by the absolute clause beside
    this one. Scoring them 0.0 would fail a correct port on a legitimately
    constant surface, which is the false positive this branch avoids.
    """
    assert F.centered_cosine(np.ones((2, 4)), np.full((2, 4), 3.0)) == 1.0
    assert F.centered_cosine(np.zeros((2, 4)), np.zeros((2, 4))) == 1.0


# --- softmax_kl ------------------------------------------------------------- #
def test_softmax_kl_is_zero_for_the_same_distribution_and_directional():
    ref = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
    assert F.softmax_kl(ref, ref) == pytest.approx(0.0, abs=1e-6)
    assert F.softmax_kl(ref, ref + 7.0) == pytest.approx(0.0, abs=1e-6)

    cand = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    forward = F.softmax_kl(ref, cand)
    backward = F.softmax_kl(cand, ref)
    assert forward > 0.5
    # KL(p||q) weights by the REFERENCE distribution. These two are mirror
    # images so the numbers coincide; what the test pins is that the reference
    # is the first argument, checked by an asymmetric third case below.
    assert forward == pytest.approx(backward, rel=1e-5)

    # Asymmetric case: KL weights by the REFERENCE distribution, so a uniform
    # reference against a peaked candidate pays for the candidate's near-zero
    # tails (~5.6 nats) while the reverse is bounded by log(3) (~1.1). Swapping
    # the two arguments in the implementation flips this inequality.
    peaked = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    flat = np.zeros((1, 3), dtype=np.float32)
    assert F.softmax_kl(flat, peaked) > 5.0
    assert F.softmax_kl(peaked, flat) < np.log(3.0) + 1e-3
    assert F.softmax_kl(flat, peaked) > F.softmax_kl(peaked, flat)


def test_softmax_kl_is_infinite_rather_than_small_on_broken_input():
    ref = np.zeros((1, 3), dtype=np.float32)
    assert F.softmax_kl(ref, np.zeros((1, 4))) == float("inf")
    assert F.softmax_kl(ref, np.array([[0.0, np.nan, 0.0]])) == float("inf")


# --- top1_agreement / logprob_of -------------------------------------------- #
def test_top1_agreement_counts_positions_not_values():
    ref = np.array([[3.0, 1.0], [1.0, 3.0], [3.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    cand = ref.copy()
    cand[2] = cand[2][::-1]
    assert F.top1_agreement(ref, ref) == 1.0
    assert F.top1_agreement(ref, cand) == 0.75
    # Scaling every logit changes every value and no argmax.
    assert F.top1_agreement(ref, ref * 100.0) == 1.0


def test_logprob_of_matches_the_closed_form_and_refuses_a_length_mismatch():
    logits = np.array([[0.0, np.log(3.0)]], dtype=np.float32)
    got = F.logprob_of(logits, [1])
    assert got.shape == (1,)
    assert float(got[0]) == pytest.approx(np.log(0.75), abs=1e-6)
    with pytest.raises(ValueError, match="token ids"):
        F.logprob_of(logits, [1, 0])


# --- reference_paged_attention_numpy ---------------------------------------- #
def _tiny_case(page_size=4, pages_per_seq=2, num_tokens=5, heads=2, kv_heads=1,
               head_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    num_pages = 6
    cache = rng.normal(size=(num_pages, page_size, 2, kv_heads, head_dim)).astype(np.float32)
    q = rng.normal(size=(num_tokens, heads, head_dim)).astype(np.float32)
    page_ids = [4, 1]                       # deliberately NOT ascending
    block_tables = np.array(page_ids, dtype=np.int32)
    return dict(
        query=q, kv_cache_layer=cache, block_tables=block_tables,
        seq_lens=np.array([num_tokens], dtype=np.int32),
        query_start_loc=np.array([0, num_tokens], dtype=np.int32),
        num_kv_heads=kv_heads, sm_scale=float(head_dim) ** -0.5,
    )


def test_the_numpy_reference_is_causal():
    """Row t must not depend on anything written after position t.

    Perturbing the KV at position 4 leaves rows 0..3 bit-unchanged and moves row
    4. A transcription that dropped the ``j <= q_pos[t]`` term fails on the first
    assertion; one that dropped the whole mask fails on both.
    """
    case = _tiny_case()
    base = F.reference_paged_attention_numpy(**case)

    perturbed = dict(case)
    cache = case["kv_cache_layer"].copy()
    # Flat slot 4 of this request lives on page_ids[1] == page 1, offset 0.
    cache[1, 0] += 10.0
    perturbed["kv_cache_layer"] = cache
    after = F.reference_paged_attention_numpy(**perturbed)

    np.testing.assert_array_equal(base[:4], after[:4])
    assert not np.allclose(base[4], after[4])


def test_the_numpy_reference_follows_the_block_table_rather_than_page_order():
    """Swap the two pages in the table AND in the cache -> identical output.

    A transcription that ignored ``block_tables`` and read pages 0..pps-1 would
    return something different for both, so the equality is the discriminator.
    """
    case = _tiny_case()
    swapped = dict(case)
    cache = case["kv_cache_layer"].copy()
    cache[[4, 1]] = cache[[1, 4]]
    swapped["kv_cache_layer"] = cache
    swapped["block_tables"] = np.array([1, 4], dtype=np.int32)
    np.testing.assert_allclose(
        F.reference_paged_attention_numpy(**case),
        F.reference_paged_attention_numpy(**swapped), rtol=0, atol=0)


def test_the_numpy_reference_matches_a_dense_sdpa_over_the_same_prefix():
    """An independent oracle: gather the pages by hand, run masked SDPA densely."""
    case = _tiny_case()
    q = case["query"]
    cache = case["kv_cache_layer"]
    page_size = cache.shape[1]
    pages = case["block_tables"]
    flat = cache[pages].reshape(-1, 2, cache.shape[3], cache.shape[4])
    n = int(case["seq_lens"][0])
    keys, values = flat[:n, 0, 0], flat[:n, 1, 0]      # kv_heads == 1
    scale = case["sm_scale"]
    want = np.empty_like(q)
    for t in range(n):
        for h in range(q.shape[1]):
            scores = (keys[: t + 1] @ q[t, h]) * scale
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            want[t, h] = probs @ values[: t + 1]
    np.testing.assert_allclose(
        F.reference_paged_attention_numpy(**case), want, rtol=1e-5, atol=1e-6)
    assert page_size == 4  # the case really does span two pages


def test_the_numpy_reference_refuses_padded_queries_and_under_allocation():
    case = _tiny_case()
    padded = dict(case)
    padded["query"] = np.concatenate([case["query"], case["query"][:1]], axis=0)
    with pytest.raises(ValueError, match="padded query tokens"):
        F.reference_paged_attention_numpy(**padded)

    short = dict(case)
    short["seq_lens"] = np.array([99], dtype=np.int32)
    with pytest.raises(ValueError, match="under-allocated"):
        F.reference_paged_attention_numpy(**short)


# --- param_stats ------------------------------------------------------------ #
def test_param_stats_is_layout_invariant_but_value_sensitive():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = F.param_stats(arr)
    b = F.param_stats(arr.T)
    assert a["norm"] == pytest.approx(b["norm"])
    assert a["shape"] == [3, 4] and b["shape"] == [4, 3]
    assert a["shape_sorted"] == b["shape_sorted"] == [3, 4]
    # A single corrupted element moves the norm -- which is the whole point.
    bad = arr.copy()
    bad[1, 1] = 100.0
    assert F.param_stats(bad)["norm"] != pytest.approx(a["norm"])
    assert F.param_stats(np.array([np.inf], dtype=np.float32))["finite"] is False


# --- _assert_logit_parity(self_comparison=...) ------------------------------ #
def test_a_flat_first_argument_cannot_buy_an_argmax_exemption():
    """`check_logits_strict` reads its decisive set from the FIRST argument.

    Four of five behavioral call sites pass a CANDIDATE array there, so a port with
    flat logits makes every gap <= argmax_margin, `decisive` all-False, and the argmax
    clause vacuous. The self-comparison mode must not have that property.

    THE FIXTURE MUST BE COSINE- AND KL-CLOSE, or this test proves nothing. An
    earlier draft used two rows whose only nonzero entries sat at DISJOINT
    indices (dot product exactly 0), so `check_logits_strict`'s own cosine gate
    failed on its own -- the `AssertionError` this test caught came from the
    generic `assert ok`, not from the `self_comparison` branch, and a no-op
    `self_comparison` (accept the kwarg, do nothing) would have raised the
    identical error. `pytest.raises(AssertionError)` cannot tell "the clause
    this test is about fired" from "some other clause fired instead", so the
    fixture below is CHECKED against the real comparator, not merely picked to
    look near-tie: a realistic base vector and a copy with its top-2 SWAPPED,
    with a gap (0.1) under `argmax_margin` (0.1875) so the golden position
    reads as a non-decisive near-tie -- which is exactly the shape the old
    (pre-fix) code passed vacuously, and cosine/KL close enough (checked below)
    that only the argmax clause can possibly still object.
    """
    TESTS_DIR = Path(__file__).resolve().parents[2]
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from test_numeric import _assert_logit_parity
    from _t2j import compare

    class _G:
        native_thresholds = compare.NumericThresholds(
            abs_tol=0.1, rel_tol=0.1, cosine_floor=0.98, max_kl=0.1,
            argmax_margin=0.1875)
        native_extras = {"centered_cosine_floor": 0.998}

    thresholds = _G.native_thresholds
    base = np.array([[10.0, 9.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    swapped = np.array([[9.9, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    assert int(base.argmax(-1)[0]) != int(swapped.argmax(-1)[0]), (
        "fixture drift: the two rows must actually disagree on their top-1 token")

    # DO NOT TAKE THE FIXTURE ON TRUST -- measure it against the real comparator.
    # If either assertion below ever fails, this fixture no longer isolates the
    # argmax clause and must be replaced, not patched around.
    ok, metrics = compare.check_logits_strict(base, swapped, thresholds)
    assert metrics["cosine_ok"] and metrics["kl_ok"], (
        "fixture drift: this pair must clear the cosine/KL gates on its own so "
        "only the argmax clause can distinguish the old and new behaviour "
        f"(got cosine_ok={metrics['cosine_ok']}, kl_ok={metrics['kl_ok']})")
    assert metrics["argmax_ok"] and metrics["n_decisive"] == 0, (
        "fixture drift: the swapped top-2 gap must be UNDER argmax_margin, so the "
        "OLD decisive-set-from-first-argument code reads this position as a "
        "non-decisive near-tie and passes it vacuously "
        f"(got argmax_ok={metrics['argmax_ok']}, n_decisive={metrics['n_decisive']})")
    # THE VACUOUS PASS ITSELF: `check_logits_strict` alone -- no self_comparison,
    # no argmax_margin override -- calls this pair fine despite the real top-1
    # disagreement above. This is the exemption the fix must close.
    assert ok is True

    # NEW BEHAVIOUR: self_comparison=True must catch what check_logits_strict
    # alone cannot.
    with pytest.raises(AssertionError):
        _assert_logit_parity(base, swapped, _G(), "native",
                             name="self", self_comparison=True)

    # THE POSITIVE CASE. Without this, the test above shows only that the branch
    # CAN fail, not that it accepts a correct port: two identical rows agree on
    # their top-1 by construction and self_comparison=True must not raise.
    _assert_logit_parity(base, base, _G(), "native",
                         name="self", self_comparison=True)
