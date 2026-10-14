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

"""PPO algorithm utilities: GAE, losses, rewards, log-probs (torch-free JAX port).

Faithful mirror of ``torch_orig/ppo/ppo_utils.py`` with ``torch`` replaced by
``jax.numpy``: same GAE recursion, clipped policy/value losses, masked-whiten, and
per-token log-probs. No TRL dependency.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import jax
import jax.numpy as jnp


def _align_time_dim(*tensors):
    valid = [t for t in tensors if t is not None]
    if not valid:
        return tensors
    target_len = min(t.shape[1] for t in valid)
    return tuple(t[:, :target_len] if t is not None else None for t in tensors)


def logprobs_from_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Per-token log-probabilities from logits.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
    Returns:
        (batch, seq_len) log probabilities
    """
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]


def masked_mean(values: jnp.ndarray, mask: jnp.ndarray, axis: Optional[int] = None) -> jnp.ndarray:
    """Mean of ``values`` over the masked positions."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / jnp.maximum(mask.sum(axis=axis), 1)
    return (values * mask).sum() / jnp.maximum(mask.sum(), 1)


def masked_var(values: jnp.ndarray, mask: jnp.ndarray, unbiased: bool = True) -> jnp.ndarray:
    """Variance of ``values`` over the masked positions (Bessel-corrected by default)."""
    mean = masked_mean(values, mask)
    centered = values - mean
    variance = masked_mean(centered**2, mask)
    if unbiased:
        n = mask.sum()
        variance = variance * n / jnp.maximum(n - 1, 1)
    return variance


def masked_whiten(values: jnp.ndarray, mask: jnp.ndarray, shift_mean: bool = True) -> jnp.ndarray:
    """Whiten ``values`` over the masked positions."""
    mean = masked_mean(values, mask)
    var = masked_var(values, mask)
    whitened = (values - mean) * jax.lax.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    mask: jnp.ndarray,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized Advantage Estimation (same recursion as torch_orig).

    Args:
        rewards: (batch, response_len) per-token rewards
        values: (batch, response_len) value predictions
        mask: (batch, response_len) response mask
        gamma: discount factor
        lam: GAE lambda
    Returns:
        advantages: (batch, response_len)
        returns: (batch, response_len) = advantages + values
    """
    gen_length = rewards.shape[1]
    lastgaelam = jnp.zeros((rewards.shape[0],), dtype=rewards.dtype)
    advantages_reversed = []
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else jnp.zeros_like(values[:, 0])
        delta = rewards[:, t] + gamma * nextvalues * mask[:, min(t + 1, gen_length - 1)] - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * mask[:, t]
        advantages_reversed.append(lastgaelam)
    advantages = jnp.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def compute_ppo_loss(
    new_logprobs: jnp.ndarray,
    old_logprobs: jnp.ndarray,
    advantages: jnp.ndarray,
    mask: jnp.ndarray,
    cliprange: float = 0.2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Clipped PPO policy loss. Returns ``(pg_loss, clipfrac)``."""
    new_logprobs, old_logprobs, advantages, mask = _align_time_dim(
        new_logprobs, old_logprobs, advantages, mask
    )
    ratio = jnp.exp(new_logprobs - old_logprobs)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(jnp.maximum(pg_losses1, pg_losses2), mask)
    clipfrac = masked_mean((pg_losses2 > pg_losses1).astype(jnp.float32), mask)
    return pg_loss, clipfrac


def compute_value_loss(
    vpred: jnp.ndarray,
    old_values: jnp.ndarray,
    returns: jnp.ndarray,
    mask: jnp.ndarray,
    cliprange_value: float = 0.2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Clipped value-function loss. Returns ``(vf_loss, vf_clipfrac)``."""
    vpred, old_values, returns, mask = _align_time_dim(vpred, old_values, returns, mask)
    vpredclipped = jnp.clip(vpred, old_values - cliprange_value, old_values + cliprange_value)
    vf_losses1 = (vpred - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(jnp.maximum(vf_losses1, vf_losses2), mask)
    vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).astype(jnp.float32), mask)
    return vf_loss, vf_clipfrac


def token_f1_reward(pred_ids: list[int], gold_ids: list[int]) -> float:
    """Token-level F1 overlap between a generated response and its gold answer.

    A decode-order-invariant multiset overlap in [0, 1]; an empty prediction or an
    empty gold sequence scores 0.0. Pure integer set-ops -- identical to torch_orig.
    """
    if not pred_ids or not gold_ids:
        return 0.0
    common = sum((Counter(pred_ids) & Counter(gold_ids)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_ids)
    recall = common / len(gold_ids)
    return 2.0 * precision * recall / (precision + recall)


def torch_gc() -> None:
    """No-op in JAX (kept for API parity with torch_orig)."""
    import gc

    gc.collect()
