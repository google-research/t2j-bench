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

"""PPO algorithm utilities: GAE, losses, rewards, log-probs.

JAX/Flax implementation — standalone, no TRL dependency.
"""

import json
from typing import Optional

import jax
import jax.numpy as jnp


def _align_time_dim(*tensors):
    valid = [t for t in tensors if t is not None]
    if not valid:
        return tensors
    target_len = min(t.shape[1] for t in valid)
    return tuple(t[:, :target_len] if t is not None else None for t in tensors)


def logprobs_from_logits(logits, labels):
    """Compute per-token log probabilities from logits.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
    Returns:
        (batch, seq_len) log probabilities
    """
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, labels[..., None], axis=-1).squeeze(-1)


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / jnp.clip(mask.sum(axis=axis), a_min=1)
    return (values * mask).sum() / jnp.clip(mask.sum(), a_min=1)


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Compute Generalized Advantage Estimation.

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
    lastgaelam = jnp.zeros(rewards.shape[0])
    advantages_reversed = []
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else jnp.zeros_like(values[:, 0])
        delta = rewards[:, t] + gamma * nextvalues * mask[:, min(t + 1, gen_length - 1)] - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * mask[:, t]
        advantages_reversed.append(lastgaelam)
    advantages = jnp.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def compute_ppo_loss(new_logprobs, old_logprobs, advantages, mask, cliprange=0.2):
    """Compute clipped PPO policy loss.

    Returns:
        pg_loss: scalar policy gradient loss
        clipfrac: fraction of samples clipped
    """
    new_logprobs, old_logprobs, advantages, mask = _align_time_dim(
        new_logprobs, old_logprobs, advantages, mask
    )
    ratio = jnp.exp(new_logprobs - old_logprobs)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(jnp.maximum(pg_losses1, pg_losses2), mask)
    clipfrac = masked_mean((pg_losses2 > pg_losses1).astype(jnp.float32), mask)
    return pg_loss, clipfrac


def compute_value_loss(vpred, old_values, returns, mask, cliprange_value=0.2):
    """Compute clipped value function loss.

    Returns:
        vf_loss: scalar value loss
        vf_clipfrac: fraction of samples clipped
    """
    vpred, old_values, returns, mask = _align_time_dim(vpred, old_values, returns, mask)
    vpredclipped = jnp.clip(vpred, old_values - cliprange_value, old_values + cliprange_value)
    vf_losses1 = (vpred - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(jnp.maximum(vf_losses1, vf_losses2), mask)
    vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).astype(jnp.float32), mask)
    return vf_loss, vf_clipfrac
