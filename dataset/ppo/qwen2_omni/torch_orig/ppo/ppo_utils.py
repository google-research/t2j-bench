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

Standalone implementations with no TRL dependency.
"""

import json
import math
from typing import Optional

import torch
import torch.nn.functional as F


def _align_time_dim(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    valid = [tensor for tensor in tensors if tensor is not None]
    if not valid:
        return tensors
    target_len = min(tensor.shape[1] for tensor in valid)
    return tuple(tensor[:, :target_len] if tensor is not None else None for tensor in tensors)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probabilities from logits.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
    Returns:
        (batch, seq_len) log probabilities
    """
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None
) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis).clamp(min=1)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def masked_var(
    values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered = values - mean
    variance = masked_mean(centered**2, mask)
    if unbiased:
        n = mask.sum()
        variance = variance * n / (n - 1).clamp(min=1)
    return variance


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten values with masked values."""
    mean = masked_mean(values, mask)
    var = masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    lastgaelam = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    advantages_reversed = []
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else torch.zeros_like(values[:, 0])
        # Only consider non-masked positions
        delta = rewards[:, t] + gamma * nextvalues * mask[:, min(t + 1, gen_length - 1)] - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * mask[:, t]
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages, returns


def compute_ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    cliprange: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute clipped PPO policy loss.

    Returns:
        pg_loss: scalar policy gradient loss
        clipfrac: fraction of samples clipped
    """
    new_logprobs, old_logprobs, advantages, mask = _align_time_dim(
        new_logprobs, old_logprobs, advantages, mask
    )
    ratio = torch.exp(new_logprobs - old_logprobs)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mask)
    clipfrac = masked_mean((pg_losses2 > pg_losses1).float(), mask)
    return pg_loss, clipfrac


def compute_value_loss(
    vpred: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    cliprange_value: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute clipped value function loss.

    Returns:
        vf_loss: scalar value loss
        vf_clipfrac: fraction of samples clipped
    """
    vpred, old_values, returns, mask = _align_time_dim(vpred, old_values, returns, mask)
    vpredclipped = torch.clamp(
        vpred, old_values - cliprange_value, old_values + cliprange_value
    )
    vf_losses1 = (vpred - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
    vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), mask)
    return vf_loss, vf_clipfrac


def get_rewards_from_server(server_url: str, messages: list[str]) -> list[torch.Tensor]:
    """Get reward scores from an API server.

    The server should accept POST requests with {"model": ..., "messages": ...}
    and return {"scores": [float, ...]}.
    """
    import requests

    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.tensor(rewards, dtype=torch.float32)


def torch_gc() -> None:
    """Run garbage collection and clear CUDA cache."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
