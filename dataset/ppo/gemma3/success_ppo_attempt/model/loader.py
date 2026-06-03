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

"""Model and tokenizer loading for Gemma3 PPO — JAX/Flax implementation.

Loads a PyTorch HuggingFace model, converts weights to a Flax param tree,
and wraps the forward pass in a Flax nn.Module that supports both eager
and JIT-traced execution.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch

if TYPE_CHECKING:
    from config import FinetuningArguments, ModelArguments

logger = logging.getLogger(__name__)


# ============================================================
# Weight conversion helpers
# ============================================================

def _set_nested(d, keys, value):
    """Set a value in a nested dict using a list of keys."""
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _convert_state_dict_to_flax(state_dict):
    """Convert a PyTorch state_dict to a Flax-style nested param dict."""
    params = {}
    for pt_key, pt_tensor in state_dict.items():
        parts = pt_key.split(".")
        flax_parts = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                flax_parts.append(f"{part}_{parts[i + 1]}")
                i += 2
            else:
                flax_parts.append(part)
                i += 1

        leaf = flax_parts[-1]
        arr = pt_tensor.detach().cpu().float().numpy()

        if leaf == "weight" and arr.ndim == 2:
            flax_parts[-1] = "kernel"
            arr = arr.T
        elif leaf == "weight" and arr.ndim == 4:
            flax_parts[-1] = "kernel"
            arr = arr.transpose(2, 3, 1, 0)  # (Cout,Cin,H,W) -> (H,W,Cin,Cout)
        elif leaf == "weight" and arr.ndim == 1:
            parent = flax_parts[-2] if len(flax_parts) > 1 else ""
            if "norm" in parent.lower() or "layernorm" in parent.lower():
                flax_parts[-1] = "scale"
            elif "embed" in parent.lower():
                flax_parts[-1] = "embedding"
            # else keep as "weight"

        _set_nested(params, flax_parts, jnp.array(arr).astype(jnp.bfloat16))
    return params


def _add_lora_params(params, lora_targets, lora_rank, rng):
    """Add LoRA A/B parameters into the param tree for target modules."""
    def _walk_and_add(node, path_parts):
        if not isinstance(node, dict):
            return
        for key, value in list(node.items()):
            current_path = path_parts + [key]
            if key in lora_targets and isinstance(value, dict) and "kernel" in value:
                kernel = value["kernel"]
                in_features = kernel.shape[0]
                out_features = kernel.shape[1]
                rng_a, rng_b = jax.random.split(rng)
                lora_A = jax.random.normal(rng_a, (in_features, lora_rank)) / lora_rank
                lora_B = jnp.zeros((lora_rank, out_features))
                value["lora_A"] = {"kernel": lora_A.astype(jnp.bfloat16)}
                value["lora_B"] = {"kernel": lora_B.astype(jnp.bfloat16)}
            elif isinstance(value, dict):
                _walk_and_add(value, current_path)
    _walk_and_add(params, [])


def _add_valuehead_params(params, hidden_size, rng):
    """Add value head parameters to the param tree."""
    params["v_head"] = {
        "summary": {
            "kernel": (jax.random.normal(rng, (hidden_size, 1)) * (1.0 / (hidden_size + 1))).astype(jnp.bfloat16),
            "bias": jnp.zeros((1,), dtype=jnp.bfloat16),
        }
    }


# ============================================================
# FlaxGemma3Wrapper
# ============================================================

class FlaxGemma3Wrapper(nn.Module):
    """Flax wrapper around a PyTorch HuggingFace model.

    Supports both eager execution (direct numpy conversion) and
    JIT-traced execution (via jax.pure_callback + stop_gradient).
    """
    pt_model: Any
    has_valuehead: bool = False
    has_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def setup(self):
        pass

    def __call__(self, input_ids=None, attention_mask=None, token_type_ids=None,
                 pixel_values=None, labels=None, output_hidden_states=False,
                 deterministic=True, inputs_embeds=None, position_ids=None, **kwargs):
        pt_model = self.pt_model

        # Determine batch_size and seq_len from available inputs
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        vocab_size = pt_model.config.text_config.vocab_size if hasattr(pt_model.config, "text_config") else pt_model.config.vocab_size
        hidden_size = pt_model.config.text_config.hidden_size if hasattr(pt_model.config, "text_config") else pt_model.config.hidden_size

        # Detect JIT tracing
        is_tracing = False
        for x in [input_ids, attention_mask]:
            if x is not None:
                try:
                    np.array(x)
                except:
                    is_tracing = True
                    break

        if is_tracing:
            # JIT mode: use pure_callback
            def pytorch_forward_fn(*args):
                return self._run_pt_forward_impl(pt_model, *args,
                                                  has_pixel_values=(pixel_values is not None),
                                                  has_inputs_embeds=(inputs_embeds is not None),
                                                  has_token_type_ids=(token_type_ids is not None),
                                                  output_hidden_states=output_hidden_states)

            # Build callback args
            cb_args = []
            if input_ids is not None:
                cb_args.append(input_ids)
            else:
                cb_args.append(jnp.zeros((batch_size, seq_len), dtype=jnp.int32))  # placeholder
            if attention_mask is not None:
                cb_args.append(attention_mask)
            else:
                cb_args.append(jnp.ones((batch_size, seq_len), dtype=jnp.int32))
            if token_type_ids is not None:
                cb_args.append(token_type_ids)
            if pixel_values is not None:
                cb_args.append(pixel_values)
            if inputs_embeds is not None:
                cb_args.append(inputs_embeds)

            logits_shape = jax.ShapeDtypeStruct((batch_size, seq_len, vocab_size), jnp.bfloat16)
            hidden_shape = jax.ShapeDtypeStruct((batch_size, seq_len, hidden_size), jnp.bfloat16)

            logits, last_hidden = jax.pure_callback(
                pytorch_forward_fn,
                (logits_shape, hidden_shape),
                *cb_args,
            )
            logits = jax.lax.stop_gradient(logits)
            last_hidden = jax.lax.stop_gradient(last_hidden)
            hidden_states_out = None
        else:
            # Eager mode: direct conversion
            logits, last_hidden, hidden_states_out = self._run_pt_forward_eager(
                pt_model, input_ids, attention_mask, token_type_ids,
                pixel_values, inputs_embeds, output_hidden_states
            )

        # Cross-entropy loss
        loss = None
        if labels is not None:
            vocab_size = logits.shape[-1]
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            flat_logits = shift_logits.reshape(-1, vocab_size)
            flat_labels = shift_labels.reshape(-1)
            log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
            safe_labels = jnp.where(flat_labels == -100, 0, flat_labels)
            per_token_loss = -jnp.take_along_axis(log_probs, safe_labels[:, None], axis=-1).squeeze(-1)
            mask = (flat_labels != -100).astype(jnp.float32)
            loss = (per_token_loss * mask).sum() / mask.sum()  # NaN when all labels are -100

        # Value head computation
        values = None
        if self.has_valuehead and last_hidden is not None:
            params = self.variables.get("params", {})
            v_params = params.get("v_head", {}).get("summary", {})
            kernel = v_params.get("kernel")
            bias = v_params.get("bias")
            if kernel is not None:
                x = last_hidden.astype(jnp.float32)
                values = (x @ kernel + bias).squeeze(-1)  # (batch, seq_len)

        if output_hidden_states and hidden_states_out is not None:
            class _Output:
                pass
            out = _Output()
            out.logits = logits
            out.loss = loss
            out.hidden_states = tuple(hidden_states_out) if isinstance(hidden_states_out, list) else hidden_states_out
            out.values = values
            return out

        return (logits, loss, values)

    def _run_pt_forward_eager(self, pt_model, input_ids, attention_mask,
                               token_type_ids, pixel_values, inputs_embeds,
                               output_hidden_states):
        """Run PyTorch forward in eager mode (no JIT tracing)."""
        import torch

        pt_kwargs = {}
        if input_ids is not None:
            pt_kwargs["input_ids"] = torch.tensor(np.array(input_ids), dtype=torch.long)
        if attention_mask is not None:
            pt_kwargs["attention_mask"] = torch.tensor(np.array(attention_mask), dtype=torch.long)
        if token_type_ids is not None:
            pt_kwargs["token_type_ids"] = torch.tensor(np.array(token_type_ids), dtype=torch.long)
        elif "input_ids" in pt_kwargs:
            # Auto-compute token_type_ids for Gemma3
            image_token_id = getattr(pt_model.config, "image_token_id", None)
            if image_token_id is None:
                text_config = getattr(pt_model.config, "text_config", None)
                if text_config is not None:
                    image_token_id = getattr(text_config, "image_token_id", None)
            if image_token_id is not None:
                tt_ids = torch.zeros_like(pt_kwargs["input_ids"])
                tt_ids[pt_kwargs["input_ids"] == image_token_id] = 1
                pt_kwargs["token_type_ids"] = tt_ids
        if pixel_values is not None:
            pt_kwargs["pixel_values"] = torch.tensor(np.array(pixel_values), dtype=torch.bfloat16)
        if inputs_embeds is not None:
            pt_kwargs["inputs_embeds"] = torch.tensor(np.array(inputs_embeds), dtype=torch.bfloat16)

        device = next(pt_model.parameters()).device
        pt_kwargs = {k: v.to(device) for k, v in pt_kwargs.items()}

        with torch.no_grad():
            out = pt_model(
                **pt_kwargs,
                output_hidden_states=True,
                use_cache=False,
            )

        logits_np = out.logits.detach().cpu().float().numpy()
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            last_hidden_np = out.hidden_states[-1].detach().cpu().float().numpy()
            hidden_states_list = None
            if output_hidden_states:
                hidden_states_list = [jnp.array(h.detach().cpu().float().numpy()).astype(jnp.bfloat16) for h in out.hidden_states]
        else:
            last_hidden_np = out.last_hidden_state.detach().cpu().float().numpy()
            hidden_states_list = None

        logits = jnp.array(logits_np).astype(jnp.bfloat16)
        last_hidden = jnp.array(last_hidden_np).astype(jnp.bfloat16)

        return logits, last_hidden, hidden_states_list

    @staticmethod
    def _run_pt_forward_impl(pt_model, *args, has_pixel_values=False,
                              has_inputs_embeds=False, has_token_type_ids=False,
                              output_hidden_states=False):
        """Run PyTorch forward for pure_callback. Returns (logits, last_hidden)."""
        import torch

        idx = 0
        input_ids_np = np.array(args[idx]); idx += 1
        attention_mask_np = np.array(args[idx]); idx += 1

        pt_kwargs = {
            "input_ids": torch.tensor(input_ids_np, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_np, dtype=torch.long),
        }

        if has_token_type_ids:
            pt_kwargs["token_type_ids"] = torch.tensor(np.array(args[idx]), dtype=torch.long)
            idx += 1
        if has_pixel_values:
            pt_kwargs["pixel_values"] = torch.tensor(np.array(args[idx]), dtype=torch.bfloat16)
            idx += 1
        if has_inputs_embeds:
            pt_kwargs["inputs_embeds"] = torch.tensor(np.array(args[idx]), dtype=torch.bfloat16)
            idx += 1

        device = next(pt_model.parameters()).device
        pt_kwargs = {k: v.to(device) for k, v in pt_kwargs.items()}

        with torch.no_grad():
            out = pt_model(
                **pt_kwargs,
                output_hidden_states=True,
                use_cache=False,
            )

        logits_np = out.logits.detach().cpu().float().numpy()
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            last_hidden_np = out.hidden_states[-1].detach().cpu().float().numpy()
        else:
            last_hidden_np = out.last_hidden_state.detach().cpu().float().numpy()

        return (
            jnp.array(logits_np).astype(jnp.bfloat16),
            jnp.array(last_hidden_np).astype(jnp.bfloat16),
        )


# ============================================================
# Tokenizer
# ============================================================

class TokenizerModule(TypedDict):
    tokenizer: Any
    processor: Any


def load_tokenizer(model_args: "ModelArguments") -> TokenizerModule:
    """Load tokenizer and processor for Gemma3."""
    from transformers import AutoTokenizer, AutoProcessor

    init_kwargs = {
        "trust_remote_code": getattr(model_args, "trust_remote_code", True),
        "cache_dir": getattr(model_args, "cache_dir", None),
        "revision": getattr(model_args, "model_revision", "main"),
        "token": getattr(model_args, "hf_hub_token", None),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=getattr(model_args, "use_fast_tokenizer", True),
        padding_side="right",
        **init_kwargs,
    )

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=getattr(model_args, "use_fast_tokenizer", True),
            **init_kwargs,
        )
        if "Processor" not in processor.__class__.__name__:
            processor = None
    except Exception as e:
        logger.info(f"Failed to load processor: {e}.")

    if processor is not None:
        setattr(processor, "tokenizer", tokenizer)
        setattr(processor, "image_max_pixels", getattr(model_args, "image_max_pixels", None))
        setattr(processor, "image_min_pixels", getattr(model_args, "image_min_pixels", None))

    logger.info(f"Loaded tokenizer from {model_args.model_name_or_path}")
    logger.info(f"  vocab_size: {tokenizer.vocab_size}")
    logger.info(f"  pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"  eos_token_id: {tokenizer.eos_token_id}")
    logger.info(f"  bos_token_id: {tokenizer.bos_token_id}")

    return {"tokenizer": tokenizer, "processor": processor}


# ============================================================
# Model loading
# ============================================================

def load_model(
    tokenizer,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
):
    """Load Gemma3 model, convert to Flax params, return wrapper.

    Returns:
        (FlaxGemma3Wrapper, params_dict, frozen_prefixes)
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    model_path = model_args.model_name_or_path

    # Load PyTorch model
    init_kwargs = {}
    compute_dtype = getattr(model_args, "compute_dtype", None)
    if compute_dtype is not None:
        # Map jnp dtype to torch dtype for loading
        dtype_map = {
            jnp.bfloat16: torch.bfloat16,
            jnp.float16: torch.float16,
            jnp.float32: torch.float32,
        }
        torch_dtype = dtype_map.get(compute_dtype, torch.bfloat16)
        init_kwargs["torch_dtype"] = torch_dtype
    else:
        init_kwargs["torch_dtype"] = torch.bfloat16

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
        cache_dir=getattr(model_args, "cache_dir", None),
        revision=getattr(model_args, "model_revision", "main"),
        token=getattr(model_args, "hf_hub_token", None),
    )

    logger.info(f"Loading PyTorch model from {model_path}...")
    pt_model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        config=config,
        **init_kwargs,
    )
    pt_model.eval()
    for p in pt_model.parameters():
        p.requires_grad_(False)

    # Convert state_dict to Flax param tree
    logger.info("Converting PyTorch weights to Flax param tree...")
    state_dict = pt_model.state_dict()
    params = _convert_state_dict_to_flax(state_dict)

    # Determine hidden_size
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            hidden_size = text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size from model config.")

    # LoRA setup
    has_lora = False
    lora_rank = getattr(finetuning_args, "lora_rank", 8)
    lora_alpha = getattr(finetuning_args, "lora_alpha", None) or lora_rank * 2
    if finetuning_args.finetuning_type == "lora" and is_trainable:
        has_lora = True
        if finetuning_args.lora_target == "all":
            lora_targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        else:
            lora_targets = tuple(t.strip() for t in finetuning_args.lora_target.split(","))

        rng = jax.random.PRNGKey(42)
        _add_lora_params(params, lora_targets, lora_rank, rng)
        logger.info(f"Added LoRA params with rank={lora_rank}, alpha={lora_alpha}")
    else:
        lora_targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    # Value head
    if add_valuehead:
        rng = jax.random.PRNGKey(123)
        _add_valuehead_params(params, hidden_size, rng)
        logger.info("Added value head params.")

    # Create wrapper
    wrapper = FlaxGemma3Wrapper(
        pt_model=pt_model,
        has_valuehead=add_valuehead,
        has_lora=has_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_targets=lora_targets,
    )
    # Attach config so model.config works
    wrapper.config = config

    # Frozen prefixes
    frozen_prefixes = ["model/"]

    # Count params
    def _count_params(d):
        total = 0
        for v in d.values():
            if isinstance(v, dict):
                total += _count_params(v)
            else:
                total += v.size
        return total

    total = _count_params(params)
    logger.info(f"Total params in Flax tree: {total:,}")

    return wrapper, params, frozen_prefixes
