# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Family metadata for generated SFT torch_orig repos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySpec:
    slug: str
    default_template: str
    multimodal: bool = False
    hf_module: str | None = None
    source_repo: str | None = None
    source_files: tuple[str, ...] = ()
    config_class: str | None = None
    default_architecture: str | None = None
    notes: str = ""


SUPPORTED_FAMILY_SPECS: list[FamilySpec] = [
    FamilySpec("bloom", "default", hf_module="bloom", config_class="BloomConfig", default_architecture="BloomForCausalLM"),
    FamilySpec("deepseek", "deepseek", hf_module="deepseek_v3"),
    FamilySpec("deepseek_r1", "deepseekr1", hf_module="deepseek_v3"),
    FamilySpec("ernie4_5", "ernie_nothink", hf_module="ernie4_5"),
    FamilySpec("falcon", "falcon", hf_module="falcon", config_class="FalconConfig", default_architecture="FalconForCausalLM"),
    FamilySpec("gemma", "gemma", hf_module="gemma", config_class="GemmaConfig", default_architecture="GemmaForCausalLM"),
    FamilySpec("gemma3", "gemma3", multimodal=True),
    FamilySpec("glm4", "glm4", hf_module="glm4"),
    FamilySpec("glm4_5", "glm4_moe", hf_module="glm4v"),
    FamilySpec("gpt2", "default", hf_module="gpt2", config_class="GPT2Config", default_architecture="GPT2LMHeadModel"),
    FamilySpec("granite", "granite3", hf_module="granite"),
    FamilySpec("hunyuan", "hunyuan", hf_module="hunyuan_v1_dense"),
    FamilySpec(
        "internlm",
        "intern2",
        source_repo="internlm/internlm2_5-1_8b-chat",
        source_files=(
            "configuration_internlm2.py",
            "modeling_internlm2.py",
            "tokenization_internlm2.py",
            "tokenization_internlm2_fast.py",
        ),
        config_class="InternLM2Config",
        default_architecture="InternLM2ForCausalLM",
    ),
    FamilySpec(
        "internvl",
        "intern_vl",
        multimodal=True,
        hf_module="internvl",
        config_class="InternVLConfig",
        default_architecture="InternVLForConditionalGeneration",
    ),
    FamilySpec(
        "intern_s1",
        "intern_s1",
        multimodal=True,
        source_repo="internlm/Intern-S1-mini",
        source_files=(
            "configuration_interns1.py",
            "modeling_interns1.py",
            "processing_interns1.py",
            "tokenization_interns1.py",
            "video_processing_interns1.py",
        ),
        config_class="InternS1Config",
        default_architecture="InternS1ForConditionalGeneration",
    ),
    FamilySpec("lfm2", "lfm2_vl", multimodal=True, hf_module="lfm2_vl"),
    FamilySpec("llama", "default", hf_module="llama", config_class="LlamaConfig", default_architecture="LlamaForCausalLM"),
    FamilySpec("llama2", "llama2", hf_module="llama", config_class="LlamaConfig", default_architecture="LlamaForCausalLM"),
    FamilySpec("llama3", "llama3", hf_module="llama", config_class="LlamaConfig", default_architecture="LlamaForCausalLM"),
    FamilySpec("llava", "llava", multimodal=True, hf_module="llava", default_architecture="LlavaForConditionalGeneration"),
    FamilySpec("llava_next", "llava_next", multimodal=True),
    FamilySpec("llava_next_video", "llava_next_video", multimodal=True, hf_module="llava_next_video", default_architecture="LlavaNextVideoForConditionalGeneration"),
    FamilySpec(
        "mimo",
        "mimo",
        source_repo="XiaomiMiMo/MiMo-7B-RL",
        source_files=("configuration_mimo.py", "modeling_mimo.py"),
        config_class="MiMoConfig",
        default_architecture="MiMoForCausalLM",
    ),
    FamilySpec(
        "minicpm4",
        "cpm4",
        source_repo="openbmb/MiniCPM4-0.5B",
        source_files=("configuration_minicpm.py", "modeling_minicpm.py"),
        config_class="MiniCPMConfig",
        default_architecture="MiniCPMForCausalLM",
    ),
    FamilySpec(
        "minicpm_o",
        "minicpm_o",
        multimodal=True,
        source_repo="openbmb/MiniCPM-o-2_6",
        source_files=(
            "configuration_minicpm.py",
            "modeling_minicpmo.py",
            "modeling_navit_siglip.py",
            "processing_minicpmo.py",
            "image_processing_minicpmv.py",
            "tokenization_minicpmo_fast.py",
            "resampler.py",
            "utils.py",
        ),
        config_class="MiniCPMOConfig",
        default_architecture="MiniCPMO",
    ),
    FamilySpec("ministral3", "ministral3", hf_module="ministral", default_architecture="MinistralForCausalLM"),
    FamilySpec("mistral", "mistral", hf_module="mistral", config_class="MistralConfig", default_architecture="MistralForCausalLM"),
    FamilySpec("paligemma", "paligemma", multimodal=True, hf_module="paligemma", config_class="PaliGemmaConfig", default_architecture="PaliGemmaForConditionalGeneration"),
    FamilySpec("phi", "phi", hf_module="phi", config_class="PhiConfig", default_architecture="PhiForCausalLM"),
    FamilySpec("phi_small", "phi_small", hf_module="phi3", config_class="Phi3Config", default_architecture="Phi3ForCausalLM"),
    FamilySpec("phi4", "phi4_mini", hf_module="phi3"),
    FamilySpec("qwen2", "qwen", hf_module="qwen2", config_class="Qwen2Config", default_architecture="Qwen2ForCausalLM"),
    FamilySpec("qwen3", "qwen3", hf_module="qwen3", config_class="Qwen3Config", default_architecture="Qwen3ForCausalLM"),
    FamilySpec("qwen2_audio", "qwen2_audio", multimodal=True, hf_module="qwen2_audio"),
    FamilySpec(
        "qwen2_omni",
        "qwen2_omni",
        multimodal=True,
        hf_module="qwen2_5_omni",
        config_class="Qwen2_5OmniConfig",
        default_architecture="Qwen2_5OmniForConditionalGeneration",
    ),
    FamilySpec(
        "qwen2_vl",
        "qwen2_vl",
        multimodal=True,
        hf_module="qwen2_vl",
        config_class="Qwen2VLConfig",
        default_architecture="Qwen2VLForConditionalGeneration",
    ),
    FamilySpec("qwen3_vl", "qwen3_vl", multimodal=True),
    FamilySpec("seed", "seed_oss", hf_module="seed_oss"),
    FamilySpec("starcoder2", "default", hf_module="starcoder2", default_architecture="Starcoder2ForCausalLM"),
    FamilySpec(
        "telechat2",
        "telechat2",
        source_repo="Tele-AI/TeleChat2-3B",
        source_files=("configuration_telechat2.py", "modeling_telechat2.py", "tokenization_telechat2.py", "generation_utils.py"),
        config_class="Telechat2Config",
        default_architecture="Telechat2ForCausalLM",
    ),
    FamilySpec(
        "yuan2",
        "yuan",
        source_repo="IEITYuan/Yuan2-2B-hf",
        source_files=("configuration_yuan.py", "yuan_hf_model.py"),
        config_class="YuanConfig",
        default_architecture="YuanForCausalLM",
    ),
]


SUPPORTED_FAMILY_BY_SLUG = {spec.slug: spec for spec in SUPPORTED_FAMILY_SPECS}
