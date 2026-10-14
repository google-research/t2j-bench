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

"""Weight-quantization formats the checkpoint ships in.

One entry: MXFP4, which `openai/gpt-oss-20b` uses for its 19.1B expert
parameters and nothing else (`quantization_config.modules_to_not_convert`
covers attention, the routers, the embedding and the head).
"""

from quant.mxfp4 import dequantize_mxfp4

__all__ = ["dequantize_mxfp4"]
