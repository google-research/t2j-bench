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

"""Shared constants (mirror of torch_orig/data/constants.py)."""

import os

IGNORE_INDEX = -100
DATA_CONFIG = "dataset_info.json"
IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<image>")
VIDEO_PLACEHOLDER = os.environ.get("VIDEO_PLACEHOLDER", "<video>")
AUDIO_PLACEHOLDER = os.environ.get("AUDIO_PLACEHOLDER", "<audio>")
FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}
