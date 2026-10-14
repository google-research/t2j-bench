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

"""Local, self-contained ``datasets`` shim (mirror of torch_orig/datasets/dataset.py).

JAX adjustment: torch_orig omits this ``__init__`` and relies on the ambient HF
``datasets`` for ``from datasets import load_dataset``; the JAX repo is self-contained
and torch-free, so the shim's ``Dataset``/``load_dataset``/``concatenate_datasets`` are
re-exported here (the repo dir is first on ``sys.path`` at import time).
"""

from .dataset import Dataset, concatenate_datasets, load_dataset  # noqa: F401

__all__ = ["Dataset", "load_dataset", "concatenate_datasets"]
