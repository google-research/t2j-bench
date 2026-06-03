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

"""Simple Dataset class replacing HuggingFace datasets for SFT training."""

import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Dataset:
    """A simple in-memory dataset backed by a list of dicts."""

    def __init__(self, data: list[dict[str, Any]]):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Dataset(self._data[idx])
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    def select(self, indices) -> "Dataset":
        """Select a subset of the dataset by indices."""
        indices = list(indices)
        return Dataset([self._data[i] for i in indices])

    def map(
        self,
        function: Callable,
        batched: bool = False,
        batch_size: int = 1000,
        remove_columns: Optional[list[str]] = None,
        num_proc: Optional[int] = None,
        load_from_cache_file: bool = False,
        desc: Optional[str] = None,
    ) -> "Dataset":
        """Apply a function to each example (or batch of examples)."""
        if desc:
            logger.info(desc)

        if batched:
            new_data = self._map_batched(function, batch_size)
        else:
            new_data = self._map_single(function, num_proc)

        return Dataset(new_data)

    def _map_single(self, function: Callable, num_proc: Optional[int] = None) -> list[dict]:
        """Map function over individual examples."""
        results = []
        for example in self._data:
            result = function(example)
            if result is not None:
                results.append(result)
        return results

    def _map_batched(self, function: Callable, batch_size: int) -> list[dict]:
        """Map function over batches. Input/output are column-oriented dicts of lists."""
        all_results: dict[str, list] = {}
        for start in range(0, len(self._data), batch_size):
            batch_rows = self._data[start : start + batch_size]
            # Convert row-oriented to column-oriented
            if not batch_rows:
                continue
            batch_cols = {key: [row.get(key) for row in batch_rows] for key in batch_rows[0]}
            result_cols = function(batch_cols)
            if result_cols is None:
                continue
            for key, values in result_cols.items():
                if key not in all_results:
                    all_results[key] = []
                all_results[key].extend(values)

        if not all_results:
            return []
        # Convert column-oriented back to row-oriented
        keys = list(all_results.keys())
        n = len(all_results[keys[0]])
        return [{key: all_results[key][i] for key in keys} for i in range(n)]

    def train_test_split(self, test_size: float = 0.1, seed: int = 42) -> dict[str, "Dataset"]:
        """Split dataset into train and test sets."""
        rng = random.Random(seed)
        indices = list(range(len(self._data)))
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        return {
            "train": self.select(train_indices),
            "test": self.select(test_indices),
        }


def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[list[str]] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Load a dataset from local files (JSON/JSONL/CSV/text)."""
    if data_files is not None:
        files = data_files
    elif os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
    else:
        raise ValueError(f"Cannot load dataset from '{path}'. Use HF datasets for hub datasets.")

    all_data = []
    for filepath in files:
        ext = os.path.splitext(filepath)[-1].lower()
        if ext in (".json", ".jsonl") or path in ("json",):
            all_data.extend(_load_json_file(filepath))
        elif ext == ".csv" or path == "csv":
            all_data.extend(_load_csv_file(filepath))
        elif ext == ".txt" or path == "text":
            all_data.extend(_load_text_file(filepath))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    logger.info(f"Loaded {len(all_data)} examples from {len(files)} file(s).")
    return Dataset(all_data)


def _load_json_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def _load_csv_file(filepath: str) -> list[dict]:
    import csv
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_text_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [{"text": line.strip()} for line in f if line.strip()]


def concatenate_datasets(datasets: list[Dataset]) -> Dataset:
    """Concatenate multiple datasets."""
    all_data = []
    for ds in datasets:
        all_data.extend(ds._data)
    return Dataset(all_data)

