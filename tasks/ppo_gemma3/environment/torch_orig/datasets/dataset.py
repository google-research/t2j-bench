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

"""Simple dataset implementation replacing HuggingFace datasets."""
import csv
import json
import random
from pathlib import Path
from typing import Any, Callable, Optional, Union


class Dataset:
    """Minimal Dataset class backed by a list of dicts."""

    def __init__(self, data: list[dict[str, Any]]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._data[idx]
        elif isinstance(idx, slice):
            return Dataset(self._data[idx])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __iter__(self):
        return iter(self._data)

    @property
    def column_names(self) -> list[str]:
        if len(self._data) == 0:
            return []
        return list(self._data[0].keys())

    def map(
        self,
        function: Callable,
        batched: bool = False,
        batch_size: int = 1000,
        num_proc: Optional[int] = None,
        remove_columns: Optional[list[str]] = None,
        desc: Optional[str] = None,
        **kwargs,
    ) -> "Dataset":
        if batched:
            new_data = []
            for i in range(0, len(self._data), batch_size):
                batch = self._data[i : i + batch_size]
                batch_dict = {}
                if batch:
                    for key in batch[0]:
                        batch_dict[key] = [item[key] for item in batch]
                result = function(batch_dict)
                if result:
                    n = len(next(iter(result.values())))
                    for j in range(n):
                        new_data.append({k: v[j] for k, v in result.items()})
            return Dataset(new_data)
        else:
            new_data = []
            for item in self._data:
                result = function(item)
                if result:
                    new_data.append(result)
            return Dataset(new_data)

    def select(self, indices: list[int]) -> "Dataset":
        return Dataset([self._data[i] for i in indices])

    def train_test_split(self, test_size: float = 0.1, seed: int = 42) -> dict:
        rng = random.Random(seed)
        indices = list(range(len(self._data)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_size))
        return {
            "train": self.select(indices[:split_idx]),
            "test": self.select(indices[split_idx:]),
        }


def load_dataset(
    path: str,
    split: Optional[str] = None,
    data_files: Optional[Union[str, list[str]]] = None,
    **kwargs,
) -> Dataset:
    """Load a dataset from file(s)."""
    if data_files is None:
        data_files = path

    if isinstance(data_files, str):
        data_files = [data_files]

    all_data = []
    for file_path in data_files:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in (".json",):
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        elif suffix in (".jsonl",):
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_data.append(json.loads(line))
        elif suffix in (".csv",):
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                all_data.extend(list(reader))
        elif suffix in (".txt",):
            with open(file_path, "r") as f:
                for line in f:
                    all_data.append({"text": line.strip()})
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    return Dataset(all_data)


def concatenate_datasets(datasets: list[Dataset]) -> Dataset:
    """Concatenate multiple datasets."""
    all_data = []
    for ds in datasets:
        all_data.extend(ds._data)
    return Dataset(all_data)
