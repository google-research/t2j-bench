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

"""Minimal dataset helpers used by generated SFT repos."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable, Optional, Union


class Dataset:
    def __init__(self, data: list[dict[str, Any]]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._data[idx]
        if isinstance(idx, slice):
            return Dataset(self._data[idx])
        raise TypeError(f"Invalid index type: {type(idx)}")

    def __iter__(self):
        return iter(self._data)

    @property
    def column_names(self) -> list[str]:
        if not self._data:
            return []
        return list(self._data[0].keys())

    def map(
        self,
        function: Callable,
        batched: bool = False,
        batch_size: int = 1000,
        remove_columns: Optional[list[str]] = None,
        **_: Any,
    ) -> "Dataset":
        if batched:
            new_data = []
            for i in range(0, len(self._data), batch_size):
                batch = self._data[i : i + batch_size]
                batch_dict = {key: [item[key] for item in batch] for key in batch[0]} if batch else {}
                result = function(batch_dict)
                if result:
                    size = len(next(iter(result.values())))
                    for j in range(size):
                        new_data.append({k: v[j] for k, v in result.items()})
            return Dataset(new_data)

        mapped = []
        for item in self._data:
            result = function(item)
            if result:
                mapped.append(result)
        return Dataset(mapped)

    def select(self, indices: list[int] | range) -> "Dataset":
        return Dataset([self._data[i] for i in indices])

    def train_test_split(self, test_size: float = 0.1, seed: int = 42) -> dict[str, "Dataset"]:
        rng = random.Random(seed)
        indices = list(range(len(self._data)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_size))
        return {"train": self.select(indices[:split_idx]), "test": self.select(indices[split_idx:])}


def load_dataset(
    path: str,
    split: Optional[str] = None,
    data_files: Optional[Union[str, list[str]]] = None,
    **_: Any,
) -> Dataset:
    if data_files is None:
        data_files = path
    if isinstance(data_files, str):
        data_files = [data_files]

    all_data: list[dict[str, Any]] = []
    for file_path in data_files:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".json":
            with open(file_path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                all_data.extend(payload)
            else:
                all_data.append(payload)
        elif suffix == ".jsonl":
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_data.append(json.loads(line))
        elif suffix == ".csv":
            with open(file_path, "r") as f:
                all_data.extend(list(csv.DictReader(f)))
        elif suffix == ".txt":
            with open(file_path, "r") as f:
                all_data.extend({"text": line.rstrip("\n")} for line in f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    return Dataset(all_data)


def concatenate_datasets(datasets: list[Dataset]) -> Dataset:
    merged: list[dict[str, Any]] = []
    for dataset in datasets:
        merged.extend(dataset._data)
    return Dataset(merged)

