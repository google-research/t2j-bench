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

"""Standalone Whisper log-mel feature extractor - no transformers dependency."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Union

import numpy as np

from .utils import BatchFeature

logger = logging.getLogger(__name__)




def hertz_to_mel(freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Slaney-scale Hz -> mel: linear below 1 kHz, log above."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def mel_to_hertz(mels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Inverse of :func:`hertz_to_mel`."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    """Slaney-normalized, Slaney-scale triangular mel filterbank, shape ``(bins, mels)``."""
    mel_min = hertz_to_mel(min_frequency)
    mel_max = hertz_to_mel(max_frequency)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters


def window_function(window_length: int, name: str = "hann") -> np.ndarray:
    """Periodic window of ``window_length`` points (``np.hanning(n + 1)[:-1]`` for hann)."""
    if name not in ("hann", "hann_window"):
        raise ValueError(f"Unknown window function '{name}'")
    return np.hanning(window_length + 1)[:-1]


def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    power: float,
    mel_filters: np.ndarray,
    mel_floor: float = 1e-10,
    min_value: float = 1e-10,
) -> np.ndarray:
    """Center-padded log10 mel spectrogram, shape ``(num_mels, num_frames)``."""
    half = int(frame_length // 2)
    waveform = np.pad(waveform, [(half, half)], mode="reflect")

    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(num_frames, frame_length),
        strides=(waveform.strides[0] * hop_length, waveform.strides[0]),
    ).astype(np.float64)
    frames *= window

    spec = np.fft.rfft(frames, n=frame_length, axis=-1).astype(np.complex64).T
    spec = np.abs(spec, dtype=np.float64) ** power
    spec = np.maximum(mel_floor, np.dot(mel_filters.T, spec))
    spec = np.log10(np.clip(spec, a_min=min_value, a_max=None))
    return np.asarray(spec, dtype=np.float32)




class WhisperFeatureExtractor:
    """Mirror of transformers' ``WhisperFeatureExtractor`` (numpy STFT path)."""

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        dither: float = 0.0,
        return_attention_mask: bool = False,
        **kwargs,
    ):
        if dither:
            raise ValueError(f"dither={dither} is not supported; this mirror is deterministic.")
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.padding_value = padding_value
        self.dither = dither
        self.return_attention_mask = return_attention_mask
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
        )
        self._window = window_function(self.n_fft, "hann")

    def _np_extract_fbank_features(self, waveform_batch: np.ndarray) -> np.ndarray:
        log_spec_batch = []
        for waveform in waveform_batch:
            log_spec = spectrogram(
                waveform,
                self._window,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                mel_filters=self.mel_filters,
            )
            log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            log_spec_batch.append(log_spec)
        return np.array(log_spec_batch)

    def __call__(
        self,
        raw_speech,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"This feature extractor expects {self.sampling_rate} Hz audio, got {sampling_rate} Hz. "
                "The clips under /data are resampled once at image build time, so a mismatch here is a "
                "bug in the bake."
            )
        if padding != "max_length":
            raise ValueError(f"Only padding='max_length' is supported, got {padding!r}.")

        is_batched = isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 1
        is_batched = is_batched or (
            isinstance(raw_speech, (list, tuple)) and raw_speech and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        )
        speeches = list(raw_speech) if is_batched else [raw_speech]
        speeches = [np.asarray(s, dtype=np.float32).reshape(-1) for s in speeches]

        target = int(max_length) if max_length else self.n_samples
        padded = np.full((len(speeches), target), self.padding_value, dtype=np.float32)
        attention_mask = np.zeros((len(speeches), target), dtype=np.int32)
        for i, s in enumerate(speeches):
            if s.size > target:
                if not truncation:
                    raise ValueError(
                        f"Audio is {s.size} samples but max_length is {target}; pass truncation=True to cut it."
                    )
                s = s[:target]
            padded[i, : s.size] = s
            attention_mask[i, : s.size] = 1

        data = {"input_features": self._np_extract_fbank_features(padded)}
        if return_attention_mask if return_attention_mask is not None else self.return_attention_mask:
            rescaled = attention_mask[:, :: self.hop_length]
            if attention_mask.shape[1] % self.hop_length != 0:
                rescaled = rescaled[:, :-1]
            data["attention_mask"] = rescaled

        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load from a ``preprocessor_config.json``, same resolution rules as the image processor: a directory, a file, or a repo id baked under ``/data``."""
        path = pretrained_model_name_or_path
        if os.path.isdir(path):
            config_file = os.path.join(path, "preprocessor_config.json")
        elif os.path.isfile(path):
            config_file = path
        else:
            config_file = os.path.join("/data", path, "preprocessor_config.json")
        with open(config_file, "r") as f:
            config = json.load(f)

        allowed = {
            "feature_size",
            "sampling_rate",
            "hop_length",
            "chunk_length",
            "n_fft",
            "padding_value",
            "dither",
            "return_attention_mask",
        }
        config = {k: v for k, v in config.items() if k in allowed}
        config.update(kwargs)
        return cls(**config)


__all__ = ["WhisperFeatureExtractor", "mel_filter_bank", "spectrogram", "window_function"]
