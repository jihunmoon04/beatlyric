"""Step 3: Pre-compute per-frame audio spectrum bars into spectrum.json."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from common import write_json


def extract_spectrum(
    wav_path: Path,
    out_path: Path,
    fps: int = 30,
    num_bars: int = 28,
) -> dict[str, Any]:
    """Compute log-spaced FFT magnitude bands per video frame using stdlib wave + numpy."""
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    duration = len(samples) / sample_rate
    total_video_frames = max(1, int(np.ceil(duration * fps)))
    fft_size = 2048
    window = np.hanning(fft_size).astype(np.float32)

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    # Musical vocal/bass range: 55 Hz -> 8000 Hz on log scale
    band_edges = np.geomspace(55.0, min(8000.0, sample_rate * 0.45), num_bars + 1)

    band_masks = []
    for b in range(num_bars):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(freqs - band_edges[b])))
            mask[nearest] = True
        band_masks.append(mask)

    raw_matrix = np.zeros((total_video_frames, num_bars), dtype=np.float32)
    half_fft = fft_size // 2

    for frame_idx in range(total_video_frames):
        center = int((frame_idx / fps) * sample_rate)
        start = center - half_fft
        end = start + fft_size

        if start < 0 or end > len(samples):
            chunk = np.zeros(fft_size, dtype=np.float32)
            src_s = max(0, start)
            src_e = min(len(samples), end)
            dst_s = src_s - start
            chunk[dst_s : dst_s + (src_e - src_s)] = samples[src_s:src_e]
        else:
            chunk = samples[start:end]

        mag = np.abs(np.fft.rfft(chunk * window))
        for b, mask in enumerate(band_masks):
            raw_matrix[frame_idx, b] = float(np.mean(mag[mask]))

    # Log-compress and normalize by 98th percentile for steady visual response
    compressed = np.log1p(raw_matrix * 4.5)
    peak = float(np.percentile(compressed, 98)) or 1.0
    normalized = np.clip(compressed / peak, 0.0, 1.0)

    # Exponential smoothing across frames to avoid jitter
    smoothed = np.zeros_like(normalized)
    smoothed[0] = normalized[0]
    for i in range(1, total_video_frames):
        # Fast attack, smooth decay
        alpha = np.where(normalized[i] > smoothed[i - 1], 0.62, 0.30)
        smoothed[i] = alpha * normalized[i] + (1.0 - alpha) * smoothed[i - 1]

    # Quantize to 2 decimals to keep JSON compact
    frames_list = [
        [round(float(val), 2) for val in row]
        for row in smoothed
    ]

    payload = {
        "fps": fps,
        "num_bars": num_bars,
        "total_frames": total_video_frames,
        "frames": frames_list,
    }
    write_json(out_path, payload)
    return payload
