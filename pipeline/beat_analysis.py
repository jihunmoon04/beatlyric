"""Beat and downbeat estimation from repeated patterns in the audio."""

from __future__ import annotations

import numpy as np


def _spectral_onsets(samples: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    hop = max(1, round(sr * 0.005))
    size = 2048
    window = np.hanning(size).astype(np.float32)
    frequencies = np.fft.rfftfreq(size, 1 / sr)
    bands = [(frequencies >= lo) & (frequencies < hi)
             for lo, hi in ((30, 160), (160, 1600), (1600, 9000))]
    padded = np.pad(samples, size // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    energy = np.empty((len(frames), len(bands)), dtype=np.float32)
    for start in range(0, len(frames), 512):
        block = np.abs(np.fft.rfft(frames[start:start + 512] * window))
        for band, bins in enumerate(bands):
            if bins.any():
                energy[start:start + len(block), band] = np.sqrt(np.mean(block[:, bins] ** 2, axis=1))
            else:
                energy[start:start + len(block), band] = 0
    # Normalize each band so a loud bass drum does not erase the other cues.
    scale = np.maximum(np.percentile(energy, 90, axis=0), 1e-5)
    compressed = np.log1p(10 * energy / scale)
    flux = np.maximum(0, compressed - np.vstack((compressed[:1], compressed[:-1])))
    peaks = np.maximum(np.percentile(flux, 98, axis=0), 0.02)
    return np.minimum(2.0, flux / peaks), hop / sr


def _pulse_grid(envelope: np.ndarray, times: np.ndarray, duration: float, period: float) -> np.ndarray:
    phases = np.arange(60) * (period / 60)
    positions = phases[:, None] + np.arange(0, duration, period)[None, :]
    values = np.interp(positions, times, envelope, right=0)
    return np.minimum(values, 0.65) + 0.1 * values


def _intro_phase(grid: np.ndarray, strongest: int, first_beat: int) -> int:
    """Trust a repeated opening pulse when later offbeats dominate the mix."""
    if grid.shape[1] < first_beat + 32:
        return strongest

    phase_count = grid.shape[0]
    # Compare the strongest pulse with its half-beat rival, allowing for small
    # offsets between different instruments' transients.
    opposite = (strongest + phase_count // 2 + np.arange(-4, 5)) % phase_count
    opening = grid[:, first_beat:first_beat + 32].reshape(phase_count, 4, 8).mean(axis=2)
    rival = int(opposite[np.argmax(opening[opposite].mean(axis=1))])
    original = opening[strongest]
    supported_blocks = np.count_nonzero(opening[rival] - original > np.maximum(0.015, 0.12 * original))
    # Require agreement across the opening and meaningful support later in the track.
    if supported_blocks >= 3 and grid[rival].mean() >= 0.4 * grid[strongest].mean():
        return rival
    return strongest


def _bar_evidence(measures: np.ndarray) -> np.ndarray:
    bass, mid, treble = np.mean(np.minimum(measures, 1), axis=0).T
    strong = 0.65 * bass + 0.35 * mid
    backbeats = (np.roll(treble, -1) + np.roll(treble, -3)) / 2
    return strong + 0.4 * backbeats - 0.25 * np.roll(strong, -2)


def estimate_rhythm(
    samples: np.ndarray, sr: int, bpm: float | None = None,
) -> tuple[float, float, int, float]:
    """Return BPM, beat phase in seconds, downbeat index, and its winning margin."""
    if len(samples) < sr or not np.any(samples):
        return bpm or 120.0, 0.0, 0, 0.0

    flux, hop_sec = _spectral_onsets(samples, sr)
    pulse = flux @ np.array([0.38, 0.34, 0.28])
    width = max(1, round(0.020 / hop_sec))
    weights = np.arange(1, width + 2, dtype=float)
    weights = np.r_[weights, weights[-2::-1]]
    envelope = np.convolve(pulse, weights / weights.sum(), mode="same")
    times = np.arange(len(pulse)) * hop_sec
    duration = len(samples) / sr

    if bpm is None:
        centered = pulse - pulse.mean()
        fft_size = 1 << (2 * len(pulse) - 1).bit_length()
        fft = np.fft.rfft(centered, n=fft_size)
        ac = np.fft.irfft(abs(fft) ** 2)
        lags = np.arange(round(60 / (180 * hop_sec)), round(60 / (85 * hop_sec)) + 1)
        peaks = lags[(ac[lags] >= ac[lags - 1]) & (ac[lags] >= ac[lags + 1])]
        if len(peaks) == 0:
            peaks = lags
        strongest = peaks[np.argsort(ac[peaks])[-4:]]
        estimates = 60 / (strongest * hop_sec)
        candidates = np.unique(np.round((estimates[:, None] + np.arange(-1.0, 1.01, 0.05)).ravel(), 2))
    else:
        candidates = np.array([bpm])

    best = (-1.0, float(candidates[0]), 0)
    for tempo in candidates:
        period = 60 / tempo
        scores = _pulse_grid(envelope, times, duration, period).mean(axis=1)
        index = int(np.argmax(scores))
        if scores[index] > best[0]:
            best = (float(scores[index]), float(tempo), index)
    _, bpm, strongest = best
    period = 60 / bpm
    grid = _pulse_grid(envelope, times, duration, period)
    active = np.flatnonzero(envelope >= 0.12)
    first_beat = int(times[active[0]] // period) if active.size else 0
    phase_index = _intro_phase(grid, strongest, first_beat)
    phase = phase_index * (period / grid.shape[0])

    # A centered analysis frame responds shortly before a transient reaches its center.
    corrected = phase + 0.27 * 2048 / sr
    cycle = int(corrected / period)
    corrected -= cycle * period
    beats = phase + np.arange(int((duration - phase) / period) + 1) * period
    if len(beats) < 12:
        return bpm, round(float(corrected), 6), cycle % 4, 0.0
    indices = np.rint((beats[:, None] + np.linspace(-0.035, 0.035, 15)) / hop_sec).astype(int)
    hits = flux[np.clip(indices, 0, len(flux) - 1)].max(axis=1)
    measures = hits[:len(hits) // 4 * 4].reshape(-1, 4, 3)
    sections = np.array_split(measures, max(1, len(measures) // 8))
    total = np.mean([_bar_evidence(section) for section in sections], axis=0)
    if phase_index != strongest:
        # Look for the bar accent in the section that establishes the beat.
        first_measure = first_beat // 4
        opening = _bar_evidence(measures[first_measure:first_measure + 8])
        total = 0.35 * total + 0.65 * opening
    ranked = np.argsort(-total, kind="stable")
    margin = float(total[ranked[0]] - total[ranked[1]])
    return bpm, round(float(corrected), 6), (int(ranked[0]) + cycle) % 4, round(margin, 3)
