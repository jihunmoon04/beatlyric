"""Bounds-checked CTC forced alignment for one known token sequence."""

from __future__ import annotations

import numpy as np


def forced_ctc_path(
    log_probs: np.ndarray, targets: list[int], blank: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the highest-scoring CTC path and its frame scores, if it exists."""
    frames, classes = log_probs.shape
    if not targets or not 0 <= blank < classes or any(
        token == blank or not 0 <= token < classes for token in targets
    ):
        return None
    repeats = sum(left == right for left, right in zip(targets, targets[1:]))
    if frames < len(targets) + repeats:
        return None

    # A state is either a transcript token or a blank before/after a token.
    labels = np.full(2 * len(targets) + 1, blank, dtype=np.intp)
    labels[1::2] = targets
    can_skip = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    scores = np.full(len(labels), -np.inf, dtype=np.float32)
    scores[:2] = log_probs[0, labels[:2]]
    moves = np.zeros((frames, len(labels)), dtype=np.uint8)
    for frame in range(1, frames):
        best = scores.copy()
        advance = scores[:-1] > best[1:]
        best[1:][advance] = scores[:-1][advance]
        moves[frame, 1:][advance] = 1

        skip = can_skip & (scores[:-2] > best[2:])
        best[2:][skip] = scores[:-2][skip]
        moves[frame, 2:][skip] = 2
        scores = best + log_probs[frame, labels]

    state = len(labels) - (1 if scores[-1] >= scores[-2] else 2)
    if not np.isfinite(scores[state]):
        return None

    path = np.empty(frames, dtype=np.intp)
    for frame in range(frames - 1, -1, -1):
        path[frame] = labels[state]
        state -= int(moves[frame, state])
    return path, log_probs[np.arange(frames), path]
