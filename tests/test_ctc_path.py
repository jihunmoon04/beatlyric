import itertools
import unittest

import numpy as np

from pipeline.ctc_path import forced_ctc_path


class CtcPathTest(unittest.TestCase):
    def test_alignment_at_minimum_frame_count(self):
        log_probs = np.zeros((2, 3), dtype=np.float32)
        path, _ = forced_ctc_path(log_probs, [1, 2], blank=0)
        np.testing.assert_array_equal(path, [1, 2])
        self.assertIsNone(forced_ctc_path(log_probs, [1, 1], blank=0))

        log_probs = np.zeros((3, 3), dtype=np.float32)
        path, _ = forced_ctc_path(log_probs, [1, 1], blank=0)
        np.testing.assert_array_equal(path, [1, 0, 1])
        self.assertIsNone(forced_ctc_path(log_probs, [1, 3], blank=0))

    def test_scores_match_the_best_path_that_collapses_to_the_transcript(self):
        log_probs = np.array([
            [-1, -2, -3], [-3, -1, -2], [-1, -3, -2],
            [-3, -1, -2], [-2, -3, -1],
        ], dtype=np.float32)
        transcript = [1, 1, 2]
        path, scores = forced_ctc_path(log_probs, transcript, blank=0)

        def collapsed(tokens):
            return [token for token, _ in itertools.groupby(tokens) if token != 0]

        possible = (
            sum(log_probs[frame, token] for frame, token in enumerate(candidate))
            for candidate in itertools.product(range(3), repeat=5)
            if collapsed(candidate) == transcript
        )
        self.assertEqual(collapsed(path), transcript)
        self.assertEqual(scores.sum(), max(possible))
