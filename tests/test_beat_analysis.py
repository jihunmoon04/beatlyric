import unittest

import numpy as np

from pipeline.beat_analysis import estimate_rhythm


class BeatAnalysisTest(unittest.TestCase):
    def test_louder_offbeat_kicks_do_not_shift_an_established_pulse(self):
        sample_rate = 22_528
        samples = np.zeros(sample_rate * 52, dtype=np.float32)
        window = np.arange(1100)

        def hit(time, frequency, amplitude, decay):
            start = round(time * sample_rate)
            attack = np.sin(2 * np.pi * frequency * window / sample_rate)
            samples[start:start + len(window)] += amplitude * attack * np.exp(-window / decay)

        first_beat = 0.125
        for beat in range(102):
            time = first_beat + 0.5 * beat
            if beat % 4 == 0:
                hit(time, 255, 0.15, 300)
            else:
                hit(time, 420, 0.05, 300)
            hit(time, 2230 if beat % 2 else 1855, 0.05, 100)

            if beat >= 32:
                hit(time + 0.25, 65, 1.0, 300)
                hit(time + 0.25, 1140, 0.35, 230)
                hit(time + 0.25, 3100, 0.27, 120)
            elif beat % 2:
                hit(time + 0.25, 65, 0.0135, 270)

        bpm, phase, downbeat, _ = estimate_rhythm(samples, sample_rate, bpm=120.0)

        self.assertEqual(bpm, 120.0)
        self.assertAlmostEqual(phase, first_beat, delta=0.02)
        self.assertEqual(downbeat, 0)
