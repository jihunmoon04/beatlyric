import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from common import read_json, write_json
from step4_score import compile_score_to_timeline
from word_alignment import (
    WordAlignmentUnavailable,
    align_saved_score_words,
    align_score_words,
    preview_score_words,
)


class FakeCtcAligner:
    duration = 4.0

    def __init__(self, *_args):
        pass

    def emissions(self, start, end):
        return None, 20, start

    def align_tokens(self, emissions, stride, tokens):
        if tokens == ["uncertain"]:
            return [{"start": 0.2, "end": 0.3, "score": -6.0}]
        return [
            {"start": 0.1, "end": 0.2, "score": -1.0},
            {"start": 0.98, "end": 1.0, "score": -1.0},
        ]


class WordAlignmentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "vocals.wav").touch()
        self.score = {
            "grid": {"bpm": 120, "t0": 0, "time_signature": [4, 4], "downbeat_offset": 1},
            "meta": {"language": "en"},
            "lines": [
                {"id": 1, "bar": 1, "beat": 3.0, "duration_beats": 2, "original": "first last",
                 "words": [], "timing_edited": True},
                {"id": 2, "bar": 2, "beat": 1.0, "duration_beats": 2, "original": "uncertain",
                 "words": [{"text": "uncertain", "rel_beat": 0, "duration_beats": 1}]},
            ],
        }
        self.timeline = compile_score_to_timeline(self.score)
        write_json(self.root / "score.json", self.score)
        write_json(self.root / "timeline.json", self.timeline)

    @patch("word_alignment.CtcVocalAligner", FakeCtcAligner)
    def test_saving_only_replaces_confident_words_and_keeps_exact_boundaries(self):
        result = align_saved_score_words(self.root)
        self.assertEqual([update["id"] for update in result["updates"]], [1])
        self.assertEqual(result["review"], [2])

        score = read_json(self.root / "score.json")
        timeline = read_json(self.root / "timeline.json")
        for previous, current in ((self.score, score), (self.timeline, timeline)):
            for field in ("start", "end"):
                self.assertEqual(previous["lines"][0][field], current["lines"][0][field])
            self.assertEqual(previous["lines"][1], current["lines"][1])
        self.assertEqual(score["grid"], self.score["grid"])
        self.assertEqual(timeline["lines"][0]["words"][-1], {"text": "last", "start": 1.98, "end": 2.0})
        self.assertFalse((self.root / "score_initial.json").exists())

    @patch("word_alignment.CtcVocalAligner", FakeCtcAligner)
    def test_bulk_keeps_manual_words_and_selected_preview_can_realign_them(self):
        self.score["lines"][0]["word_timing_edited"] = True
        bulk = align_score_words(self.root, self.score)
        picked = align_score_words(self.root, self.score, {1}, preserve_manual=False)
        self.assertEqual(bulk["excluded"], 1)
        self.assertEqual(bulk["updates"], [])
        self.assertEqual([update["id"] for update in picked["updates"]], [1])
        self.assertEqual(read_json(self.root / "timeline.json"), self.timeline)

    def test_failed_preview_worker_releases_lock_and_preserves_saved_files(self):
        with patch("word_alignment.run_cmd", return_value=CompletedProcess([], 0xC0000374, "", "worker crash")):
            with self.assertLogs("word_alignment", level="ERROR"):
                with self.assertRaises(WordAlignmentUnavailable):
                    preview_score_words(self.root, self.score)

        def successful_worker(cmd, **_options):
            write_json(Path(cmd[-1]), {"updates": [], "review": []})
            return CompletedProcess(cmd, 0, "", "")

        with patch("word_alignment.run_cmd", side_effect=successful_worker):
            self.assertEqual(preview_score_words(self.root, self.score)["updates"], [])

        self.assertEqual(read_json(self.root / "score.json"), self.score)
        self.assertEqual(read_json(self.root / "timeline.json"), self.timeline)
