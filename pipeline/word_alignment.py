"""Align lyric words against vocals inside already fixed score boundaries."""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic
from typing import Any

from common import JobInterrupted, read_json, run_cmd, write_json
from step2_align import CtcVocalAligner, _separate_vocals_demucs
from step4_score import compile_score_to_timeline


_CTC_LOCK = Lock()
_PREVIEW_LOCK = Lock()
_LOG = logging.getLogger(__name__)


class WordAlignmentUnavailable(RuntimeError):
    """The audio or optional word alignment runtime is unavailable."""


def _tokens(line: dict[str, Any]) -> list[str]:
    original = str(line["original"]).split()
    editor_words = [str(word["text"]) for word in line.get("words", [])]
    if editor_words and "".join(original) == "".join("".join(word.split()) for word in editor_words):
        return editor_words
    return original


def _fit_words(
    aligner: CtcVocalAligner,
    acoustic_frames: tuple[Any, float, float],
    tokens: list[str],
    start: float,
    end: float,
    origin: float,
    beat_sec: float,
) -> list[dict[str, Any]] | None:
    emissions, stride_ms, audio_origin = acoustic_frames
    matches = aligner.align_tokens(emissions, stride_ms, tokens)
    if len(matches) != len(tokens) or any(match is None for match in matches):
        return None
    scores = [float(match["score"]) for match in matches]
    # Reject forced paths that squeezed a word through poor acoustic evidence.
    if not all(isfinite(score) for score in scores) or min(scores) < -4.8 or sum(scores) / len(scores) < -2.8:
        return None

    words = []
    previous_end = start
    for index, (token, match) in enumerate(zip(tokens, matches)):
        word_start = max(previous_end, audio_origin + float(match["start"]))
        following = audio_origin + float(matches[index + 1]["start"]) if index + 1 < len(matches) else end
        word_end = min(end, following, audio_origin + float(match["end"]))
        if word_end - word_start < 0.010 - 1e-8:
            return None
        words.append({
            "text": token,
            "rel_beat": round((word_start - origin) / beat_sec, 9),
            "duration_beats": round((word_end - word_start) / beat_sec, 9),
        })
        previous_end = word_end
    return words


def align_score_words(
    work_dir: Path,
    score: dict[str, Any],
    line_ids: set[int] | None = None,
    *,
    isolate_vocals: bool = False,
    preserve_manual: bool = True,
) -> dict[str, Any]:
    """Return word-only changes and review statuses without writing the score."""
    grid = score["grid"]
    beat_sec = 60.0 / float(grid["bpm"])
    beats_per_bar = grid.get("time_signature", [4, 4])[0]
    rendered = {line["id"]: line for line in compile_score_to_timeline(deepcopy(score))["lines"]}
    result: dict[str, Any] = {"updates": [], "review": [], "excluded": 0, "audio_source": None}
    pending = []
    for line in score["lines"]:
        line_id = line["id"]
        if line_ids is not None and line_id not in line_ids:
            continue
        if (line.get("whole_line") or line["original"].strip().startswith("♪")
                or (preserve_manual and line.get("word_timing_edited"))):
            result["excluded"] += 1
            continue
        tokens = _tokens(line)
        bounds = rendered[line_id]
        if not tokens or bounds["end"] - bounds["start"] < 0.04:
            result["review"].append(line_id)
            continue
        beat = (line["bar"] - 1) * beats_per_bar + (line["beat"] - 1)
        origin = float(grid["t0"]) + beat * beat_sec
        pending.append((line_id, tokens, line["original"].split(), bounds, origin))

    if not pending:
        return result
    vocals = work_dir / "vocals.wav"
    mix = work_dir / "analysis.wav"
    if not vocals.exists() and not mix.exists():
        raise WordAlignmentUnavailable("단어 정렬에 필요한 원곡 또는 분리 보컬 파일이 없습니다.")
    with _CTC_LOCK:
        if isolate_vocals and not vocals.exists():
            try:
                _separate_vocals_demucs(mix, vocals)
            except (ImportError, OSError, RuntimeError):
                pass
        audio = vocals if vocals.exists() else mix
        result["audio_source"] = "vocals" if audio == vocals else "mix"
        try:
            aligner = CtcVocalAligner(audio, str(score.get("meta", {}).get("language") or "en"))
        except ImportError as exc:
            raise WordAlignmentUnavailable("CTC 단어 정렬에 필요한 패키지가 없습니다. uv sync --extra align") from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise WordAlignmentUnavailable("CTC 단어 정렬 모델을 불러올 수 없습니다.") from exc

        for line_id, tokens, original_words, bounds, origin in pending:
            start, end = bounds["start"], min(bounds["end"], aligner.duration)
            words = None
            if end - start >= 0.04:
                try:
                    frames = aligner.emissions(start, end)
                except (ValueError, RuntimeError):
                    result["review"].append(line_id)
                    continue
                variants = [tokens]
                if original_words != tokens and len(original_words) > 1:
                    # Stable-TS may have split Japanese into unusably fine kana
                    # or punctuation. The written word breaks can be clearer.
                    variants.append(original_words)
                for variant in variants:
                    try:
                        words = _fit_words(aligner, frames, variant, start, end, origin, beat_sec)
                    except (ValueError, RuntimeError, IndexError, KeyError):
                        continue
                    if words is not None:
                        break
            if words is None:
                result["review"].append(line_id)
            else:
                result["updates"].append({"id": line_id, "words": words})
    return result


def preview_score_words(
    work_dir: Path, score: dict[str, Any], line_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Give each web preview a process boundary around the audio model."""
    if not _PREVIEW_LOCK.acquire(blocking=False):
        raise WordAlignmentUnavailable("다른 단어 정렬이 진행 중입니다. 완료된 뒤 다시 시도해 주세요.")
    try:
        with TemporaryDirectory(prefix="beatlyric-word-preview-") as temporary:
            request = Path(temporary) / "request.json"
            response = Path(temporary) / "response.json"
            write_json(request, {"score": score, "line_ids": sorted(line_ids) if line_ids is not None else None})
            deadline = monotonic() + 600
            try:
                proc = run_cmd(
                    [sys.executable, "-X", "utf8", "-X", "faulthandler", str(Path(__file__).resolve()),
                     str(work_dir.resolve()), str(request), str(response)],
                    check=False, interrupted=lambda: monotonic() >= deadline,
                )
            except JobInterrupted as exc:
                raise WordAlignmentUnavailable("시간 제한을 넘겨 단어 정렬을 멈췄습니다.") from exc
            if proc.returncode or not response.exists():
                _LOG.error("Word-alignment worker exit %s: %s", proc.returncode, proc.stderr[-2500:])
                raise WordAlignmentUnavailable("단어 정렬 작업이 비정상 종료됐습니다. 편집 내용을 보존했습니다.")
            result = read_json(response)
            if "error" in result:
                raise WordAlignmentUnavailable(result["error"])
            return result
    finally:
        _PREVIEW_LOCK.release()



def align_saved_score_words(work_dir: Path, *, initial: bool = False, isolate_vocals: bool = False) -> dict[str, Any]:
    """Apply word-only changes to both score and timeline, preserving saved line bounds."""
    score_path = work_dir / "score.json"
    timeline_path = work_dir / "timeline.json"
    previous_score = read_json(score_path)
    previous_timeline = read_json(timeline_path)
    result = align_score_words(work_dir, previous_score, isolate_vocals=isolate_vocals)
    updates = {update["id"]: update["words"] for update in result["updates"]}
    if not updates:
        return result
    if read_json(score_path) != previous_score or read_json(timeline_path) != previous_timeline:
        raise RuntimeError("단어 분석 중 저장된 악보가 변경되었습니다.")

    score = deepcopy(previous_score)
    for line in score["lines"]:
        if line["id"] in updates:
            line.update(words=updates[line["id"]], word_alignment="ctc")
    compiled = {line["id"]: line for line in compile_score_to_timeline(deepcopy(score))["lines"]}
    timeline = deepcopy(previous_timeline)
    for line in timeline["lines"]:
        if line["id"] in updates:
            row = compiled[line["id"]]
            if (row["start"], row["end"]) != (line["start"], line["end"]):
                raise RuntimeError(f"#{line['id']}의 저장된 줄 경계와 악보가 서로 다릅니다.")
            line["words"] = row["words"]

    write_json(score_path, score)
    try:
        write_json(timeline_path, timeline)
    except Exception:
        write_json(score_path, previous_score)
        raise
    if initial:
        write_json(work_dir / "score_initial.json", score)
    return result


def run_automatic_word_alignment(work_dir: Path, *, isolate_vocals: bool = False) -> None:
    """Run the second alignment stage and leave uncertain lines for review."""
    try:
        result = align_saved_score_words(work_dir, initial=True, isolate_vocals=isolate_vocals)
    except WordAlignmentUnavailable as exc:
        print(f"단어 정렬 생략 · {exc}")
        return

    print(f"단어 정렬 {len(result['updates'])}줄 · 검수 필요 {len(result['review'])}줄")


if __name__ == "__main__":
    directory, request_path, response_path = map(Path, sys.argv[1:])
    job = read_json(request_path)
    ids = job["line_ids"]
    try:
        result = align_score_words(
            directory, job["score"], set(ids) if ids is not None else None, preserve_manual=ids is None,
        )
    except WordAlignmentUnavailable as exc:
        result = {"error": str(exc)}
    write_json(response_path, result)
