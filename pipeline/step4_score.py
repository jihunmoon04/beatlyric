"""Step 4: Automated beat grid estimation, 0.5-beat score quantization, and timeline compilation."""

from __future__ import annotations

import re
import wave
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from beat_analysis import estimate_rhythm
from common import read_json, write_json

CONSONANT_LEAD_SEC = 0.115
PAREN_GROUP_RE = re.compile(r"\(([^)]+)\)")

DISPLAY_META_DEFAULTS: dict[str, Any] = {
    "content_scale": 1.18,
    "karaoke_sweep": True,
    "show_word_highlight": True,
    "show_pronunciation": True,
    "show_comment": True,
    "smooth_lyric_scroll": True,
    "lyric_word_lift": True,
    "show_song_context": True,
}


def _ensure_display_meta_defaults(meta: dict[str, Any]) -> bool:
    """Populate missing display-option defaults in meta; return True if any key was added."""
    changed = False
    for key, val in DISPLAY_META_DEFAULTS.items():
        if key not in meta:
            meta[key] = val
            changed = True
    return changed


def _read_mono_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sr


def estimate_beat_grid(
    analysis_wav: Path, bpm: float | None = None,
) -> tuple[float, float, int, float]:
    samples, sr = _read_mono_wav(analysis_wav)
    return estimate_rhythm(samples, sr, bpm)


@lru_cache(maxsize=8)
def waveform_peaks(path: Path, last_modified_ns: int) -> dict[str, Any]:
    """Downsample the analysis audio to roughly 10 ms min/max sections."""
    samples, sr = _read_mono_wav(path)
    step = max(1, round(sr * 0.01))
    windows = np.pad(samples, (0, -len(samples) % step)).reshape(-1, step)
    return {
        "interval": step / sr,
        "min": (windows.min(axis=1) * 127).astype(np.int8).tolist(),
        "max": (windows.max(axis=1) * 127).astype(np.int8).tolist(),
    }


def _quantize_words_to_rel_beats(
    words: list[dict[str, Any]],
    line_start_sec: float,
    line_dur_beats: float,
    beat_sec: float,
) -> list[dict[str, Any]]:
    """Convert word timestamps into 0.25-beat offsets relative to line start."""
    if not words:
        return []

    q_words: list[dict[str, Any]] = []
    min_step = 0.25
    cursor_rel = 0.0

    for idx, w in enumerate(words):
        raw_rel = max(0.0, (float(w["start"]) - line_start_sec) / beat_sec)
        raw_end_rel = max(raw_rel + min_step, (float(w["end"]) - line_start_sec) / beat_sec)

        rel_beat = round(raw_rel * 4.0) / 4.0
        rel_beat = max(cursor_rel, min(max(0.0, line_dur_beats - min_step), rel_beat))

        end_rel = round(raw_end_rel * 4.0) / 4.0
        end_rel = max(rel_beat + min_step, min(line_dur_beats, end_rel))

        # Leave room for remaining words if crowded near end of line
        remaining = len(words) - idx - 1
        max_end = max(rel_beat + min_step, line_dur_beats - remaining * 0.125)
        if end_rel > max_end and line_dur_beats >= len(words) * 0.125:
            end_rel = round(max_end * 4.0) / 4.0
            end_rel = max(rel_beat + 0.125, end_rel)

        dur_b = round(max(0.125, end_rel - rel_beat), 3)
        q_words.append({
            "text": w["text"],
            "rel_beat": round(rel_beat, 3),
            "duration_beats": dur_b,
        })
        cursor_rel = round(rel_beat + dur_b, 3)

    return q_words


def _compute_vocal_envelope(wav_path: Path, hop_sec: float = 0.01) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute normalized RMS and positive onset envelope from isolated vocals (or mix)."""
    samples, sr = _read_mono_wav(wav_path)
    hop = max(1, int(round(sr * hop_sec)))
    n_frames = max(1, len(samples) // hop)
    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_scale = float(np.percentile(rms, 95))
    norm_rms = np.clip(rms / (rms_scale if rms_scale > 1e-6 else 1.0), 0.0, 2.0)

    onset = np.maximum(0.0, np.diff(norm_rms, prepend=norm_rms[0]))
    pos = onset[onset > 0]
    onset_scale = float(np.percentile(pos, 95)) if len(pos) > 0 else 1.0
    norm_onset = np.clip(onset / (onset_scale if onset_scale > 1e-6 else 1.0), 0.0, 2.0)
    return norm_rms, norm_onset, hop / float(sr)


def _line_start_hint(line: dict[str, Any], beat_sec: float) -> float:
    start = float(line["start"])
    words = line.get("words") or []
    if not words:
        return start

    first_word = float(words[0]["start"])
    lrc_start = line.get("lrc_start")
    if lrc_start is not None and start - first_word > beat_sec:
        # An upstream overlap cut can push the line past its own words.
        # Recover the earlier window only when the original LRC supports it.
        if abs(float(lrc_start) - first_word) <= beat_sec:
            return first_word
    if abs(first_word - start) <= beat_sec * 0.25:
        return (start + first_word) / 2
    return start


def _choose_line_starts(
    lines: list[dict[str, Any]],
    t0: float,
    beat_sec: float,
    beats_per_bar: int,
    downbeat_offset: int,
    vocal_env: tuple[np.ndarray, np.ndarray, float] | None,
) -> list[float]:
    """Choose a compatible sequence of half-beat line entries."""
    if not lines:
        return []

    minimum = np.floor(-t0 / beat_sec * 2.0) / 2.0
    options: list[dict[float, tuple[float, float | None]]] = []
    for line in lines:
        instrumental = str(line.get("original", "")).strip().startswith("♪")
        hint = float(line["start"]) if instrumental else _line_start_hint(line, beat_sec) - CONSONANT_LEAD_SEC
        target = (hint - t0) / beat_sec
        center = round(target * 2.0) / 2.0
        candidates = {max(minimum, center)}
        previous = options[-1] if options else {}
        if not instrumental:
            candidates.update(max(minimum, center + delta) for delta in (-1.0, -0.5, 0.5, 1.0))
        if previous:
            candidates.add(min(previous) + 0.5)

        scores: dict[float, tuple[float, float | None]] = {}
        for beat in sorted(candidates):
            compatible = [old for old in previous if old <= beat - 0.5]
            if previous and not compatible:
                continue
            parent = max(compatible, key=lambda old: previous[old][0]) if compatible else None
            value = -1.15 * abs(beat - target)
            if not instrumental:
                phase = (beat - downbeat_offset) % beats_per_bar
                if phase == 0:
                    value += 0.18
                elif phase == round(phase):
                    value += 0.06
                if vocal_env is not None:
                    rms, onset, hop = vocal_env
                    idx = round(max(0.0, t0 + beat * beat_sec) / hop)
                    early = rms[max(0, idx - round(0.12 / hop)):idx]
                    late = rms[idx:idx + round(0.14 / hop)]
                    attacks = onset[idx:idx + round(0.15 / hop)]
                    if not len(late) or not len(attacks):
                        attacks = np.zeros(1)
                    gain = (float(late.mean()) if len(late) else 0) - (float(early.mean()) if len(early) else 0)
                    value += 0.20 * gain + 0.14 * float(attacks.max())
            scores[beat] = (value + (previous[parent][0] if parent is not None else 0), parent)
        options.append(scores)

    beat = max(options[-1], key=lambda candidate: options[-1][candidate][0])
    selected = []
    for scores in reversed(options):
        selected.append(beat)
        beat = scores[beat][1]
    return selected[::-1]


def _should_use_whole_line(original: str) -> bool:
    """Detect dual-voice lines mixing outside singing with long parenthetical narration."""
    matches = PAREN_GROUP_RE.findall(original)
    if not matches:
        return False
    outside = PAREN_GROUP_RE.sub("", original).strip()
    if len(outside) < 3:
        return False
    for inner in matches:
        tokens = [re.sub(r"[^\w]", "", tok.lower()) for tok in inner.split()]
        tokens = [t for t in tokens if t]
        if len(tokens) >= 4 and len(set(tokens)) >= 3:
            return True
    return False


def _apply_repeated_line_consensus(score_lines: list[dict[str, Any]]) -> None:
    """Regularize pathological word crowding across repeated lines (e.g., choruses) via median consensus."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for ln in score_lines:
        words = ln.get("words") or []
        if ln.get("whole_line") or len(words) < 3:
            continue
        key = (re.sub(r"\s+", " ", str(ln.get("original", "")).strip().lower()), len(words))
        groups[key].append(ln)

    for (_key, n_words), items in groups.items():
        if len(items) < 2:
            continue
        rel_fracs = []
        dur_fracs = []
        for ln in items:
            dur_b = max(0.5, float(ln["duration_beats"]))
            rel_fracs.append([float(w.get("rel_beat", 0.0)) / dur_b for w in ln["words"]])
            dur_fracs.append([float(w.get("duration_beats", 0.25)) / dur_b for w in ln["words"]])

        med_rel = np.median(np.array(rel_fracs), axis=0)
        med_dur = np.median(np.array(dur_fracs), axis=0)

        for idx, ln in enumerate(items):
            max_dev = float(np.max(np.abs(np.array(rel_fracs[idx]) - med_rel)))
            if max_dev > 0.14:
                dur_b = max(0.5, float(ln["duration_beats"]))
                for j, w in enumerate(ln["words"]):
                    new_rel = round(float(med_rel[j]) * dur_b * 4.0) / 4.0
                    new_dur = max(0.25, round(float(med_dur[j]) * dur_b * 4.0) / 4.0)
                    w["rel_beat"] = round(max(0.0, min(dur_b - 0.25, new_rel)), 3)
                    w["duration_beats"] = round(min(dur_b - w["rel_beat"], new_dur), 3)


def enforce_score_integrity(score: dict[str, Any]) -> dict[str, Any]:
    """Enforce zero line overlaps, close <=0.5b micro-gaps, separate colliding words, and bound word durations."""
    grid = score.get("grid", {})
    bpm = float(grid.get("bpm", 120.0))
    ts = grid.get("time_signature", [4, 4])
    t0 = float(grid.get("t0", 0.0))
    beats_per_bar = float(ts[0])
    beat_sec = 60.0 / bpm

    lines = sorted(score.get("lines", []), key=lambda x: x["id"])

    def _start_beats(ln: dict[str, Any]) -> float:
        return (int(ln["bar"]) - 1) * beats_per_bar + (float(ln["beat"]) - 1.0)

    # Line overlap / <=0.5b micro-gap pass
    for i in range(len(lines) - 1):
        cur = lines[i]
        nxt = lines[i + 1]
        cur_s = _start_beats(cur)
        cur_d = max(0.5, float(cur["duration_beats"]))
        nxt_s = _start_beats(nxt)
        gap = round(nxt_s - (cur_s + cur_d), 2)
        if gap < 0.0:
            cur["duration_beats"] = round(max(0.5, nxt_s - cur_s), 2)
        elif 0.0 < gap <= 0.5 and not str(cur.get("original", "")).strip().startswith("♪"):
            cur["duration_beats"] = round(cur_d + gap, 2)

    # Sync start/end seconds & clamp intra-line words
    for ln in lines:
        s_b = _start_beats(ln)
        d_b = max(0.5, float(ln["duration_beats"]))
        ln["start"] = round(max(0.0, t0 + s_b * beat_sec), 3)
        ln["end"] = round(max(0.001, t0 + (s_b + d_b) * beat_sec), 3)

        words = ln.get("words") or []
        if words and not ln.get("whole_line"):
            for j, w in enumerate(words):
                rb = max(0.0, round(float(w.get("rel_beat", 0.0)) * 4.0) / 4.0)
                if j > 0:
                    prev_rb = float(words[j - 1]["rel_beat"])
                    if rb <= prev_rb:
                        rb = round(min(d_b - 0.125, prev_rb + 0.25), 3)
                w["rel_beat"] = round(min(max(0.0, d_b - 0.125), rb), 3)

            for j, w in enumerate(words):
                rb = float(w["rel_beat"])
                next_rb = float(words[j + 1]["rel_beat"]) if j + 1 < len(words) else d_b
                cap = min(d_b, max(rb + 0.125, next_rb))
                db = max(0.125, round(float(w.get("duration_beats", 0.25)) * 4.0) / 4.0)
                if rb + db > cap + 1e-6:
                    db = round(max(0.125, cap - rb), 3)
                w["duration_beats"] = db

    return score


def quantize_timeline_to_score(
    timeline_or_aligned: dict[str, Any],
    bpm: float,
    t0: float,
    time_signature: list[int] | None = None,
    vocals_wav: Path | None = None,
    downbeat_offset: int = 0,
) -> dict[str, Any]:
    """Quantize timed lines & words onto a 0.5-beat musical bar/beat score."""
    ts = time_signature or [4, 4]
    beats_per_bar = ts[0]
    beat_sec = 60.0 / bpm

    vocal_env: tuple[np.ndarray, np.ndarray, float] | None = None
    if vocals_wav is not None and vocals_wav.exists():
        vocal_env = _compute_vocal_envelope(vocals_wav)

    raw_lines = sorted(timeline_or_aligned.get("lines", []), key=lambda x: x["id"])

    snapped_starts = _choose_line_starts(raw_lines, t0, beat_sec, beats_per_bar, downbeat_offset, vocal_env)

    # Pass 2: Snap end beats without overlapping subsequent snapped line starts
    score_lines: list[dict[str, Any]] = []
    for i, ln in enumerate(raw_lines):
        start_b = snapped_starts[i]
        raw_end_b = max(start_b + 0.5, (float(ln["end"]) - t0) / beat_sec)
        end_b = max(start_b + 0.5, round(raw_end_b * 2.0) / 2.0)
        if i + 1 < len(snapped_starts):
            end_b = max(start_b + 0.5, min(end_b, snapped_starts[i + 1]))

        dur_beats = round(end_b - start_b, 2)
        bar = 1 + int(start_b // beats_per_bar)
        beat = round(1.0 + (start_b % beats_per_bar), 2)

        line_start_sec = t0 + start_b * beat_sec
        line_end_sec = t0 + (start_b + dur_beats) * beat_sec

        q_words = _quantize_words_to_rel_beats(
            ln.get("words", []),
            line_start_sec=line_start_sec,
            line_dur_beats=dur_beats,
            beat_sec=beat_sec,
        )

        score_lines.append({
            "id": ln["id"],
            "bar": bar,
            "beat": beat,
            "duration_beats": dur_beats,
            "start": round(line_start_sec, 3),
            "end": round(line_end_sec, 3),
            "original": ln["original"],
            "words": q_words,
            "whole_line": bool(ln.get("whole_line", False)) or _should_use_whole_line(str(ln["original"])),
            "pronunciation": ln.get("pronunciation", ""),
            "translation": ln.get("translation", ""),
            "comment": ln.get("comment"),
            "culture_note": ln.get("culture_note"),
        })

    _apply_repeated_line_consensus(score_lines)

    meta = dict(timeline_or_aligned.get("meta", {}))
    _ensure_display_meta_defaults(meta)

    score = {
        "grid": {
            "bpm": bpm,
            "time_signature": ts,
            "t0": t0,
        },
        "meta": meta,
        "lines": score_lines,
    }
    return enforce_score_integrity(score)


def compile_score_to_timeline(score: dict[str, Any]) -> dict[str, Any]:
    """Deterministically compile score.json (bars/beats + relative word beats) into timeline.json."""
    grid = score.get("grid", {})
    bpm = float(grid.get("bpm", 120.0))
    ts = grid.get("time_signature", [4, 4])
    t0 = float(grid.get("t0", 0.0))
    beats_per_bar = float(ts[0])
    beat_sec = 60.0 / bpm

    compiled_lines: list[dict[str, Any]] = []
    precise_ids = {ln["id"] for ln in score.get("lines", []) if ln.get("word_alignment") == "ctc"}
    for ln in score.get("lines", []):
        bar = int(ln["bar"])
        beat = float(ln["beat"])
        dur_beats = max(0.001, float(ln["duration_beats"]))
        total_beats = (bar - 1) * beats_per_bar + (beat - 1.0)

        start = round(max(0.0, t0 + total_beats * beat_sec), 3)
        end = round(max(start + 0.001, t0 + (total_beats + dur_beats) * beat_sec), 3)

        # Keep score line start/end in sync as well
        ln["start"] = start
        ln["end"] = end

        raw_words = ln.get("words") or []
        minimum_word = 0.001 if ln.get("word_alignment") == "ctc" else 0.04
        words_out: list[dict[str, Any]] = []
        if raw_words:
            for w in raw_words:
                if "rel_beat" in w:
                    rel_b = max(0.0, float(w["rel_beat"]))
                    w_dur = max(0.001, float(w.get("duration_beats", 0.25)))
                    ws = round(max(start, min(end - minimum_word, t0 + (total_beats + rel_b) * beat_sec)), 3)
                    we = round(min(end, t0 + (total_beats + rel_b + w_dur) * beat_sec), 3)
                else:
                    ws = round(max(start, min(end - 0.04, float(w.get("start", start)))), 3)
                    we = round(max(ws + 0.04, min(end, float(w.get("end", ws + 0.1)))), 3)
                    if words_out and ws < words_out[-1]["end"]:
                        ws = words_out[-1]["end"]
                if we <= ws:
                    we = round(min(end, ws + minimum_word), 3)
                words_out.append({
                    "text": w["text"],
                    "start": ws,
                    "end": max(ws, we),
                })
        else:
            tokens = str(ln["original"]).split()
            span = end - start
            step = span / max(1, len(tokens))
            for i, tok in enumerate(tokens):
                words_out.append({
                    "text": tok,
                    "start": round(start + i * step, 3),
                    "end": round(start + (i + 1) * step, 3),
                })

        compiled_lines.append({
            "id": ln["id"],
            "start": start,
            "end": end,
            "original": ln["original"],
            "words": words_out,
            "whole_line": bool(ln.get("whole_line", False)),
            "pronunciation": ln.get("pronunciation", ""),
            "translation": ln.get("translation", ""),
            "comment": ln.get("comment"),
            "culture_note": ln.get("culture_note"),
        })

    # Ensure strictly non-overlapping bounds if user nudged lines tightly
    compiled_lines.sort(key=lambda x: (x["start"], x["id"]))
    for i in range(1, len(compiled_lines)):
        prev = compiled_lines[i - 1]
        cur = compiled_lines[i]
        if cur["start"] < prev["end"] and prev["start"] < cur["start"]:
            prev["end"] = cur["start"]
            minimum_word = 0.001 if prev["id"] in precise_ids else 0.04
            for w in prev["words"]:
                w["start"] = min(w["start"], max(prev["start"], prev["end"] - minimum_word))
                w["end"] = min(w["end"], prev["end"])

    meta = dict(score.get("meta", {}))
    _ensure_display_meta_defaults(meta)

    return {
        "meta": meta,
        "lines": compiled_lines,
    }


def ensure_score_file(
    work_dir: Path,
    bpm: float | None = None,
    t0: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Load or build score.json for a work directory, upgrading word relative beats if absent."""
    score_path = work_dir / "score.json"
    timeline_path = work_dir / "timeline.json"
    aligned_path = work_dir / "aligned_lyrics.json"
    raw_llm_path = work_dir / "raw_lyrics_for_llm.json"

    source_tl: dict[str, Any] | None = None
    for candidate in (aligned_path, raw_llm_path, timeline_path):
        if candidate.exists():
            source_tl = read_json(candidate)
            break
    if source_tl is None:
        raise FileNotFoundError(f"No timeline.json or aligned_lyrics.json found in {work_dir}")

    initial_path = work_dir / "score_initial.json"

    if score_path.exists() and not force:
        score = read_json(score_path)
        if "meta" not in score:
            score["meta"] = dict(source_tl.get("meta", {}))
        grid = score.get("grid", {})
        beat_sec = 60.0 / float(grid.get("bpm", 120.0))
        tl_by_id = {ln["id"]: ln for ln in source_tl.get("lines", [])}
        upgraded = _ensure_display_meta_defaults(score.setdefault("meta", {}))
        for ln in score.get("lines", []):
            if "whole_line" not in ln:
                ln["whole_line"] = False
                upgraded = True
            if not ln.get("words"):
                tl_ln = tl_by_id.get(ln["id"])
                if tl_ln and tl_ln.get("words"):
                    ln["words"] = _quantize_words_to_rel_beats(
                        tl_ln["words"],
                        line_start_sec=float(ln["start"]),
                        line_dur_beats=float(ln["duration_beats"]),
                        beat_sec=beat_sec,
                    )
                    upgraded = True
                else:
                    tokens = str(ln["original"]).split()
                    dur_b = float(ln["duration_beats"])
                    step = dur_b / max(1, len(tokens))
                    ln["words"] = [
                        {
                            "text": tok,
                            "rel_beat": round(i * step, 3),
                            "duration_beats": round(step, 3),
                        }
                        for i, tok in enumerate(tokens)
                    ]
                    upgraded = True
        if upgraded:
            write_json(score_path, score)
        if not initial_path.exists():
            write_json(initial_path, score)
        return score

    offset = 0
    margin = None
    if t0 is None:
        bpm, t0, offset, margin = estimate_beat_grid(work_dir / "analysis.wav", bpm=bpm)
    elif bpm is None:
        bpm, _, _, _ = estimate_beat_grid(work_dir / "analysis.wav")

    vocals_wav = (work_dir / "vocals.wav") if (work_dir / "vocals.wav").exists() else (work_dir / "analysis.wav")
    score = quantize_timeline_to_score(
        source_tl, bpm=bpm, t0=t0, vocals_wav=vocals_wav, downbeat_offset=offset,
    )
    score["grid"]["downbeat_offset"] = offset
    if margin is not None:
        score["grid"]["downbeat_margin"] = margin
    write_json(score_path, score)
    write_json(initial_path, score)
    compiled = compile_score_to_timeline(score)
    write_json(timeline_path, compiled)
    return score


def _copy_rebased_timing(
    line: dict[str, Any], previous: dict[str, Any],
    old_grid: dict[str, Any], new_grid: dict[str, Any],
) -> None:
    """Copy a line's timing while keeping its original position in the song."""
    old_period = 60.0 / old_grid["bpm"]
    new_period = 60.0 / new_grid["bpm"]
    beats_per_bar = new_grid["time_signature"][0]
    beats = (previous["bar"] - 1) * beats_per_bar + (previous["beat"] - 1)
    ratio = old_period / new_period
    new_beats = (old_grid["t0"] - new_grid["t0"]) / new_period + beats * ratio
    bar = int(new_beats // beats_per_bar)
    line.update(bar=bar + 1, beat=1 + (new_beats - bar * beats_per_bar),
                duration_beats=previous["duration_beats"] * ratio)
    line["words"] = [dict(word) for word in previous.get("words", [])]
    for word in line["words"]:
        if "rel_beat" in word:
            word["rel_beat"] *= ratio
        if "duration_beats" in word:
            word["duration_beats"] *= ratio


def _keep_manual_neighbors(
    lines: list[dict[str, Any]], previous: dict[int, dict[str, Any]],
    old_grid: dict[str, Any], grid: dict[str, Any],
) -> None:
    """Keep an old neighboring placement when a new one would clip a manual line."""
    beats_per_bar = grid["time_signature"][0]
    protected = {line["id"] for line in lines if line.get("timing_edited")}

    def start(line: dict[str, Any]) -> float:
        return (line["bar"] - 1) * beats_per_bar + line["beat"] - 1

    # Reverting a neighbor can expose another collision at its other boundary.
    pairs = list(zip(lines, lines[1:]))
    while protected:
        added = set()
        for left, right in pairs:
            if start(left) + left["duration_beats"] <= start(right) + 1e-7:
                continue
            one, other = left, right
            if right["id"] in protected:
                one, other = right, left
            if one["id"] in protected and other["id"] not in protected and other["id"] in previous:
                _copy_rebased_timing(other, previous[other["id"]], old_grid, grid)
                added.add(other["id"])
        if not added:
            return
        protected.update(added)


def requantize_score_with_grid(
    work_dir: Path,
    bpm: float,
    t0: float,
    current_score: dict[str, Any] | None = None,
    persist: bool = True,
    lines_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-quantize original aligned lyrics onto a new (bpm, t0) grid while preserving annotations."""
    if not 40 <= bpm <= 240 or not -2 <= t0 <= 30:
        raise ValueError("BPM이나 박자 위치 범위를 확인하세요.")
    aligned_path = work_dir / "aligned_lyrics.json"
    raw_llm_path = work_dir / "raw_lyrics_for_llm.json"
    timeline_path = work_dir / "timeline.json"
    score_path = work_dir / "score.json"

    source_tl: dict[str, Any] | None = None
    for candidate in (aligned_path, raw_llm_path, timeline_path):
        if candidate.exists():
            source_tl = read_json(candidate)
            break
    if source_tl is None:
        raise FileNotFoundError(f"No aligned_lyrics.json or timeline.json found in {work_dir}")

    base_score = current_score
    if base_score is None and score_path.exists():
        base_score = read_json(score_path)

    vocals_wav = (work_dir / "vocals.wav") if (work_dir / "vocals.wav").exists() else (work_dir / "analysis.wav")
    ts = (base_score or {}).get("grid", {}).get("time_signature", [4, 4])
    new_score = quantize_timeline_to_score(
        source_tl,
        bpm=round(float(bpm), 2),
        t0=round(float(t0), 6),
        time_signature=ts,
        vocals_wav=vocals_wav,
        downbeat_offset=(base_score or {}).get("grid", {}).get("downbeat_offset", 0),
    )

    if base_score:
        new_score["meta"].update(base_score.get("meta", {}))
        base_grid = base_score.get("grid", {})
        new_score["grid"]["downbeat_offset"] = base_grid.get("downbeat_offset", 0)
        if base_grid.get("t0") == new_score["grid"]["t0"] and base_grid.get("bpm") == new_score["grid"]["bpm"]:
            if "downbeat_margin" in base_grid:
                new_score["grid"]["downbeat_margin"] = base_grid["downbeat_margin"]

        baseline_path = work_dir / "score_initial.json"
        baseline = read_json(baseline_path) if baseline_path.exists() else {}
        baseline_grid = baseline.get("grid", {})
        same_grid = all(baseline_grid.get(key) == base_grid.get(key) for key in ("bpm", "t0"))
        initial_lines = {row["id"]: row for row in baseline.get("lines", [])} if same_grid else {}
        timing_fields = ("bar", "beat", "duration_beats") if lines_only else ("bar", "beat", "duration_beats", "words")
        by_id = {int(ln["id"]): ln for ln in base_score.get("lines", []) if "id" in ln}
        for ln in new_score.get("lines", []):
            prev = by_id.get(int(ln["id"]))
            if not prev:
                continue
            for field in ("original", "pronunciation", "translation", "comment", "culture_note", "whole_line"):
                if field in prev:
                    ln[field] = prev[field]

            original = initial_lines.get(ln["id"])
            legacy_edit = original and any(prev.get(field) != original.get(field) for field in timing_fields)
            if lines_only or prev.get("word_alignment") == "ctc" or prev.get("word_timing_edited"):
                ln["words"] = deepcopy(prev.get("words", []))
                for field in ("word_alignment", "word_timing_edited"):
                    if field in prev:
                        ln[field] = prev[field]
            if prev.get("timing_edited") or legacy_edit:
                _copy_rebased_timing(ln, prev, base_grid, new_score["grid"])
                ln["timing_edited"] = True

        _keep_manual_neighbors(new_score["lines"], by_id, base_grid, new_score["grid"])
    compiled = compile_score_to_timeline(new_score)
    if persist:
        write_json(score_path, new_score)
        write_json(timeline_path, compiled)
    return new_score, compiled


def reset_to_algorithm_score(
    work_dir: Path,
    current_score: dict[str, Any] | None = None,
    persist: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reanalyze the audio at the current BPM while preserving lyric timing."""
    score_path = work_dir / "score.json"
    if current_score is None:
        current_score = read_json(score_path)

    score = deepcopy(current_score)
    previous_grid = current_score["grid"]
    bpm = float(previous_grid["bpm"])
    if not 40 <= bpm <= 240:
        raise ValueError("BPM은 40~240이어야 합니다.")

    _, phase, downbeat, margin = estimate_beat_grid(work_dir / "analysis.wav", bpm=bpm)
    grid = score["grid"]
    beats_per_bar = grid.setdefault("time_signature", [4, 4])[0]
    grid["t0"] = phase
    if beats_per_bar == 4:
        grid.update(downbeat_offset=downbeat, downbeat_margin=margin)
    else:
        grid.pop("downbeat_margin", None)

    if phase != previous_grid["t0"]:
        for line, previous_line in zip(score["lines"], current_score["lines"]):
            _copy_rebased_timing(line, previous_line, previous_grid, grid)

    _ensure_display_meta_defaults(score.setdefault("meta", {}))
    timeline = compile_score_to_timeline(score)
    if persist:
        write_json(score_path, score)
        write_json(work_dir / "timeline.json", timeline)
    return score, timeline
