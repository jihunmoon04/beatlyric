"""Step 4: Automated beat grid estimation, 0.5-beat score quantization, and timeline compilation."""

from __future__ import annotations

import re
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import read_json, write_json

CONSONANT_LEAD_SEC = 0.085
PAREN_GROUP_RE = re.compile(r"\(([^)]+)\)")


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
    analysis_wav: Path,
    aligned_lines: list[dict[str, Any]] | None = None,
) -> tuple[float, float]:
    """Estimate studio BPM and beat-1 phase t0 (seconds) from audio onsets."""
    samples, sr = _read_mono_wav(analysis_wav)
    hop = 256
    n_frames = max(1, len(samples) // hop)
    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    env = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    onset = np.maximum(0.0, np.diff(env, prepend=env[0]))

    # Autocorrelation over 90..165 BPM
    ac_input = onset - onset.mean()
    ac = np.correlate(ac_input, ac_input, mode="full")[len(onset) - 1 :]
    min_lag = max(1, int((60.0 / 165.0) * sr / hop))
    max_lag = min(len(ac) - 1, int((60.0 / 90.0) * sr / hop))
    best_lag = min_lag + int(np.argmax(ac[min_lag : max_lag + 1]))
    rough_bpm = 60.0 / (best_lag * hop / sr)

    # Fine search around rough_bpm (and double/half octave check) for onset pulse energy
    duration = len(samples) / sr
    best_score = -1.0
    best_bpm = rough_bpm
    best_phi = 0.0

    candidates = np.linspace(max(80.0, rough_bpm - 1.8), min(180.0, rough_bpm + 1.8), 73)
    for bpm in candidates:
        b_sec = 60.0 / float(bpm)
        for phi in np.linspace(0.0, b_sec, 60, endpoint=False):
            beats = np.arange(phi, min(duration, 180.0), b_sec)
            idx = np.clip((beats * sr / hop).astype(int), 0, len(onset) - 1)
            score = float(np.mean(onset[idx]))
            if score > best_score:
                best_score = score
                best_bpm = float(bpm)
                best_phi = float(phi)

    # Snap BPM to nearest 0.5 if very close to studio integer/half-integer BPM
    snapped_bpm = round(best_bpm * 2.0) / 2.0
    if abs(best_bpm - snapped_bpm) <= 0.25:
        best_bpm = snapped_bpm
        # Re-optimize phase t0 for snapped BPM
        b_sec = 60.0 / best_bpm
        best_phi_score = -1.0
        for phi in np.linspace(0.0, b_sec, 120, endpoint=False):
            beats = np.arange(phi, min(duration, 210.0), b_sec)
            idx = np.clip((beats * sr / hop).astype(int), 0, len(onset) - 1)
            score = float(np.mean(onset[idx]))
            if score > best_phi_score:
                best_phi_score = score
                best_phi = float(phi)

    # Optional phase alignment hint from aligned singing lines (keep within 4-beat bar phase)
    if aligned_lines:
        singing = [ln for ln in aligned_lines if not ln.get("original", "").startswith("♪")]
        if singing:
            b_sec = 60.0 / best_bpm
            # Pick which beat of the 4-beat bar corresponds to Bar 1 Beat 1
            # Test 4 beat offsets (0, 1, 2, 3) and choose the one placing most lines near Beat 1
            best_bar_offset = 0
            best_bar_hits = -1
            for off in range(4):
                cand_t0 = best_phi + off * b_sec
                hits = sum(
                    1
                    for ln in singing[:16]
                    if ((round((ln["start"] - cand_t0) / b_sec)) % 4) == 0
                )
                if hits > best_bar_hits:
                    best_bar_hits = hits
                    best_bar_offset = off
            best_phi = (best_phi + best_bar_offset * b_sec) % (4.0 * b_sec)
            # Keep t0 within [0, 4*b_sec) and prefer small positive offset if > 2*b_sec
            while best_phi >= b_sec:
                # Shift back by 4 beats only if still >= 0
                if best_phi - 4.0 * b_sec >= 0.0:
                    best_phi -= 4.0 * b_sec
                else:
                    break

    return round(best_bpm, 2), round(best_phi, 4)


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


def _snap_line_start_beat_with_vocals(
    raw_start_b: float,
    min_beat: float,
    bpm: float,
    t0: float,
    beats_per_bar: int,
    rms: np.ndarray,
    onset: np.ndarray,
    hop_sec: float,
) -> float:
    """Pick the best 0.5-beat line entry within [-1.5, +1.5] beats using vocal attack contrast."""
    beat_sec = 60.0 / bpm
    raw_lead_b = max(0.0, raw_start_b - CONSONANT_LEAD_SEC / beat_sec)
    base_b = round(raw_lead_b * 2.0) / 2.0
    candidates = [
        base_b + off
        for off in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
        if base_b + off >= min_beat
    ]
    if not candidates:
        return max(min_beat, base_b)

    pre_n = max(1, int(round(0.18 / hop_sec)))
    post_n = max(1, int(round(0.28 / hop_sec)))
    on_pre_n = max(1, int(round(0.06 / hop_sec)))
    on_post_n = max(1, int(round(0.08 / hop_sec)))

    best_b = candidates[0]
    best_score = -1e9
    for cand_b in candidates:
        t = t0 + cand_b * beat_sec
        idx = int(round(t / hop_sec))
        pre_s = rms[max(0, idx - pre_n) : max(1, idx)]
        post_s = rms[max(0, idx) : min(len(rms), idx + post_n + 1)]
        onset_w = onset[max(0, idx - on_pre_n) : min(len(onset), idx + on_post_n + 1)]

        pre_rms = float(np.mean(pre_s)) if len(pre_s) else 0.0
        post_rms = float(np.mean(post_s)) if len(post_s) else 0.0
        peak_onset = float(np.max(onset_w)) if len(onset_w) else 0.0

        attack = (post_rms - 0.55 * pre_rms) + 1.2 * peak_onset
        prox_penalty = 0.32 * abs(cand_b - raw_lead_b)
        in_bar = round((cand_b % beats_per_bar) * 2.0) / 2.0
        prior = 0.0
        if in_bar == 0.0:
            prior = 0.22
        elif in_bar == 2.0:
            prior = 0.12
        elif abs(in_bar - (beats_per_bar - 0.5)) < 0.01:
            prior = 0.10

        total = attack + prior - prox_penalty
        if total > best_score:
            best_score = total
            best_b = cand_b

    return round(best_b, 2)


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
        ln["start"] = round(t0 + s_b * beat_sec, 3)
        ln["end"] = round(t0 + (s_b + d_b) * beat_sec, 3)

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
) -> dict[str, Any]:
    """Quantize timed lines & words onto a 0.5-beat musical bar/beat score."""
    ts = time_signature or [4, 4]
    beats_per_bar = ts[0]
    beat_sec = 60.0 / bpm

    vocal_env: tuple[np.ndarray, np.ndarray, float] | None = None
    if vocals_wav is not None and vocals_wav.exists():
        vocal_env = _compute_vocal_envelope(vocals_wav)

    raw_lines = sorted(timeline_or_aligned.get("lines", []), key=lambda x: x["id"])

    # Pass 1: Choose 0.5-beat line starts guided by vocal onset contrast when available
    snapped_starts: list[float] = []
    min_start = 0.0
    for ln in raw_lines:
        raw_start_b = max(0.0, (float(ln["start"]) - t0) / beat_sec)
        is_instrumental = str(ln.get("original", "")).strip().startswith("♪")
        if vocal_env is not None and not is_instrumental:
            rms, onset, hop_sec = vocal_env
            start_b = _snap_line_start_beat_with_vocals(
                raw_start_b=raw_start_b,
                min_beat=min_start,
                bpm=bpm,
                t0=t0,
                beats_per_bar=beats_per_bar,
                rms=rms,
                onset=onset,
                hop_sec=hop_sec,
            )
        else:
            start_b = max(min_start, round(raw_start_b * 2.0) / 2.0)
        snapped_starts.append(start_b)
        min_start = start_b + 0.5

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
        })

    _apply_repeated_line_consensus(score_lines)

    meta = dict(timeline_or_aligned.get("meta", {}))
    meta.setdefault("content_scale", 1.18)

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
    for ln in score.get("lines", []):
        bar = int(ln["bar"])
        beat = float(ln["beat"])
        dur_beats = max(0.25, float(ln["duration_beats"]))
        total_beats = (bar - 1) * beats_per_bar + (beat - 1.0)

        start = round(t0 + total_beats * beat_sec, 3)
        end = round(t0 + (total_beats + dur_beats) * beat_sec, 3)

        # Keep score line start/end in sync as well
        ln["start"] = start
        ln["end"] = end

        raw_words = ln.get("words") or []
        words_out: list[dict[str, Any]] = []
        if raw_words:
            for w in raw_words:
                if "rel_beat" in w:
                    rel_b = max(0.0, float(w["rel_beat"]))
                    w_dur = max(0.05, float(w.get("duration_beats", 0.25)))
                    ws = round(min(end - 0.04, start + rel_b * beat_sec), 3)
                    we = round(min(end, start + (rel_b + w_dur) * beat_sec), 3)
                else:
                    ws = round(max(start, min(end - 0.04, float(w.get("start", start)))), 3)
                    we = round(max(ws + 0.04, min(end, float(w.get("end", ws + 0.1)))), 3)
                    if words_out and ws < words_out[-1]["end"]:
                        ws = words_out[-1]["end"]
                if we <= ws:
                    we = round(min(end, ws + 0.04), 3)
                words_out.append({
                    "text": w["text"],
                    "start": ws,
                    "end": max(ws, we),
                })
        else:
            tokens = str(ln["original"]).split()
            span = max(0.2, end - start)
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
        })

    # Ensure strictly non-overlapping bounds if user nudged lines tightly
    compiled_lines.sort(key=lambda x: (x["start"], x["id"]))
    for i in range(1, len(compiled_lines)):
        prev = compiled_lines[i - 1]
        cur = compiled_lines[i]
        if cur["start"] < prev["end"] and prev["start"] < cur["start"]:
            prev["end"] = cur["start"]
            for w in prev["words"]:
                w["start"] = min(w["start"], max(prev["start"], prev["end"] - 0.04))
                w["end"] = min(w["end"], prev["end"])

    meta = dict(score.get("meta", {}))
    meta.setdefault("content_scale", 1.18)

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
    for candidate in (timeline_path, aligned_path, raw_llm_path):
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
        upgraded = False
        if "content_scale" not in score.get("meta", {}):
            score["meta"]["content_scale"] = 1.18
            upgraded = True
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

    if bpm is None or t0 is None:
        est_bpm, est_t0 = estimate_beat_grid(work_dir / "analysis.wav", source_tl.get("lines"))
        bpm = bpm or est_bpm
        t0 = est_t0 if t0 is None else t0

    vocals_wav = (work_dir / "vocals.wav") if (work_dir / "vocals.wav").exists() else (work_dir / "analysis.wav")
    score = quantize_timeline_to_score(source_tl, bpm=bpm, t0=t0, vocals_wav=vocals_wav)
    write_json(score_path, score)
    write_json(initial_path, score)
    compiled = compile_score_to_timeline(score)
    write_json(timeline_path, compiled)
    return score


def requantize_score_with_grid(
    work_dir: Path,
    bpm: float,
    t0: float,
    current_score: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-quantize original aligned lyrics onto a new (bpm, t0) grid while preserving annotations."""
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
        t0=round(float(t0), 4),
        time_signature=ts,
        vocals_wav=vocals_wav,
    )

    if base_score:
        new_score.setdefault("meta", {})
        for key in ("style", "content_scale", "title", "artist", "language", "accent_color", "spotify_track_id", "spotify_url", "album", "added_at"):
            if key in base_score.get("meta", {}):
                new_score["meta"][key] = base_score["meta"][key]

        by_id = {int(ln["id"]): ln for ln in base_score.get("lines", []) if "id" in ln}
        for ln in new_score.get("lines", []):
            prev = by_id.get(int(ln["id"]))
            if not prev:
                continue
            for field in ("original", "pronunciation", "translation", "comment", "whole_line"):
                if field in prev:
                    ln[field] = prev[field]

    new_score = enforce_score_integrity(new_score)
    compiled = compile_score_to_timeline(new_score)
    write_json(score_path, new_score)
    write_json(timeline_path, compiled)
    return new_score, compiled


def reset_to_algorithm_score(work_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore score.json from algorithm baseline (score_initial.json) and recompile timeline.json."""
    initial_path = work_dir / "score_initial.json"
    score_path = work_dir / "score.json"
    timeline_path = work_dir / "timeline.json"

    if initial_path.exists():
        score = read_json(initial_path)
        score.setdefault("meta", {}).setdefault("content_scale", 1.18)
        for ln in score.get("lines", []):
            ln.setdefault("whole_line", False)
        compiled = compile_score_to_timeline(score)
        write_json(score_path, score)
        write_json(timeline_path, compiled)
        return score, compiled

    score = ensure_score_file(work_dir, force=True)
    compiled = compile_score_to_timeline(score)
    write_json(timeline_path, compiled)
    return score, compiled
