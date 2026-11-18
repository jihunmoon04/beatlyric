"""Step 4: Automated beat grid estimation, 0.5-beat score quantization, and timeline compilation."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from common import read_json, write_json


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
    base_b = round(raw_start_b * 2.0) / 2.0
    candidates = [
        base_b + off
        for off in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
        if base_b + off >= min_beat
    ]
    if not candidates:
        return max(min_beat, base_b)

    beat_sec = 60.0 / bpm
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
        prox_penalty = 0.38 * abs(cand_b - raw_start_b)
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
            "whole_line": bool(ln.get("whole_line", False)),
            "pronunciation": ln.get("pronunciation", ""),
            "translation": ln.get("translation", ""),
            "comment": ln.get("comment"),
        })

    meta = dict(timeline_or_aligned.get("meta", {}))
    meta.setdefault("content_scale", 1.18)

    return {
        "grid": {
            "bpm": bpm,
            "time_signature": ts,
            "t0": t0,
        },
        "meta": meta,
        "lines": score_lines,
    }


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
