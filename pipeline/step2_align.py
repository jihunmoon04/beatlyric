"""Step 2: Lyrics acquisition, Demucs vocal isolation, and forced alignment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from common import write_json


LRC_TIME_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
JA_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

ISO1_TO_ISO3 = {
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "zh": "chi",
    "pt": "por",
}


def parse_lrc(lrc_text: str, total_duration: float) -> list[dict[str, Any]]:
    """Parse [mm:ss.xx] LRC lines, preserving empty timestamp lines as explicit end markers."""
    events: list[tuple[float, str]] = []
    for raw_line in lrc_text.splitlines():
        matches = list(LRC_TIME_RE.finditer(raw_line))
        if not matches:
            continue
        text = LRC_TIME_RE.sub("", raw_line).strip()
        for m in matches:
            mins = int(m.group(1))
            secs = float(m.group(2))
            events.append((mins * 60.0 + secs, text))

    events.sort(key=lambda x: x[0])

    segments: list[dict[str, Any]] = []
    for idx, (ts, text) in enumerate(events):
        # Empty LRC marker acts as an explicit end-of-line boundary for the preceding lyric
        if not text:
            if segments:
                prev = segments[-1]
                if ts > prev["start"] + 0.25 and ts < prev["end"]:
                    prev["end"] = round(ts, 3)
                    prev["words"] = _interpolate_words(prev["original"], prev["start"], prev["end"])
            continue

        # Find the next event timestamp (even if empty) as the upper bound
        next_ts = events[idx + 1][0] if idx + 1 < len(events) else total_duration
        end = max(ts + 0.5, min(next_ts, ts + 10.0))
        segments.append({
            "id": len(segments),
            "start": round(ts, 3),
            "end": round(end, 3),
            "original": text,
            "words": _interpolate_words(text, ts, end),
        })
    return segments


def _interpolate_words(text: str, start: float, end: float) -> list[dict[str, Any]]:
    """Fallback proportional word timing within a line."""
    tokens = text.split()
    if not tokens:
        return []
    span = max(0.2, end - start)
    total_chars = sum(max(1, len(t)) for t in tokens)
    cursor = start
    words: list[dict[str, Any]] = []
    for tok in tokens:
        weight = max(1, len(tok)) / total_chars
        w_end = cursor + span * weight
        words.append({
            "text": tok,
            "start": round(cursor, 3),
            "end": round(w_end, 3),
        })
        cursor = w_end
    return words


def _fetch_lyrics_syncedlyrics(artist: str, title: str) -> str | None:
    """Look up synced or plain lyrics via syncedlyrics (LRCLIB, etc.)."""
    try:
        import syncedlyrics
    except ImportError:
        return None
    query = f"{artist} {title}".strip() if artist != "Unknown Artist" else title
    return syncedlyrics.search(query)


def _detect_chunk_language(lines: list[dict[str, Any]], default_lang: str) -> str:
    """Detect whether a chunk is predominantly Japanese or uses the track's default language."""
    joined = " ".join(seg["original"] for seg in lines)
    non_space = re.sub(r"\s+", "", joined)
    if not non_space:
        return default_lang
    ja_count = len(JA_CHAR_RE.findall(non_space))
    if ja_count / len(non_space) >= 0.25:
        return "ja"
    return default_lang


def _separate_vocals_demucs(analysis_wav: Path, vocals_out: Path) -> Path:
    """Isolate clean vocals using Demucs (cached to vocals.wav)."""
    if vocals_out.exists():
        return vocals_out

    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    audio_np, sr = sf.read(str(analysis_wav), dtype="float32")
    if audio_np.ndim == 1:
        # htdemucs expects stereo (2, samples)
        wav = torch.from_numpy(audio_np).unsqueeze(0).repeat(2, 1)
    else:
        wav = torch.from_numpy(audio_np.T)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

    if sr != model.samplerate:
        wav = AF.resample(wav, sr, model.samplerate)

    ref_mean = wav.mean()
    ref_std = wav.std() + 1e-8
    norm_wav = (wav - ref_mean) / ref_std

    with torch.no_grad():
        sources = apply_model(
            model,
            norm_wav.unsqueeze(0).to(device),
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
        )[0]

    vocals_idx = model.sources.index("vocals")
    vocals = (sources[vocals_idx].cpu() * ref_std + ref_mean).mean(dim=0)
    # Resample to 16 kHz mono for Whisper/stable-ts
    vocals_16k = AF.resample(vocals.unsqueeze(0), model.samplerate, 16000).squeeze(0).numpy()
    sf.write(str(vocals_out), vocals_16k, 16000)
    return vocals_out


def _snap_to_vocal_energy(
    segments: list[dict[str, Any]],
    vocals_16k: np.ndarray,
    sr: int = 16000,
) -> list[dict[str, Any]]:
    """Trim trailing/leading silence on each line using isolated vocal RMS energy."""
    hop = int(sr * 0.02)  # 20ms frames
    n_frames = max(1, len(vocals_16k) // hop)
    trimmed = vocals_16k[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    peak = float(np.percentile(rms, 98)) or 1.0
    threshold = max(peak * 0.045, 0.008)

    for idx, seg in enumerate(segments):
        if seg["original"].startswith("♪"):
            continue
        f_start = max(0, int(seg["start"] * sr / hop))
        f_end = min(n_frames - 1, int(seg["end"] * sr / hop))
        if f_end <= f_start + 3:
            continue

        active_frames = np.where(rms[f_start : f_end + 1] >= threshold)[0]
        if len(active_frames) == 0:
            continue

        # Stop at the first silence gap > 1.0s (50 frames at 20ms hop) so trailing alignment never jumps across instrumental breaks
        last_active = active_frames[-1]
        for k in range(len(active_frames) - 1):
            if active_frames[k + 1] - active_frames[k] > 50:
                last_active = active_frames[k]
                break

        actual_first = (f_start + int(active_frames[0])) * hop / sr
        actual_last = (f_start + int(last_active) + 1) * hop / sr

        # Gently trim dead silence at start/end while keeping a small natural breath margin
        new_start = max(seg["start"], actual_first - 0.06)
        new_end = min(seg["end"], actual_last + 0.14)

        if new_end - new_start >= 0.35:
            seg["start"] = round(new_start, 3)
            seg["end"] = round(new_end, 3)
            # Clamp word timings inside the trimmed line bounds
            for w in seg.get("words", []):
                w["start"] = round(max(seg["start"], min(seg["end"] - 0.05, w["start"])), 3)
                w["end"] = round(max(w["start"] + 0.04, min(seg["end"], w["end"])), 3)

        # Ensure strictly non-overlapping bounds with previous line
        if idx > 0 and not segments[idx - 1]["original"].startswith("♪"):
            prev = segments[idx - 1]
            if seg["start"] < prev["end"]:
                if prev["end"] < seg["end"] - 0.25:
                    seg["start"] = prev["end"]
                else:
                    mid = round((prev["end"] + seg["start"]) / 2.0, 3)
                    mid = max(prev["start"] + 0.2, min(seg["end"] - 0.2, mid))
                    prev["end"] = mid
                    seg["start"] = mid

    return segments


def _group_chunks(
    singing_lines: list[dict[str, Any]],
    default_lang: str,
) -> list[list[dict[str, Any]]]:
    """Group consecutive lines into local windows split by gaps, max span, or language shifts."""
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_lang: str | None = None

    for seg in singing_lines:
        seg_lang = _detect_chunk_language([seg], default_lang)
        if not cur:
            cur = [seg]
            cur_lang = seg_lang
            continue

        gap = seg["start"] - cur[-1]["end"]
        span = seg["end"] - cur[0]["start"]
        lang_changed = seg_lang != cur_lang and len(cur) >= 2
        if gap > 1.8 or span > 22.0 or len(cur) >= 6 or lang_changed:
            chunks.append(cur)
            cur = [seg]
            cur_lang = seg_lang
        else:
            cur.append(seg)

    if cur:
        chunks.append(cur)
    return chunks


def _romanize_token(token: str, default_iso3: str, kakasi_conv: Any = None) -> str:
    """Convert a single display token into space-separated uroman characters for CTC alignment."""
    import ctc_forced_aligner as cfa

    cleaned = token.replace("&", " and ").replace("7-Eleven", "seven eleven").replace("4-Boyz", "four boyz")
    is_ja = bool(JA_CHAR_RE.search(cleaned))
    if is_ja and kakasi_conv is not None:
        # Convert kanji -> hiragana first so uroman reads Japanese instead of Chinese pinyin
        cleaned = "".join(item["hira"] for item in kakasi_conv.convert(cleaned))
        lang_code = "jpn"
    else:
        lang_code = "jpn" if is_ja else default_iso3

    norm = cfa.text_normalize(cleaned.strip(), lang_code)
    if not norm:
        return ""
    urom_list = cfa.get_uroman_tokens([norm], lang_code)
    return urom_list[0].strip() if urom_list else ""


def _align_chunks_with_ctc(
    vocals_wav: Path,
    base_segments: list[dict[str, Any]],
    default_lang: str,
    lrc_anchor_tolerance: float = 0.9,
) -> list[dict[str, Any]]:
    """Run MMS-300M CTC forced alignment per LRC-anchored chunk and clamp outliers back to .lrc."""
    import ctc_forced_aligner as cfa
    import soundfile as sf
    import torch

    try:
        import pykakasi

        kakasi_conv = pykakasi.kakasi()
    except ImportError:
        kakasi_conv = None

    vocals_16k, sr = sf.read(str(vocals_wav), dtype="float32")
    if vocals_16k.ndim > 1:
        vocals_16k = vocals_16k.mean(axis=1)
    if sr != 16000:
        import torchaudio.functional as AF

        vocals_16k = AF.resample(torch.from_numpy(vocals_16k).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        sr = 16000

    total_dur = len(vocals_16k) / sr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, tokenizer = cfa.load_alignment_model(device, dtype=dtype)
    default_iso3 = ISO1_TO_ISO3.get(default_lang, "eng")

    intro_lines = [s for s in base_segments if s["original"].startswith("♪")]
    singing_lines = [s for s in base_segments if not s["original"].startswith("♪")]
    chunks = _group_chunks(singing_lines, default_lang)

    aligned_singing: list[dict[str, Any]] = []

    for chunk in chunks:
        win_start = max(0.0, chunk[0]["start"] - 1.2)
        win_end = min(total_dur, chunk[-1]["end"] + 1.8)
        s_idx = int(win_start * sr)
        e_idx = int(win_end * sr)
        chunk_tensor = torch.from_numpy(vocals_16k[s_idx:e_idx]).to(model.device, dtype=model.dtype)
        emissions, stride = cfa.generate_emissions(model, chunk_tensor, batch_size=4)
        stride_sec = float(stride) / 1000.0
        n_em_frames = emissions.shape[0]

        # Align each .lrc line inside its own padded slice of `emissions` so repeated lines never steal neighbors
        line_pad = 0.85
        for orig_seg in chunk:
            tokens_starred: list[str] = ["<star>"]
            text_starred: list[str] = ["<star>"]
            word_entries: list[tuple[str, bool]] = []

            for w_tok in orig_seg["original"].split():
                urom = _romanize_token(w_tok, default_iso3, kakasi_conv)
                if urom:
                    tokens_starred.extend([urom, "<star>"])
                    text_starred.extend([w_tok, "<star>"])
                    word_entries.append((w_tok, True))
                else:
                    word_entries.append((w_tok, False))

            if not any(flag for _, flag in word_entries):
                aligned_singing.append(orig_seg)
                continue

            sub_start = max(win_start, orig_seg["start"] - line_pad)
            sub_end = min(win_end, orig_seg["end"] + line_pad)
            em_s = max(0, int((sub_start - win_start) / stride_sec))
            em_e = min(n_em_frames, int((sub_end - win_start) / stride_sec))
            if em_e <= em_s + 5:
                aligned_singing.append(orig_seg)
                continue

            sub_emissions = emissions[em_s:em_e]
            slice_origin = win_start + em_s * stride_sec

            segments_ctc, scores, blank = cfa.get_alignments(sub_emissions, tokens_starred, tokenizer)
            spans = cfa.get_spans(tokens_starred, segments_ctc, blank)
            ctc_words = cfa.postprocess_results(text_starred, spans, stride, scores)

            words: list[dict[str, Any]] = []
            line_scores: list[float] = []
            ctc_cursor = 0

            for w_tok, is_aligned in word_entries:
                if is_aligned and ctc_cursor < len(ctc_words):
                    cw = ctc_words[ctc_cursor]
                    ctc_cursor += 1
                    ws = round(slice_origin + float(cw["start"]), 3)
                    we = round(slice_origin + float(cw["end"]), 3)
                    if we <= ws:
                        we = round(ws + 0.06, 3)
                    words.append({"text": w_tok, "start": ws, "end": we})
                    line_scores.append(float(cw["score"]))
                else:
                    prev_end = words[-1]["end"] if words else orig_seg["start"]
                    words.append({"text": w_tok, "start": prev_end, "end": round(prev_end + 0.12, 3)})

            if not words:
                aligned_singing.append(orig_seg)
                continue

            ctc_start = words[0]["start"]
            ctc_end = words[-1]["end"]
            avg_score = sum(line_scores) / len(line_scores) if line_scores else -9.9

            if avg_score < -4.5 or ctc_end <= ctc_start:
                aligned_singing.append(orig_seg)
                continue

            aligned_singing.append({
                "id": orig_seg["id"],
                "start": ctc_start,
                "end": ctc_end,
                "original": orig_seg["original"],
                "words": words,
            })

    combined = intro_lines + aligned_singing
    combined.sort(key=lambda s: s["id"])
    return _snap_to_vocal_energy(combined, vocals_16k, sr=sr)


def _align_chunks_with_stable_ts(
    vocals_wav: Path,
    base_segments: list[dict[str, Any]],
    default_lang: str,
    model_size: str = "medium",
) -> list[dict[str, Any]]:
    """Run windowed forced alignment on isolated vocals so bilingual/rap songs stay locked in time."""
    import soundfile as sf
    import stable_whisper
    import torch

    vocals_16k, sr = sf.read(str(vocals_wav), dtype="float32")
    if vocals_16k.ndim > 1:
        vocals_16k = vocals_16k.mean(axis=1)
    if sr != 16000:
        import torchaudio.functional as AF
        vocals_16k = AF.resample(torch.from_numpy(vocals_16k).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        sr = 16000

    total_dur = len(vocals_16k) / sr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = stable_whisper.load_model(model_size, device=device)

    intro_lines = [s for s in base_segments if s["original"].startswith("♪")]
    singing_lines = [s for s in base_segments if not s["original"].startswith("♪")]
    chunks = _group_chunks(singing_lines, default_lang)

    aligned_singing: list[dict[str, Any]] = []

    for chunk in chunks:
        win_start = max(0.0, chunk[0]["start"] - 2.5)
        win_end = min(total_dur, chunk[-1]["end"] + 4.5)
        s_idx = int(win_start * sr)
        e_idx = int(win_end * sr)
        chunk_audio = vocals_16k[s_idx:e_idx]
        chunk_lang = _detect_chunk_language(chunk, default_lang)
        chunk_text = "\n".join(seg["original"] for seg in chunk)

        result = model.align(
            chunk_audio,
            chunk_text,
            language=chunk_lang,
            original_split=True,
            max_word_dur=2.2,
            nonspeech_skip=2.0,
            fast_mode=False,
            vad=False,
        )

        if result and len(result.segments) == len(chunk):
            for orig_seg, aligned_seg in zip(chunk, result.segments):
                a_start = round(win_start + float(aligned_seg.start), 3)
                a_end = round(win_start + float(aligned_seg.end), 3)
                # Guard against wild outliers far outside the local chunk window
                if abs(a_start - orig_seg["start"]) > 6.5 or a_end <= a_start:
                    aligned_singing.append(orig_seg)
                    continue
                words = [
                    {
                        "text": w.word.strip(),
                        "start": round(win_start + float(w.start), 3),
                        "end": round(win_start + float(w.end), 3),
                    }
                    for w in (aligned_seg.words or [])
                    if w.word.strip()
                ]
                aligned_singing.append({
                    "id": orig_seg["id"],
                    "start": a_start,
                    "end": a_end,
                    "original": orig_seg["original"],
                    "words": words or _interpolate_words(orig_seg["original"], a_start, a_end),
                })
        else:
            aligned_singing.extend(chunk)

    combined = intro_lines + aligned_singing
    combined.sort(key=lambda s: s["id"])
    return _snap_to_vocal_energy(combined, vocals_16k, sr=sr)


def _run_stable_ts_full(
    vocals_wav: Path,
    plain_lyrics: str | None,
    language: str,
    model_size: str = "medium",
) -> list[dict[str, Any]]:
    """Full-track alignment or ASR fallback when no LRC timestamps exist."""
    import soundfile as sf
    import stable_whisper
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = stable_whisper.load_model(model_size, device=device)

    if plain_lyrics:
        result = model.align(
            str(vocals_wav),
            plain_lyrics,
            language=language,
            original_split=True,
            max_word_dur=2.5,
        )
    else:
        result = model.transcribe(
            str(vocals_wav),
            language=language,
            word_timestamps=True,
            vad=True,
        )

    segments: list[dict[str, Any]] = []
    for seg in result.segments:
        text = seg.text.strip()
        if not text:
            continue
        words = [
            {
                "text": w.word.strip(),
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
            }
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append({
            "id": len(segments),
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "original": text,
            "words": words or _interpolate_words(text, float(seg.start), float(seg.end)),
        })

    vocals_16k, sr = sf.read(str(vocals_wav), dtype="float32")
    if vocals_16k.ndim > 1:
        vocals_16k = vocals_16k.mean(axis=1)
    return _snap_to_vocal_energy(segments, vocals_16k, sr=sr)


def run_align(
    work_dir: Path,
    meta: dict[str, Any],
    lyrics_file: Path | None = None,
    use_demucs: bool = True,
    whisper_model: str = "medium",
    force_neural_align: bool = False,
    aligner: str = "ctc",
) -> dict[str, Any]:
    """Produce timed lyric segments and LLM-ready raw_lyrics_for_llm.json."""
    analysis_wav = work_dir / "analysis.wav"
    vocals_wav = work_dir / "vocals.wav"
    duration = float(meta["duration"])
    language = meta.get("language", "en")

    raw_text: str | None = None
    source_label = "unknown"

    if lyrics_file and lyrics_file.exists():
        raw_text = lyrics_file.read_text(encoding="utf-8")
        source_label = f"file:{lyrics_file.name}"
    else:
        fetched = _fetch_lyrics_syncedlyrics(meta["artist"], meta["title"])
        if fetched:
            raw_text = fetched
            source_label = "syncedlyrics"

    has_lrc_timestamps = bool(raw_text and LRC_TIME_RE.search(raw_text))
    plain_text = LRC_TIME_RE.sub("", raw_text).strip() if raw_text else None

    segments: list[dict[str, Any]]
    alignment_mode: str

    if has_lrc_timestamps and not force_neural_align:
        segments = parse_lrc(raw_text or "", duration)
        alignment_mode = "lrc_interpolated_words"
    else:
        try:
            vocal_path = (
                _separate_vocals_demucs(analysis_wav, vocals_wav)
                if use_demucs
                else analysis_wav
            )
            if has_lrc_timestamps:
                base_segments = parse_lrc(raw_text or "", duration)
                if aligner == "ctc":
                    try:
                        segments = _align_chunks_with_ctc(
                            vocals_wav=vocal_path,
                            base_segments=base_segments,
                            default_lang=language,
                        )
                        alignment_mode = "demucs_ctc_lrc_fused"
                    except ImportError:
                        segments = _align_chunks_with_stable_ts(
                            vocals_wav=vocal_path,
                            base_segments=base_segments,
                            default_lang=language,
                            model_size=whisper_model,
                        )
                        alignment_mode = "demucs_windowed_stable_ts"
                else:
                    segments = _align_chunks_with_stable_ts(
                        vocals_wav=vocal_path,
                        base_segments=base_segments,
                        default_lang=language,
                        model_size=whisper_model,
                    )
                    alignment_mode = "demucs_windowed_stable_ts"
            else:
                segments = _run_stable_ts_full(
                    vocals_wav=vocal_path,
                    plain_lyrics=plain_text,
                    language=language,
                    model_size=whisper_model,
                )
                alignment_mode = "demucs_stable_ts_align" if plain_text else "demucs_stable_ts_transcribe"
        except ImportError as exc:
            if has_lrc_timestamps:
                segments = parse_lrc(raw_text or "", duration)
                alignment_mode = "lrc_interpolated_words"
            elif plain_text:
                lines = [ln.strip() for ln in plain_text.splitlines() if ln.strip()]
                step = duration / max(1, len(lines))
                segments = []
                for idx, ln in enumerate(lines):
                    s = idx * step
                    e = (idx + 1) * step
                    segments.append({
                        "id": idx,
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "original": ln,
                        "words": _interpolate_words(ln, s, e),
                    })
                alignment_mode = "uniform_text_fallback"
            else:
                raise RuntimeError(
                    "No LRC lyrics found and stable-ts/demucs are not installed. "
                    "Provide --lyrics <file> or install alignment dependencies."
                ) from exc

    aligned_payload = {
        "meta": meta,
        "lyrics_source": source_label,
        "alignment_mode": alignment_mode,
        "lines": segments,
    }
    write_json(work_dir / "aligned_lyrics.json", aligned_payload)

    llm_payload = {
        "meta": meta,
        "instructions": (
            "Fill each line with natural Korean 'translation', optional Hangul 'pronunciation', "
            "and selective language-learning 'comment' ({tag, phrase, meaning, tip} or null). "
            "Save the completed result as timeline.json in this same directory."
        ),
        "lines": [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["original"],
                "words": seg["words"],
                "pronunciation": "",
                "translation": "",
                "comment": None,
            }
            for seg in segments
        ],
    }
    write_json(work_dir / "raw_lyrics_for_llm.json", llm_payload)
    return llm_payload
