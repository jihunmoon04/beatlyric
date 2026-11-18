"""Resolve local audio files or download matching audio via yt-dlp + cover artwork."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests

from common import AUDIO_EXTS, ROOT_DIR, get_ffmpeg_exe, probe_media, slugify


INPUT_DIR = ROOT_DIR / "input"
_NON_WORD_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_NOISE_PENALTY_WORDS = ("cover", "nightcore", "slowed", "reverb", "8d audio", "1 hour", "karaoke", "instrumental", "live")


def _norm_tokens(text: str) -> set[str]:
    cleaned = _NON_WORD_RE.sub(" ", text.lower())
    return {tok for tok in cleaned.split() if len(tok) >= 2}


def find_local_audio_match(title: str, artist: str, duration_ms: int | None = None) -> Path | None:
    """Check input/ for an existing audio file matching title/artist and duration."""
    if not INPUT_DIR.exists():
        return None
    target_tokens = _norm_tokens(f"{artist} {title}")
    title_tokens = _norm_tokens(title)
    if not title_tokens:
        return None
    target_sec = (duration_ms / 1000.0) if duration_ms else None

    best_path: Path | None = None
    best_score = 0.0

    for cand in INPUT_DIR.iterdir():
        if not cand.is_file() or cand.suffix.lower() not in AUDIO_EXTS:
            continue
        stem_tokens = _norm_tokens(cand.stem)
        title_overlap = len(title_tokens & stem_tokens) / max(1, len(title_tokens))
        if title_overlap < 0.7:
            # Also inspect embedded tags if filename differs
            try:
                probe = probe_media(cand)
                tags = probe.get("format", {}).get("tags", {}) or {}
                tag_text = f"{tags.get('artist', '')} {tags.get('title', '')}"
                tag_tokens = _norm_tokens(tag_text)
                title_overlap = len(title_tokens & tag_tokens) / max(1, len(title_tokens))
                full_overlap = len(target_tokens & (stem_tokens | tag_tokens)) / max(1, len(target_tokens))
                dur = float(probe.get("format", {}).get("duration") or 0.0)
            except Exception:
                continue
        else:
            full_overlap = len(target_tokens & stem_tokens) / max(1, len(target_tokens))
            try:
                probe = probe_media(cand)
                dur = float(probe.get("format", {}).get("duration") or 0.0)
            except Exception:
                dur = 0.0

        if title_overlap < 0.7:
            continue
        if target_sec and dur > 0 and abs(dur - target_sec) > 8.0:
            continue

        score = title_overlap * 2.0 + full_overlap
        if score > best_score:
            best_score = score
            best_path = cand

    return best_path


def _score_yt_candidate(entry: dict[str, Any], title: str, artist: str, target_sec: float | None) -> float:
    cand_title = str(entry.get("title") or "").lower()
    uploader = str(entry.get("uploader") or entry.get("channel") or "").lower()
    dur = float(entry.get("duration") or 0.0)

    if dur <= 15 or dur >= 1200:
        return -1e9

    score = 0.0
    if target_sec and dur > 0:
        diff = abs(dur - target_sec)
        if diff <= 3.0:
            score += 40.0
        elif diff <= 8.0:
            score += 20.0
        elif diff <= 20.0:
            score += 5.0
        else:
            score -= diff * 1.5

    req_lower = f"{artist} {title}".lower()
    for bad in _NOISE_PENALTY_WORDS:
        if bad in cand_title and bad not in req_lower:
            score -= 25.0

    if "topic" in uploader or "official audio" in cand_title or "provided to youtube" in str(entry.get("description") or "").lower():
        score += 18.0
    if "lyrics" in cand_title:
        score += 4.0

    title_tokens = _norm_tokens(title)
    cand_tokens = _norm_tokens(f"{cand_title} {uploader}")
    if title_tokens:
        score += 15.0 * (len(title_tokens & cand_tokens) / len(title_tokens))

    primary_artist = artist.split(",")[0].strip()
    artist_tokens = _norm_tokens(primary_artist)
    if artist_tokens:
        score += 10.0 * (len(artist_tokens & cand_tokens) / len(artist_tokens))

    return score


def download_audio_from_youtube(
    title: str,
    artist: str,
    duration_ms: int | None = None,
    slug: str | None = None,
    url: str | None = None,
) -> Path:
    """Download matching audio via yt-dlp into input/<slug>.<ext>."""
    import yt_dlp

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_slug = slug or slugify(f"{artist}-{title}")
    target_sec = (duration_ms / 1000.0) if duration_ms else None
    ffmpeg_dir = str(Path(get_ffmpeg_exe()).parent)

    chosen_url = url
    if not chosen_url:
        primary_artist = artist.split(",")[0].strip() if artist and artist != "Unknown Artist" else ""
        query = f"ytsearch5:{primary_artist} {title} audio".strip()
        search_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "noplaylist": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(search_opts) as ydl:
            info = ydl.extract_info(query, download=False)
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise RuntimeError(f"yt-dlp found no matches for: {artist} - {title}")
        best = max(entries, key=lambda e: _score_yt_candidate(e, title, artist, target_sec))
        chosen_url = best.get("webpage_url") or best.get("url")
        if not chosen_url:
            raise RuntimeError(f"yt-dlp candidate missing URL for: {artist} - {title}")

    out_tmpl = str(INPUT_DIR / f"{safe_slug}.%(ext)s")
    dl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "ffmpeg_location": ffmpeg_dir,
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        dl_info = ydl.extract_info(chosen_url, download=True)
        downloaded = Path(ydl.prepare_filename(dl_info))

    if downloaded.exists() and downloaded.suffix.lower() in AUDIO_EXTS:
        return downloaded

    # Fallback scan for any matching file created with safe_slug
    for cand in INPUT_DIR.glob(f"{safe_slug}.*"):
        if cand.suffix.lower() in AUDIO_EXTS:
            return cand
    raise RuntimeError(f"Downloaded audio file not found for {safe_slug}")


def download_cover_image(cover_url: str, dest_path: Path) -> Path | None:
    """Download remote Spotify album art locally."""
    if not cover_url:
        return None
    try:
        res = requests.get(cover_url, timeout=20)
        if not res.ok:
            return None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(res.content)
        return dest_path
    except Exception:
        return None


def resolve_audio_for_track(
    title: str,
    artist: str,
    duration_ms: int | None,
    slug: str,
    cover_url: str | None = None,
    work_dir: Path | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Resolve audio file (local input/ first, else yt-dlp) and download cover art if provided."""
    local_match = None if source_url else find_local_audio_match(title, artist, duration_ms)
    if local_match:
        audio_path = local_match
        source_kind = "local_input"
    else:
        audio_path = download_audio_from_youtube(
            title=title,
            artist=artist,
            duration_ms=duration_ms,
            slug=slug,
            url=source_url,
        )
        source_kind = "yt_dlp"

    cover_path: Path | None = None
    if cover_url and work_dir:
        cover_path = download_cover_image(cover_url, work_dir / "cover_spotify.jpg")

    return {
        "audio_path": audio_path,
        "source_kind": source_kind,
        "cover_path": cover_path,
    }
