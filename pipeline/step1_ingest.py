"""Step 1: Ingest audio or video input and prepare normalized assets."""

from __future__ import annotations

import colorsys
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from common import (
    AUDIO_EXTS,
    VIDEO_EXTS,
    get_ffmpeg_exe,
    probe_media,
    run_cmd,
    write_json,
)


def _extract_embedded_cover(input_path: Path, cover_out: Path) -> bool:
    """Try extracting embedded album art using mutagen."""
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return False

    audio = MutagenFile(str(input_path))
    if not audio:
        return False

    # ID3 (MP3)
    if getattr(audio, "tags", None):
        for key in audio.tags.keys():
            if key.startswith("APIC:"):
                img = Image.open(io.BytesIO(audio.tags[key].data)).convert("RGB")
                img.save(cover_out, "JPEG", quality=92)
                return True

    # FLAC
    if hasattr(audio, "pictures") and audio.pictures:
        img = Image.open(io.BytesIO(audio.pictures[0].data)).convert("RGB")
        img.save(cover_out, "JPEG", quality=92)
        return True

    # MP4 / M4A
    if getattr(audio, "tags", None) and "covr" in audio.tags:
        covers = audio.tags["covr"]
        if covers:
            img = Image.open(io.BytesIO(bytes(covers[0]))).convert("RGB")
            img.save(cover_out, "JPEG", quality=92)
            return True

    return False


def _extract_tags(input_path: Path, probe: dict[str, Any]) -> tuple[str, str]:
    """Read title and artist from container metadata, falling back to filename."""
    fmt_tags = probe.get("format", {}).get("tags", {}) or {}
    lowered = {k.lower(): v for k, v in fmt_tags.items()}
    title = lowered.get("title")
    artist = lowered.get("artist") or lowered.get("album_artist")

    if not title or not artist:
        stem = input_path.stem
        if " - " in stem:
            a, t = stem.split(" - ", 1)
            artist = artist or a.strip()
            title = title or t.strip()
        else:
            title = title or stem
            artist = artist or "Unknown Artist"
    return str(title), str(artist)


def _create_fallback_cover(cover_out: Path) -> None:
    """Create a dark studio minimalist square artwork if no cover exists."""
    size = 800
    img = Image.new("RGB", (size, size), (18, 13, 20))
    draw = ImageDraw.Draw(img)
    for r in range(360, 0, -4):
        alpha = int(45 * (1 - r / 360))
        color = (22 + alpha * 2, 14 + alpha // 2, 28 + alpha)
        draw.ellipse((400 - r, 360 - r, 400 + r, 360 + r), fill=color)
    draw.ellipse((280, 240, 520, 480), fill=(242, 224, 226))
    draw.rectangle((0, 540, size, size), fill=(10, 8, 12))
    img.save(cover_out, "JPEG", quality=92)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Compute WCAG 2.x relative luminance for an sRGB (0..255) triple."""
    channels = []
    for c in rgb:
        s = c / 255.0
        channels.append(s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    l1 = _relative_luminance(rgb1)
    l2 = _relative_luminance(rgb2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _extract_dominant_accent(cover_path: Path) -> str:
    """Extract dominant cover hue via 24-bin HSL/HSV histogram with >= 5.5 WCAG contrast vs #0c0a0f."""
    img = Image.open(cover_path).convert("RGB").resize((64, 64))
    pixels = list(img.getdata())

    bins: list[list[tuple[float, float, float]]] = [[] for _ in range(24)]
    bin_weights = [0.0] * 24
    chromatic_count = 0

    for r, g, b in pixels:
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        h, l_val, s_hls = colorsys.rgb_to_hls(rf, gf, bf)
        _, s_hsv, v_hsv = colorsys.rgb_to_hsv(rf, gf, bf)
        if s_hsv < 0.22 or v_hsv < 0.20 or l_val < 0.12 or l_val > 0.92:
            continue
        chromatic_count += 1
        weight = s_hsv * (0.4 + 0.6 * v_hsv)
        bin_idx = int(h * 24.0) % 24
        bins[bin_idx].append((h, s_hls, weight))
        bin_weights[bin_idx] += weight

    if chromatic_count < 12:
        return "#d88392"

    best_bin = max(range(24), key=lambda idx: bin_weights[idx])
    selected: list[tuple[float, float, float]] = []
    for offset in (-1, 0, 1):
        selected.extend(bins[(best_bin + offset) % 24])

    if not selected:
        return "#d88392"

    total_w = sum(w for _, _, w in selected) or 1.0
    sin_sum = sum(w * math.sin(2.0 * math.pi * h) for h, _, w in selected)
    cos_sum = sum(w * math.cos(2.0 * math.pi * h) for h, _, w in selected)
    mean_h = (math.atan2(sin_sum, cos_sum) / (2.0 * math.pi)) % 1.0
    mean_s = sum(w * s for _, s, w in selected) / total_w
    clamped_s = max(0.42, min(0.78, mean_s))

    bg_rgb = (12, 10, 15)
    lightness = 0.66
    out_rgb = (216, 131, 146)
    while True:
        rf, gf, bf = colorsys.hls_to_rgb(mean_h, lightness, clamped_s)
        out_rgb = (
            max(0, min(255, int(round(rf * 255)))),
            max(0, min(255, int(round(gf * 255)))),
            max(0, min(255, int(round(bf * 255)))),
        )
        if _contrast_ratio(out_rgb, bg_rgb) >= 5.5 or lightness >= 0.82:
            break
        lightness = round(lightness + 0.02, 4)

    return f"#{out_rgb[0]:02x}{out_rgb[1]:02x}{out_rgb[2]:02x}"


extract_dominant_accent = _extract_dominant_accent


def _build_preblurred_background(cover_path: Path, bg_out: Path) -> None:
    """Pre-render a 1920x1080 dark blurred background so Linux headless rendering stays fast."""
    img = Image.open(cover_path).convert("RGB")
    target_w, target_h = 1920, 1080
    # Center-crop to 16:9
    w, h = img.size
    target_ratio = target_w / target_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    bg = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=80))
    bg = ImageEnhance.Color(bg).enhance(1.30)

    # Even dark studio wash across the entire frame (guarantees lyric contrast without dark edge borders)
    studio_dark = Image.new("RGB", (target_w, target_h), (12, 10, 15))
    combined = Image.blend(bg, studio_dark, alpha=0.72)
    combined.save(bg_out, "PNG", optimize=True)


def run_ingest(
    input_path: Path,
    work_dir: Path,
    fps: int = 30,
    style: str = "minimal",
    language: str = "en",
    title_override: str | None = None,
    artist_override: str | None = None,
    cover_override_path: Path | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest audio/video file into standardized work directory assets."""
    ext = input_path.suffix.lower()
    if ext not in AUDIO_EXTS and ext not in VIDEO_EXTS:
        raise ValueError(f"Unsupported input format: {ext}")

    work_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_media(input_path)
    duration = float(probe["format"]["duration"])
    title, artist = _extract_tags(input_path, probe)
    if title_override:
        title = title_override
    if artist_override:
        artist = artist_override

    master_audio = work_dir / "master_audio.m4a"
    analysis_wav = work_dir / "analysis.wav"
    cover_jpg = work_dir / "cover.jpg"
    bg_blur_png = work_dir / "bg_blur.png"

    ffmpeg_bin = get_ffmpeg_exe()

    # 1. Extract high-quality master audio (AAC 320k for universal MP4 muxing)
    run_cmd([
        ffmpeg_bin, "-y", "-i", str(input_path),
        "-vn", "-c:a", "aac", "-b:a", "320k",
        str(master_audio),
    ])

    # 2. Extract 16-bit 44.1kHz mono WAV for FFT spectrum & forced alignment
    run_cmd([
        ffmpeg_bin, "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le",
        str(analysis_wav),
    ])

    # 3. Extract cover art (explicit override, audio tag, mid-video frame, or procedural fallback)
    has_cover = False
    if cover_override_path and cover_override_path.exists():
        img = Image.open(cover_override_path).convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img.crop((left, top, left + side, top + side)).resize((800, 800), Image.Resampling.LANCZOS).save(
            cover_jpg, "JPEG", quality=92
        )
        has_cover = True
    elif ext in AUDIO_EXTS:
        has_cover = _extract_embedded_cover(input_path, cover_jpg)
    else:
        has_cover = False
        seek_sec = max(1.0, duration * 0.25)
        proc = run_cmd([
            ffmpeg_bin, "-y", "-ss", f"{seek_sec:.2f}",
            "-i", str(input_path),
            "-frames:v", "1", "-q:v", "2",
            str(cover_jpg),
        ], check=False)
        if proc.returncode == 0 and cover_jpg.exists():
            # Center-crop video frame to square cover
            frame_img = Image.open(cover_jpg).convert("RGB")
            fw, fh = frame_img.size
            side = min(fw, fh)
            left = (fw - side) // 2
            top = (fh - side) // 2
            frame_img.crop((left, top, left + side, top + side)).resize((800, 800)).save(cover_jpg, "JPEG", quality=92)
            has_cover = True

    if not has_cover:
        _create_fallback_cover(cover_jpg)

    accent_color = _extract_dominant_accent(cover_jpg)
    _build_preblurred_background(cover_jpg, bg_blur_png)

    meta = {
        "title": title,
        "artist": artist,
        "language": language,
        "duration": round(duration, 3),
        "fps": fps,
        "style": style,
        "accent_color": accent_color,
        "source_type": "video" if ext in VIDEO_EXTS else "audio",
        "source_path": str(input_path.resolve()),
    }
    if extra_meta:
        meta.update(extra_meta)
    write_json(work_dir / "meta.json", meta)
    return meta
