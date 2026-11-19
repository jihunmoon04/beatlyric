"""Shared work-dir resolution and Remotion + FFmpeg rendering helper."""

from __future__ import annotations

import platform
import shutil
import threading
from collections.abc import Callable
from tempfile import TemporaryDirectory
from pathlib import Path

from common import (
    OUTPUT_DIR,
    RENDERER_DIR,
    WORK_DIR,
    get_ffmpeg_exe,
    read_json,
    run_cmd,
    JobInterrupted,
    slugify,
)


STYLE_TO_COMP = {
    "minimal": "MinimalWave",
    "editorial": "Editorial",
    "classic2": "Classic2",
    "shorts": "ShortsVertical",
}

_RENDER_LOCK = threading.Lock()


def is_render_busy() -> bool:
    """Return True if a Remotion render currently holds the shared renderer/public/track lock."""
    return _RENDER_LOCK.locked()


def resolve_work_dir(target_str: str) -> Path:
    target = Path(target_str)
    if target.is_file():
        return WORK_DIR / slugify(target.stem)
    if target.is_dir():
        return target.resolve()
    return WORK_DIR / slugify(target_str)


def render_work_dir(
    work_dir: Path,
    style: str | None = None,
    output: Path | None = None,
    crf: int = 18,
    gl: str | None = None,
    keep_silent: bool = False,
    clip_start: float | None = None,
    clip_end: float | None = None,
    interrupted: Callable[[], bool] | None = None,
) -> Path:
    """Render work_dir/timeline.json via Remotion and mux master_audio.m4a via FFmpeg."""
    timeline_path = work_dir / "timeline.json"
    spectrum_path = work_dir / "spectrum.json"
    cover_path = work_dir / "cover.jpg"
    bg_path = work_dir / "bg_blur.png"
    master_audio = work_dir / "master_audio.m4a"

    for required in (timeline_path, spectrum_path, cover_path, bg_path, master_audio):
        if not required.exists():
            raise FileNotFoundError(f"Missing required track asset: {required}")

    timeline = read_json(timeline_path)
    chosen_style = style or timeline.get("meta", {}).get("style", "minimal")
    if chosen_style not in STYLE_TO_COMP:
        raise ValueError(f"Unknown style '{chosen_style}'. Choose from: {list(STYLE_TO_COMP)}")

    comp_id = STYLE_TO_COMP[chosen_style]

    has_clip = clip_start is not None or clip_end is not None
    start_sec = 0.0
    end_sec = 0.0
    start_frame = 0
    end_frame = 0
    fps = int(timeline.get("meta", {}).get("fps") or 30)
    if has_clip:
        duration = float(timeline.get("meta", {}).get("duration") or 0.0)
        start_sec = max(0.0, float(clip_start or 0.0))
        if clip_end is not None:
            end_sec = min(duration, float(clip_end)) if duration > 0 else float(clip_end)
        else:
            end_sec = duration
        if end_sec <= start_sec + 0.2:
            raise ValueError(f"Invalid clip range: {start_sec:.2f}s..{end_sec:.2f}s (must exceed 0.2s)")
        start_frame = int(round(start_sec * fps))
        end_frame = max(start_frame, int(round(end_sec * fps)))

    public_track = RENDERER_DIR / "public" / "track"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retained_silent = work_dir / f"silent_{chosen_style}.mp4"
    if output:
        final_mp4 = output.resolve()
    elif has_clip:
        final_mp4 = OUTPUT_DIR / f"{work_dir.name}_{chosen_style}_{int(start_sec)}s-{int(end_sec)}s.mp4"
    else:
        final_mp4 = OUTPUT_DIR / f"{work_dir.name}_{chosen_style}.mp4"

    npx_bin = "npx.cmd" if platform.system() == "Windows" else "npx"
    final_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Retain the old export until the replacement has been fully muxed.
    with _RENDER_LOCK, TemporaryDirectory(prefix=".beatlyric-", dir=final_mp4.parent) as temporary:
        if interrupted and interrupted():
            raise JobInterrupted()
        muxed = Path(temporary) / "video.mp4"
        silent_mp4 = Path(temporary) / "silent.mp4"
        remotion_cmd = [npx_bin, "remotion", "render", "src/index.ts", comp_id,
                        str(silent_mp4), "--crf", str(crf)]
        if has_clip:
            remotion_cmd.extend(["--frames", f"{start_frame}-{end_frame}"])
        if gl:
            remotion_cmd.extend(["--gl", gl])

        public_track.mkdir(parents=True, exist_ok=True)
        shutil.copy2(timeline_path, public_track / "timeline.json")
        shutil.copy2(spectrum_path, public_track / "spectrum.json")
        shutil.copy2(cover_path, public_track / "cover.jpg")
        shutil.copy2(bg_path, public_track / "bg_blur.png")

        print(f"[1/2] Rendering Remotion composition '{comp_id}'")
        run_cmd(remotion_cmd, cwd=RENDERER_DIR, interrupted=interrupted)

        print(f"[2/2] Muxing master audio with FFmpeg -> {final_mp4}")
        mux_cmd = [get_ffmpeg_exe(), "-y", "-i", str(silent_mp4)]
        if has_clip:
            clip_dur = (end_frame - start_frame + 1) / fps
            mux_cmd.extend(["-ss", f"{start_sec:.3f}", "-i", str(master_audio),
                            "-t", f"{clip_dur:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k"])
        else:
            mux_cmd.extend(["-i", str(master_audio), "-c:v", "copy", "-c:a", "copy"])
        mux_cmd.extend(["-shortest", str(muxed)])
        run_cmd(mux_cmd, interrupted=interrupted)
        muxed.replace(final_mp4)
        if keep_silent:
            shutil.move(str(silent_mp4), str(retained_silent))
        else:
            retained_silent.unlink(missing_ok=True)
    return final_mp4
