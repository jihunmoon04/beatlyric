"""Shared work-dir resolution and Remotion + FFmpeg rendering helper."""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

from common import (
    OUTPUT_DIR,
    RENDERER_DIR,
    WORK_DIR,
    get_ffmpeg_exe,
    read_json,
    run_cmd,
    slugify,
)


STYLE_TO_COMP = {
    "minimal": "MinimalWave",
    "editorial": "Editorial",
    "classic2": "Classic2",
}


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

    public_track = RENDERER_DIR / "public" / "track"
    public_track.mkdir(parents=True, exist_ok=True)
    shutil.copy2(timeline_path, public_track / "timeline.json")
    shutil.copy2(spectrum_path, public_track / "spectrum.json")
    shutil.copy2(cover_path, public_track / "cover.jpg")
    shutil.copy2(bg_path, public_track / "bg_blur.png")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    silent_mp4 = work_dir / f"silent_{chosen_style}.mp4"
    final_mp4 = output.resolve() if output else (OUTPUT_DIR / f"{work_dir.name}_{chosen_style}.mp4")

    npx_bin = "npx.cmd" if platform.system() == "Windows" else "npx"
    remotion_cmd = [
        npx_bin, "remotion", "render",
        "src/index.ts",
        comp_id,
        str(silent_mp4),
        "--crf", str(crf),
    ]
    if gl:
        remotion_cmd.extend(["--gl", gl])

    print(f"[1/2] Rendering Remotion composition '{comp_id}' -> {silent_mp4.name}")
    run_cmd(remotion_cmd, cwd=RENDERER_DIR)

    print(f"[2/2] Muxing master audio with FFmpeg -> {final_mp4}")
    ffmpeg_bin = get_ffmpeg_exe()
    run_cmd([
        ffmpeg_bin, "-y",
        "-i", str(silent_mp4),
        "-i", str(master_audio),
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(final_mp4),
    ])

    if not keep_silent and silent_mp4.exists():
        silent_mp4.unlink()

    return final_mp4
