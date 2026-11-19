"""Shared utilities for the lyric-study video pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT_DIR / "work"
OUTPUT_DIR = ROOT_DIR / "output"
RENDERER_DIR = ROOT_DIR / "renderer"

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def slugify(name: str) -> str:
    """Create a filesystem-safe slug from a filename stem."""
    cleaned = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    cleaned = re.sub(r"[-\s]+", "-", cleaned)
    return cleaned or "track"


def get_ffmpeg_exe() -> str:
    """Return system ffmpeg if available, otherwise fall back to imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class JobInterrupted(Exception):
    """The queue interrupted a single job between safe operations."""


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    interrupted: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an external command, optionally terminating its process tree on a queue request."""
    options = {
        "cwd": str(cwd) if cwd else None,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if interrupted is None:
        proc = subprocess.run(cmd, capture_output=True, **options)
    else:
        if interrupted():
            raise JobInterrupted()
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              start_new_session=os.name != "nt", **options) as child:
            while True:
                try:
                    stdout, stderr = child.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if not interrupted():
                        continue
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                       capture_output=True, check=False)
                    else:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if child.poll() is None:
                        child.kill()
                    child.communicate()
                    raise JobInterrupted()
            proc = subprocess.CompletedProcess(cmd, child.returncode, stdout, stderr)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def probe_media(media_path: Path) -> dict[str, Any]:
    """Inspect duration and tags using ffprobe if installed, else ffmpeg + mutagen."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        proc = run_cmd([
            ffprobe_bin,
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(media_path),
        ])
        return json.loads(proc.stdout)

    # Fallback: parse duration from `ffmpeg -i` stderr and tags via mutagen
    ffmpeg_bin = get_ffmpeg_exe()
    proc = run_cmd([ffmpeg_bin, "-i", str(media_path)], check=False)
    duration = 0.0
    m = _DURATION_RE.search(proc.stderr or "")
    if m:
        hrs, mins, secs = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = hrs * 3600 + mins * 60 + secs

    tags: dict[str, str] = {}
    try:
        from mutagen import File as MutagenFile

        mf = MutagenFile(str(media_path), easy=True)
        if mf:
            if not duration and getattr(mf, "info", None) and getattr(mf.info, "length", None):
                duration = float(mf.info.length)
            if mf.tags:
                for k, v in mf.tags.items():
                    if isinstance(v, list) and v:
                        tags[k] = str(v[0])
                    else:
                        tags[k] = str(v)
    except Exception:
        pass

    return {
        "format": {
            "duration": str(duration),
            "tags": tags,
        }
    }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(data, f, ensure_ascii=False, indent=2)
        except BaseException:
            f.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
