"""Local Beat Score Studio & Automation Dashboard HTTP server."""

from __future__ import annotations

import json
import mimetypes
import os
import platform
import re
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode

from common import OUTPUT_DIR, WORK_DIR, read_json, write_json
from env_utils import get_env
from render_lib import is_render_busy, render_work_dir
from spotify_sync import (
    build_authorize_url,
    clear_spotify_token,
    exchange_code_for_token,
    fetch_user_playlists,
    get_spotify_auth_status,
)
from step1_ingest import extract_dominant_accent
from step4_score import (
    compile_score_to_timeline,
    ensure_score_file,
    estimate_beat_grid,
    waveform_peaks,
    requantize_score_with_grid,
    reset_to_algorithm_score,
)
from step5_enrich import DEFAULT_ACP_MODEL, is_agy_acp_available
from studio_state import StudioQueueManager
from word_alignment import WordAlignmentUnavailable, preview_score_words
from youtube_upload import authorize_youtube_interactive, get_youtube_status


UI_PATH = Path(__file__).resolve().parent / "studio_ui.html"
DASHBOARD_UI_PATH = Path(__file__).resolve().parent / "dashboard_ui.html"
RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def open_output_folder(
    slug: str | None = None,
    output_path: str | None = None,
    recorded_output: str | None = None,
) -> dict[str, Any]:
    """Open the rendered output directory (selecting the target MP4 when present)."""
    out_root = OUTPUT_DIR.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    selected_file: Path | None = None

    for raw_cand in (output_path, recorded_output):
        if not raw_cand:
            continue
        try:
            cand = Path(str(raw_cand)).resolve()
        except OSError:
            continue
        if cand.is_file() and cand.parent == out_root:
            selected_file = cand
            break

    if selected_file is None and slug:
        clean_slug = str(slug).strip()
        if (
            clean_slug
            and not clean_slug.startswith(".")
            and "/" not in clean_slug
            and "\\" not in clean_slug
            and Path(clean_slug).name == clean_slug
        ):
            matches = [
                p.resolve()
                for p in out_root.glob(f"{clean_slug}_*.mp4")
                if p.is_file() and p.resolve().parent == out_root
            ]
            if matches:
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                selected_file = matches[0]

    system = platform.system()
    if system == "Windows":
        if selected_file is not None:
            subprocess.Popen(["explorer", f"/select,{selected_file}"])
        else:
            os.startfile(str(out_root))
    elif system == "Darwin":
        if selected_file is not None:
            subprocess.Popen(["open", "-R", str(selected_file)])
        else:
            subprocess.Popen(["open", str(out_root)])
    else:
        subprocess.Popen(["xdg-open", str(out_root)])

    return {
        "folder": str(out_root),
        "selected_file": str(selected_file) if selected_file else None,
    }


def _apply_display_meta(score: dict[str, Any], body: dict[str, Any]) -> None:
    """Apply optional UI display toggles and style properties onto score.meta."""
    meta = score.setdefault("meta", {})
    if body.get("style"):
        meta["style"] = body["style"]
    if body.get("content_scale") is not None:
        meta["content_scale"] = float(body["content_scale"])
    if isinstance(body.get("accent_color"), str) and HEX_COLOR_RE.match(body["accent_color"].strip()):
        meta["accent_color"] = body["accent_color"].strip().lower()
    for bool_key in ("karaoke_sweep", "show_word_highlight", "show_pronunciation", "show_comment",
                     "smooth_lyric_scroll", "lyric_word_lift", "show_song_context"):
        if bool_key in body and body[bool_key] is not None:
            meta[bool_key] = bool(body[bool_key])


class StudioHandler(BaseHTTPRequestHandler):
    default_work_dir: Path | None = None
    default_style: str | None = None
    default_output: Path | None = None
    queue_mgr: StudioQueueManager
    oauth_states: set[str]

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _parse_url(self) -> tuple[str, dict[str, list[str]]]:
        raw_path, _, raw_qs = self.path.partition("?")
        return unquote(raw_path), parse_qs(raw_qs)

    def _resolve_track_dir(self, qs: dict[str, list[str]], body: dict[str, Any] | None = None) -> Path:
        slug = ""
        if "track" in qs and qs["track"]:
            slug = qs["track"][0].strip()
        elif body and body.get("track"):
            slug = str(body["track"]).strip()

        if slug:
            safe_slug = Path(slug).name
            cand = (WORK_DIR / safe_slug).resolve()
            if WORK_DIR.resolve() in cand.parents and cand.is_dir():
                return cand
            raise FileNotFoundError(f"Track work directory not found: {safe_slug}")

        if self.default_work_dir and self.default_work_dir.exists():
            return self.default_work_dir

        for child in sorted(WORK_DIR.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and (child / "meta.json").exists():
                return child.resolve()

        raise FileNotFoundError("No prepared track available in work/")

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        size = file_path.stat().st_size
        ctype, _ = mimetypes.guess_type(str(file_path))
        if file_path.suffix == ".m4a":
            ctype = "audio/mp4"
        elif file_path.suffix == ".wav":
            ctype = "audio/wav"
        elif file_path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif file_path.suffix == ".json":
            ctype = "application/json; charset=utf-8"
        elif not ctype:
            ctype = "application/octet-stream"

        range_header = self.headers.get("Range")
        if range_header:
            m = RANGE_RE.match(range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                chunk_len = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(chunk_len))
                self.end_headers()
                with file_path.open("rb") as f:
                    f.seek(start)
                    remaining = chunk_len
                    while remaining > 0:
                        buf = f.read(min(65536, remaining))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        remaining -= len(buf)
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with file_path.open("rb") as f:
            while True:
                buf = f.read(65536)
                if not buf:
                    break
                self.wfile.write(buf)

    def do_GET(self) -> None:
        path, qs = self._parse_url()

        if path in {"/", "/index.html"}:
            if self.default_work_dir is not None and self.default_work_dir.exists() and "track" not in qs:
                self._serve_file(UI_PATH)
            elif "track" in qs:
                self._serve_file(UI_PATH)
            else:
                self._serve_file(DASHBOARD_UI_PATH)
            return

        if path == "/dashboard":
            self._serve_file(DASHBOARD_UI_PATH)
            return

        if path == "/studio":
            self._serve_file(UI_PATH)
            return

        if path == "/api/dashboard":
            snap = self.queue_mgr.get_dashboard_snapshot()
            snap["spotify"] = get_spotify_auth_status()
            snap["youtube"] = get_youtube_status()
            snap["acp"] = {
                "available": is_agy_acp_available(),
                "model": get_env("AGY_ACP_MODEL", DEFAULT_ACP_MODEL) or DEFAULT_ACP_MODEL,
            }
            self._send_json(snap)
            return

        if path == "/api/spotify/login":
            try:
                auth_url, csrf_state = build_authorize_url()
                self.oauth_states.add(csrf_state)
                self._redirect(auth_url)
            except Exception as exc:
                self._redirect("/dashboard?" + urlencode({"spotify_error": str(exc)}))
            return

        if path == "/api/spotify/callback":
            if "error" in qs:
                self._redirect("/dashboard?" + urlencode({"spotify_error": qs["error"][0]}))
                return
            code = (qs.get("code") or [""])[0]
            state = (qs.get("state") or [""])[0]
            if not code:
                self._redirect("/dashboard?" + urlencode({"spotify_error": "missing_code"}))
                return
            if self.oauth_states and state not in self.oauth_states:
                self._redirect("/dashboard?" + urlencode({"spotify_error": "invalid_oauth_state"}))
                return
            self.oauth_states.discard(state)
            try:
                exchange_code_for_token(code)
                self._redirect("/dashboard?spotify=connected")
            except Exception as exc:
                self._redirect("/dashboard?" + urlencode({"spotify_error": str(exc)}))
            return

        if path == "/api/spotify/playlists":
            try:
                playlists = fetch_user_playlists()
                self._send_json({"ok": True, "playlists": playlists})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/youtube/status":
            self._send_json(get_youtube_status())
            return

        if path == "/api/waveform":
            try:
                wav = self._resolve_track_dir(qs) / "analysis.wav"
                peaks = waveform_peaks(wav, wav.stat().st_mtime_ns)
                self._send_json(peaks)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=404)
            return

        if path == "/api/state":
            try:
                work_dir = self._resolve_track_dir(qs)
            except FileNotFoundError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=404)
                return
            score = ensure_score_file(work_dir)
            timeline_path = work_dir / "timeline.json"
            timeline = read_json(timeline_path) if timeline_path.exists() else compile_score_to_timeline(score)
            self._send_json({
                "track_name": work_dir.name,
                "work_dir": str(work_dir),
                "score": score,
                "timeline": timeline,
            })
            return

        if path.startswith("/media/"):
            rel = path[len("/media/") :]
            parts = rel.split("/", 1)
            target_base = self.default_work_dir
            rel_sub = rel
            if len(parts) == 2:
                cand_dir = (WORK_DIR / Path(parts[0]).name).resolve()
                if WORK_DIR.resolve() in cand_dir.parents and cand_dir.is_dir():
                    target_base = cand_dir
                    rel_sub = parts[1]

            if not target_base:
                self.send_error(HTTPStatus.NOT_FOUND, "Track media base not found")
                return

            safe_path = (target_base / rel_sub).resolve()
            if target_base.resolve() not in safe_path.parents:
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid media path")
                return
            try:
                self._serve_file(safe_path)
            except ConnectionError:
                pass
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path, qs = self._parse_url()
        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
            return

        if path == "/api/spotify/logout":
            clear_spotify_token()
            self._send_json({"ok": True})
            return

        if path == "/api/spotify/sync":
            try:
                res = self.queue_mgr.sync_spotify_playlist(
                    playlist_id_or_url=body.get("playlist_id") or None,
                    mode=str(body.get("mode") or "incremental"),
                    limit=int(body.get("limit") or 5),
                    style=str(body.get("style") or self.default_style or "minimal"),
                )
                self._send_json({"ok": True, **res})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/youtube/auth":
            try:
                status = authorize_youtube_interactive()
                self._send_json({"ok": True, "youtube": status})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/tracks/enqueue":
            try:
                track = self.queue_mgr.enqueue_manual_track(
                    title=str(body.get("title") or ""),
                    artist=str(body.get("artist") or "Unknown Artist"),
                    source_url=str(body.get("source_url") or "").strip() or None,
                    language=str(body.get("language") or "").strip() or None,
                    style=str(body.get("style") or "minimal"),
                    upload_to_youtube=bool(body.get("upload_to_youtube", True)),
                )
                self._send_json({"ok": True, "track": track})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        m_approve = re.fullmatch(r"/api/tracks/([^/]+)/approve", path)
        if m_approve:
            slug = Path(m_approve.group(1)).name
            try:
                track = self.queue_mgr.approve_track(
                    slug=slug,
                    style=body.get("style"),
                    upload_to_youtube=bool(body.get("upload_to_youtube", True)),
                )
                self._send_json({"ok": True, "track": track})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        control = re.fullmatch(r"/api/tracks/([^/]+)/(pause|resume|cancel)", path)
        if control:
            try:
                slug, action = control.groups()
                track = self.queue_mgr.control_track(Path(slug).name, action)
                self._send_json({"ok": True, "track": track})
            except (KeyError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        m_retry = re.fullmatch(r"/api/tracks/([^/]+)/retry", path)
        if m_retry:
            slug = Path(m_retry.group(1)).name
            try:
                track = self.queue_mgr.retry_track(slug=slug)
                self._send_json({"ok": True, "track": track})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        m_acp = re.fullmatch(r"/api/tracks/([^/]+)/enrich_acp", path)
        if m_acp:
            slug = Path(m_acp.group(1)).name
            try:
                res = self.queue_mgr.enrich_track_with_acp(slug=slug, model_id=body.get("model"))
                self._send_json({"ok": True, **res})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        m_del = re.fullmatch(r"/api/tracks/([^/]+)/delete", path)
        if m_del:
            slug = Path(unquote(m_del.group(1))).name
            self._handle_delete_track(slug, delete_output=bool(body.get("delete_output", False)))
            return

        if path == "/api/open_output_folder":
            slug = str(body.get("track") or body.get("slug") or "").strip() or None
            output_path = str(body.get("output_path") or "").strip() or None
            recorded_output: str | None = None
            if slug:
                with self.queue_mgr._lock:
                    tr_item = self.queue_mgr._state.get("tracks", {}).get(slug)
                    if tr_item and tr_item.get("output_path"):
                        recorded_output = str(tr_item["output_path"])
            try:
                res = open_output_folder(slug=slug, output_path=output_path, recorded_output=recorded_output)
                self._send_json({"ok": True, **res})
            except OSError as exc:
                self._send_json({"ok": False, "error": f"폴더를 열 수 없습니다: {exc}"}, status=500)
            return

        try:
            work_dir = self._resolve_track_dir(qs, body)
        except FileNotFoundError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=404)
            return

        if path == "/api/save":
            score = body.get("score")
            if not score:
                self._send_json({"ok": False, "error": "Missing 'score' payload"}, status=400)
                return
            _apply_display_meta(score, body)
            timeline = compile_score_to_timeline(score)
            write_json(work_dir / "score.json", score)
            write_json(work_dir / "timeline.json", timeline)
            self._send_json({"ok": True, "timeline": timeline})
            return

        if path == "/api/reextract_accent":
            cover_path = work_dir / "cover.jpg"
            if not cover_path.exists():
                self._send_json({"ok": False, "error": "cover.jpg not found in work directory"}, status=404)
                return
            try:
                accent = extract_dominant_accent(cover_path)
                self._send_json({"ok": True, "accent_color": accent})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path in {"/api/reset_score", "/api/reanalyze_beat_grid"}:
            try:
                score, timeline = reset_to_algorithm_score(
                    work_dir, current_score=body.get("score"),
                    persist=path == "/api/reset_score" and not body.get("preview", False),
                )
                self._send_json({"ok": True, "score": score, "timeline": timeline})
            except FileNotFoundError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=404)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/analyze_beat":
            try:
                bpm = float(body["bpm"])
                if not 40 <= bpm <= 240:
                    raise ValueError("BPM은 40~240이어야 합니다.")
                _, phase, downbeat, margin = estimate_beat_grid(work_dir / "analysis.wav", bpm=bpm)
                self._send_json({"ok": True, "phase": phase, "downbeat_offset": downbeat, "margin": margin})
            except (KeyError, OSError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/align_words":
            score = body.get("score")
            requested = body.get("line_ids")
            if not isinstance(score, dict) or not isinstance(score.get("lines"), list):
                self._send_json({"ok": False, "error": "편집 중인 악보가 필요합니다."}, status=400)
                return
            if requested is not None and (
                not isinstance(requested, list) or not requested or any(type(value) is not int for value in requested)
            ):
                self._send_json({"ok": False, "error": "정렬할 줄을 선택해 주세요."}, status=400)
                return
            try:
                result = preview_score_words(work_dir, score, set(requested) if requested is not None else None)
                self._send_json({"ok": True, **result})
            except WordAlignmentUnavailable as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=503)
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/requantize":
            try:
                bpm = float(body.get("bpm") or 120.0)
                t0 = float(body.get("t0") if body.get("t0") is not None else 0.0)
                score, timeline = requantize_score_with_grid(
                    work_dir=work_dir,
                    bpm=bpm,
                    t0=t0,
                    current_score=body.get("score"),
                    persist=not body.get("preview", False),
                    lines_only=bool(body.get("lines_only")),
                )
                self._send_json({"ok": True, "score": score, "timeline": timeline})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/render":
            if not body.get("queue") and is_render_busy():
                self._send_json(
                    {
                        "ok": False,
                        "error": "다른 트랙이 렌더링 중입니다. 완료 후 다시 시도하거나 '검수 완료 · 업로드 대기열 추가'를 사용하세요.",
                    },
                    status=409,
                )
                return

            score = body.get("score")
            timeline = None
            if score:
                _apply_display_meta(score, body)
                timeline = compile_score_to_timeline(score)

            try:
                style = body.get("style") or self.default_style
                raw_cs = body.get("clip_start")
                raw_ce = body.get("clip_end")
                clip_start = None if raw_cs in (None, "") else float(raw_cs)
                clip_end = None if raw_ce in (None, "") else float(raw_ce)
                if body.get("queue"):
                    track = self.queue_mgr.enqueue_render(
                        work_dir.name, style=style,
                        output=self.default_output if work_dir == self.default_work_dir else None,
                        clip_start=clip_start, clip_end=clip_end,
                        score=score, timeline=timeline,
                    )
                    self._send_json({"ok": True, "track": track}, status=202)
                    return
                if score and timeline:
                    write_json(work_dir / "score.json", score)
                    write_json(work_dir / "timeline.json", timeline)
                out_path = render_work_dir(
                    work_dir,
                    style=style,
                    output=self.default_output if work_dir == self.default_work_dir else None,
                    clip_start=clip_start,
                    clip_end=clip_end,
                )
                self.queue_mgr.mark_track_rendered(work_dir.name, out_path, style=style)
                self._send_json({"ok": True, "output_path": str(out_path)})
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle_delete_track(self, slug: str, delete_output: bool) -> None:
        try:
            res = self.queue_mgr.delete_track(slug=slug, delete_output=delete_output)
            self._send_json({"ok": True, **res})
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except FileNotFoundError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=404)
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=409)
        except OSError as exc:
            self._send_json({"ok": False, "error": f"파일 삭제 중 오류가 발생했습니다: {exc}"}, status=500)

    def do_DELETE(self) -> None:
        path, qs = self._parse_url()
        m_del = re.fullmatch(r"/api/tracks/([^/]+)", path)
        if not m_del:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        slug = Path(unquote(m_del.group(1))).name
        delete_output = (qs.get("delete_output", [""])[0] or "").strip().lower() in {"1", "true", "yes"}
        if int(self.headers.get("Content-Length", "0") or 0) > 0:
            try:
                body = self._read_json_body()
                if isinstance(body, dict) and "delete_output" in body:
                    delete_output = bool(body.get("delete_output"))
            except Exception as exc:
                self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
                return

        self._handle_delete_track(slug, delete_output=delete_output)


def start_studio_server(
    work_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    style: str | None = None,
    output: Path | None = None,
) -> ThreadingHTTPServer:
    if work_dir is not None:
        ensure_score_file(work_dir)

    queue_mgr = StudioQueueManager()
    queue_mgr.start_workers()

    class BoundHandler(StudioHandler):
        pass

    BoundHandler.default_work_dir = work_dir.resolve() if work_dir else None
    BoundHandler.default_style = style
    BoundHandler.default_output = output
    BoundHandler.queue_mgr = queue_mgr
    BoundHandler.oauth_states = set()
    return ThreadingHTTPServer((host, port), BoundHandler)


def serve_forever(
    work_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    style: str | None = None,
    output: Path | None = None,
) -> None:
    server = start_studio_server(work_dir, host=host, port=port, style=style, output=output)
    url = f"http://{host}:{port}"
    print(f"Beat Score Studio running at: {url}")
    if work_dir is not None:
        print(f"Editing work directory     : {work_dir.resolve()}")
    else:
        print("Mode                       : Multi-Track Automation Dashboard")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping studio server.")
    finally:
        server.RequestHandlerClass.queue_mgr.stop_workers()
        server.server_close()
