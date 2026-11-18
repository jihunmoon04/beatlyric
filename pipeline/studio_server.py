"""Local Beat Score Studio & Automation Dashboard HTTP server."""

from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode

from common import WORK_DIR, read_json, write_json
from env_utils import get_env
from render_lib import render_work_dir
from spotify_sync import (
    build_authorize_url,
    clear_spotify_token,
    exchange_code_for_token,
    fetch_user_playlists,
    get_spotify_auth_status,
)
from step4_score import (
    compile_score_to_timeline,
    ensure_score_file,
    requantize_score_with_grid,
    reset_to_algorithm_score,
)
from step5_enrich import DEFAULT_ACP_MODEL, is_agy_acp_available
from studio_state import StudioQueueManager
from youtube_upload import authorize_youtube_interactive, get_youtube_status


UI_PATH = Path(__file__).resolve().parent / "studio_ui.html"
DASHBOARD_UI_PATH = Path(__file__).resolve().parent / "dashboard_ui.html"
RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


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
            if self.default_work_dir is not None and "track" not in qs:
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
            self._serve_file(safe_path)
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
            if body.get("style"):
                score.setdefault("meta", {})["style"] = body["style"]
            if body.get("content_scale") is not None:
                score.setdefault("meta", {})["content_scale"] = float(body["content_scale"])
            timeline = compile_score_to_timeline(score)
            write_json(work_dir / "score.json", score)
            write_json(work_dir / "timeline.json", timeline)
            self._send_json({"ok": True, "timeline": timeline})
            return

        if path == "/api/reset_score":
            try:
                score, timeline = reset_to_algorithm_score(work_dir)
                self._send_json({"ok": True, "score": score, "timeline": timeline})
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
                )
                self._send_json({"ok": True, "score": score, "timeline": timeline})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if path == "/api/render":
            score = body.get("score")
            if score:
                if body.get("style"):
                    score.setdefault("meta", {})["style"] = body["style"]
                if body.get("content_scale") is not None:
                    score.setdefault("meta", {})["content_scale"] = float(body["content_scale"])
                timeline = compile_score_to_timeline(score)
                write_json(work_dir / "score.json", score)
                write_json(work_dir / "timeline.json", timeline)

            try:
                style = body.get("style") or self.default_style
                out_path = render_work_dir(
                    work_dir,
                    style=style,
                    output=self.default_output if work_dir == self.default_work_dir else None,
                )
                self.queue_mgr.mark_track_rendered(work_dir.name, out_path, style=style)
                self._send_json({"ok": True, "output_path": str(out_path)})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


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
        server.server_close()
