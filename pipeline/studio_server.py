"""Local Beat Score Studio HTTP server with Range audio streaming, save, and MP4 render."""

from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from common import read_json, write_json
from render_lib import render_work_dir
from step4_score import compile_score_to_timeline, ensure_score_file, reset_to_algorithm_score


UI_PATH = Path(__file__).resolve().parent / "studio_ui.html"
RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


class StudioHandler(BaseHTTPRequestHandler):
    work_dir: Path
    default_style: str | None = None
    default_output: Path | None = None

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output concise
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

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
        path = unquote(self.path.split("?", 1)[0])
        if path == "/" or path == "/index.html":
            self._serve_file(UI_PATH)
            return

        if path == "/api/state":
            score = ensure_score_file(self.work_dir)
            timeline_path = self.work_dir / "timeline.json"
            timeline = read_json(timeline_path) if timeline_path.exists() else compile_score_to_timeline(score)
            self._send_json({
                "track_name": self.work_dir.name,
                "work_dir": str(self.work_dir),
                "score": score,
                "timeline": timeline,
            })
            return

        if path.startswith("/media/"):
            rel = path[len("/media/") :]
            safe_path = (self.work_dir / rel).resolve()
            if self.work_dir.resolve() not in safe_path.parents:
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid media path")
                return
            self._serve_file(safe_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
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
            write_json(self.work_dir / "score.json", score)
            write_json(self.work_dir / "timeline.json", timeline)
            self._send_json({"ok": True, "timeline": timeline})
            return

        if path == "/api/reset_score":
            try:
                score, timeline = reset_to_algorithm_score(self.work_dir)
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
                write_json(self.work_dir / "score.json", score)
                write_json(self.work_dir / "timeline.json", timeline)

            try:
                style = body.get("style") or self.default_style
                out_path = render_work_dir(
                    self.work_dir,
                    style=style,
                    output=self.default_output,
                )
                self._send_json({"ok": True, "output_path": str(out_path)})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")


def start_studio_server(
    work_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    style: str | None = None,
    output: Path | None = None,
) -> ThreadingHTTPServer:
    ensure_score_file(work_dir)

    class BoundHandler(StudioHandler):
        pass

    BoundHandler.work_dir = work_dir.resolve()
    BoundHandler.default_style = style
    BoundHandler.default_output = output
    return ThreadingHTTPServer((host, port), BoundHandler)


def serve_forever(
    work_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    style: str | None = None,
    output: Path | None = None,
) -> None:
    server = start_studio_server(work_dir, host=host, port=port, style=style, output=output)
    url = f"http://{host}:{port}"
    print(f"Beat Score Studio running at: {url}")
    print(f"Editing work directory     : {work_dir.resolve()}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping studio server.")
        server.server_close()
