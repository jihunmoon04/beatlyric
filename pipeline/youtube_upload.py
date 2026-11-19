"""YouTube Data API v3 private video uploader with OAuth token caching."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common import JobInterrupted, ROOT_DIR, WORK_DIR, read_json, write_json
from env_utils import get_env


STUDIO_STATE_DIR = WORK_DIR / ".studio"
YOUTUBE_TOKEN_PATH = STUDIO_STATE_DIR / "youtube_token.json"
YOUTUBE_UPLOAD_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]


def get_client_secrets_path() -> Path:
    configured = get_env("YOUTUBE_CLIENT_SECRETS_FILE")
    if configured:
        p = Path(configured)
        return p if p.is_absolute() else (ROOT_DIR / p)
    return ROOT_DIR / "client_secrets.json"


def get_youtube_status() -> dict[str, Any]:
    secrets_path = get_client_secrets_path()
    has_secrets = secrets_path.exists()
    connected = False
    if YOUTUBE_TOKEN_PATH.exists():
        try:
            tok = read_json(YOUTUBE_TOKEN_PATH)
            connected = bool(tok.get("refresh_token") or tok.get("token"))
        except Exception:
            connected = False
    return {
        "configured": has_secrets,
        "connected": connected,
        "client_secrets_path": str(secrets_path),
    }


def _load_or_authorize_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets_path = get_client_secrets_path()
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"YouTube client_secrets.json not found at {secrets_path}. "
            "Google Cloud Console에서 YouTube Data API v3 OAuth 클라이언트 JSON을 저장하세요."
        )

    creds = None
    if YOUTUBE_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(YOUTUBE_TOKEN_PATH), YOUTUBE_UPLOAD_SCOPE)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        YOUTUBE_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        YOUTUBE_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        return creds

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), YOUTUBE_UPLOAD_SCOPE)
        creds = flow.run_local_server(port=0, prompt="consent")
        YOUTUBE_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        YOUTUBE_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


def authorize_youtube_interactive() -> dict[str, Any]:
    """Trigger local OAuth browser login once and cache refresh token."""
    _load_or_authorize_credentials()
    return get_youtube_status()


def _fmt_yt_timestamp(sec: float) -> str:
    total = max(0, int(sec))
    mins = total // 60
    rem = total % 60
    return f"{mins}:{rem:02d}"


def build_youtube_metadata(work_dir: Path) -> dict[str, Any]:
    """Generate clean YouTube title, timestamped study description, and tags."""
    timeline = read_json(work_dir / "timeline.json")
    meta = timeline.get("meta", {})
    title_raw = str(meta.get("title") or work_dir.name).strip()
    artist_raw = str(meta.get("artist") or "").strip()

    if artist_raw and artist_raw != "Unknown Artist":
        yt_title = f"{artist_raw} - {title_raw} | 가사 · 발음 · 한국어 해석"
    else:
        yt_title = f"{title_raw} | 가사 · 발음 · 한국어 해석"
    if len(yt_title) > 95:
        yt_title = yt_title[:92].rstrip() + "..."

    lines = timeline.get("lines") or []
    ts_lines: list[str] = []
    vocab_lines: list[str] = []
    for ln in lines:
        orig = str(ln.get("original") or "").strip()
        if not orig or orig.startswith("♪"):
            continue
        if len(ts_lines) < 30:
            ts_lines.append(f"{_fmt_yt_timestamp(float(ln.get('start', 0)))} {orig}")
        c = ln.get("comment")
        if isinstance(c, dict) and c.get("phrase") and len(vocab_lines) < 12:
            vocab_lines.append(f"• {c['phrase']} ({c.get('tag', 'NOTE')}): {c.get('meaning', '')}")

    if ts_lines and not ts_lines[0].startswith("0:00 "):
        ts_lines.insert(0, "0:00 Intro")

    parts = [f"{artist_raw} - {title_raw}" if artist_raw else title_raw]
    if meta.get("spotify_url"):
        parts.append(f"Spotify: {meta['spotify_url']}")
    if vocab_lines:
        parts.append("핵심 표현 노트:\n" + "\n".join(vocab_lines))
    if ts_lines:
        parts.append("가사 타임스탬프:\n" + "\n".join(ts_lines))

    description = "\n\n".join(parts)[:4800]
    tags = [t for t in [artist_raw, title_raw, "가사해석", "노래로배우는언어", "발음", "lyric study"] if t]
    return {
        "title": yt_title,
        "description": description,
        "tags": tags[:15],
        "category_id": "10",  # Music
    }


def upload_private_video(
    work_dir: Path,
    video_path: Path,
    interrupted: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Upload rendered MP4 to YouTube as private video."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    if not video_path.exists():
        raise FileNotFoundError(f"Rendered video not found: {video_path}")

    creds = _load_or_authorize_credentials()
    youtube = build("youtube", "v3", credentials=creds)
    yt_meta = build_youtube_metadata(work_dir)

    body = {
        "snippet": {
            "title": yt_meta["title"],
            "description": yt_meta["description"],
            "tags": yt_meta["tags"],
            "categoryId": yt_meta["category_id"],
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    if interrupted is not None:
        checkpoint = work_dir / ".youtube-upload.json"
        stat = video_path.stat()
        identity = {"file": str(video_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        saved = read_json(checkpoint) if checkpoint.exists() else {}
        if saved.get("video") == identity:
            response = saved.get("response")
            request.resumable_uri = saved.get("uri")
        if interrupted():
            raise JobInterrupted()

        if not response and request.resumable_uri:
            # Reconcile the offset with the server even if the last chunk succeeded
            # just before a shutdown.
            reply, content = request.http.request(
                request.resumable_uri, "PUT",
                headers={"Content-Range": f"bytes */{stat.st_size}", "Content-Length": "0"},
            )
            if reply.status in (200, 201):
                response = json.loads(content)
            elif reply.status == 308:
                uploaded = reply.get("range")
                request.resumable_progress = int(uploaded.rsplit("-", 1)[1]) + 1 if uploaded else 0
            elif reply.status in (404, 410):
                raise RuntimeError("이전 YouTube 업로드 세션이 만료되었습니다. 업로드 완료 여부를 확인한 뒤 새 렌더를 시작하세요.")
            else:
                raise RuntimeError(f"YouTube 업로드 진행 위치 확인 실패 ({reply.status})")

        if not response and not request.resumable_uri:
            # Persist the session before the first PUT so even a completed first
            # chunk can be reconciled after a restart.
            headers = {**request.headers,
                       "X-Upload-Content-Type": media.mimetype(),
                       "X-Upload-Content-Length": str(stat.st_size),
                       "Content-Length": str(request.body_size)}
            reply, _ = request.http.request(request.uri, request.method, body=request.body, headers=headers)
            if reply.status != 200 or "location" not in reply:
                raise RuntimeError(f"YouTube 업로드 세션 생성 실패 ({reply.status})")
            request.resumable_uri = reply["location"]
            write_json(checkpoint, {"video": identity, "uri": request.resumable_uri})

    while response is None:
        if interrupted is not None and interrupted():
            raise JobInterrupted()
        _, response = request.next_chunk()
        if interrupted is not None and response is None:
            write_json(checkpoint, {"video": identity, "uri": request.resumable_uri})

    if interrupted is not None:
        write_json(checkpoint, {"video": identity, "uri": request.resumable_uri, "response": response})

    video_id = response["id"]
    return {
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}",
        "studio_url": f"https://studio.youtube.com/video/{video_id}/edit",
        "privacy_status": "private",
        "title": yt_meta["title"],
    }
