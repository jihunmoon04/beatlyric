"""Spotify OAuth and incremental playlist synchronization."""

from __future__ import annotations

import base64
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from common import WORK_DIR, read_json, write_json
from env_utils import get_env


STUDIO_STATE_DIR = WORK_DIR / ".studio"
SPOTIFY_TOKEN_PATH = STUDIO_STATE_DIR / "spotify_token.json"
SPOTIFY_SCOPES = "playlist-read-private playlist-read-collaborative user-read-private"
_PLAYLIST_ID_RE = re.compile(r"(?:playlist[:/])([A-Za-z0-9]+)")


def parse_spotify_playlist_id(raw_value: str) -> str:
    """Extract a 22-char Spotify playlist ID from a URL, URI, or raw ID."""
    val = (raw_value or "").strip()
    if not val:
        return ""
    m = _PLAYLIST_ID_RE.search(val)
    if m:
        return m.group(1)
    return val.split("?", 1)[0].split("#", 1)[0].strip()


def get_spotify_config() -> dict[str, str]:
    """Read Spotify credentials and default playlist from environment / .env."""
    return {
        "client_id": get_env("SPOTIFY_CLIENT_ID"),
        "client_secret": get_env("SPOTIFY_CLIENT_SECRET"),
        "redirect_uri": get_env("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8765/api/spotify/callback"),
        "default_playlist_raw": get_env("SPOTIFY_PLAYLIST_ID"),
        "default_playlist_id": parse_spotify_playlist_id(get_env("SPOTIFY_PLAYLIST_ID")),
    }


def is_spotify_configured() -> bool:
    cfg = get_spotify_config()
    return bool(cfg["client_id"] and cfg["client_secret"])


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def load_spotify_token() -> dict[str, Any] | None:
    if not SPOTIFY_TOKEN_PATH.exists():
        return None
    try:
        return read_json(SPOTIFY_TOKEN_PATH)
    except Exception:
        return None


def save_spotify_token(token_data: dict[str, Any]) -> None:
    write_json(SPOTIFY_TOKEN_PATH, token_data)


def clear_spotify_token() -> None:
    if SPOTIFY_TOKEN_PATH.exists():
        SPOTIFY_TOKEN_PATH.unlink()


def build_authorize_url(state: str | None = None) -> tuple[str, str]:
    """Build Spotify OAuth Authorization Code URL and state token."""
    cfg = get_spotify_config()
    if not cfg["client_id"] or not cfg["client_secret"]:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env")
    csrf_state = state or secrets.token_urlsafe(24)
    params = {
        "client_id": cfg["client_id"],
        "response_type": "code",
        "redirect_uri": cfg["redirect_uri"],
        "scope": SPOTIFY_SCOPES,
        "state": csrf_state,
        "show_dialog": "false",
    }
    return f"https://accounts.spotify.com/authorize?{urlencode(params)}", csrf_state


def exchange_code_for_token(code: str) -> dict[str, Any]:
    """Exchange OAuth authorization code for access + refresh tokens and profile."""
    cfg = get_spotify_config()
    res = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": cfg["redirect_uri"],
        },
        headers={"Authorization": _basic_auth_header(cfg["client_id"], cfg["client_secret"])},
        timeout=20,
    )
    if not res.ok:
        raise RuntimeError(f"Spotify token exchange failed ({res.status_code}): {res.text}")

    payload = res.json()
    expires_in = int(payload.get("expires_in", 3600))
    token_data: dict[str, Any] = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", ""),
        "scope": payload.get("scope", ""),
        "expires_at": int(time.time()) + expires_in,
    }

    me_res = requests.get(
        "https://api.spotify.com/v1/me",
        headers={"Authorization": f"Bearer {token_data['access_token']}"},
        timeout=15,
    )
    if me_res.ok:
        me = me_res.json()
        token_data["user"] = {
            "id": me.get("id", ""),
            "display_name": me.get("display_name") or me.get("id", ""),
        }
    save_spotify_token(token_data)
    return token_data


def _refresh_token(token_data: dict[str, Any]) -> dict[str, Any]:
    cfg = get_spotify_config()
    refresh_tok = token_data.get("refresh_token")
    if not refresh_tok:
        raise RuntimeError("Spotify refresh_token missing; please reconnect Spotify login.")

    res = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_tok,
        },
        headers={"Authorization": _basic_auth_header(cfg["client_id"], cfg["client_secret"])},
        timeout=20,
    )
    if not res.ok:
        raise RuntimeError(f"Spotify token refresh failed ({res.status_code}): {res.text}")

    payload = res.json()
    updated = dict(token_data)
    updated["access_token"] = payload["access_token"]
    if payload.get("refresh_token"):
        updated["refresh_token"] = payload["refresh_token"]
    updated["expires_at"] = int(time.time()) + int(payload.get("expires_in", 3600))
    save_spotify_token(updated)
    return updated


def get_valid_access_token() -> str:
    """Return a non-expired Spotify user access token, refreshing when needed."""
    token_data = load_spotify_token()
    if not token_data or not token_data.get("access_token"):
        raise RuntimeError("Spotify OAuth login required. Click 'Spotify 로그인' first.")
    if int(token_data.get("expires_at", 0)) <= int(time.time()) + 60:
        token_data = _refresh_token(token_data)
    return str(token_data["access_token"])


def get_spotify_auth_status() -> dict[str, Any]:
    cfg = get_spotify_config()
    token_data = load_spotify_token()
    connected = bool(token_data and (token_data.get("refresh_token") or token_data.get("access_token")))
    return {
        "configured": is_spotify_configured(),
        "connected": connected,
        "redirect_uri": cfg["redirect_uri"],
        "default_playlist_id": cfg["default_playlist_id"],
        "user": (token_data or {}).get("user"),
    }


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _spotify_get(path_or_url: str, token: str, params: dict[str, Any] | None = None) -> requests.Response:
    url = path_or_url if path_or_url.startswith("http") else f"https://api.spotify.com/v1{path_or_url}"
    return requests.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )


def fetch_user_playlists(limit: int = 50) -> list[dict[str, Any]]:
    """List playlists owned/followed by the signed-in Spotify user."""
    token = get_valid_access_token()
    res = _spotify_get("/me/playlists", token, params={"limit": limit})
    if not res.ok:
        raise RuntimeError(f"Failed to list playlists ({res.status_code}): {res.text}")
    out: list[dict[str, Any]] = []
    for pl in res.json().get("items", []) or []:
        if not pl:
            continue
        out.append({
            "id": pl.get("id", ""),
            "name": pl.get("name", ""),
            "public": pl.get("public"),
            "owner": (pl.get("owner") or {}).get("display_name", ""),
        })
    return out


def fetch_playlist_tracks(
    playlist_id_or_url: str | None = None,
    since_iso: str | None = None,
) -> dict[str, Any]:
    """Fetch playlist metadata and tracks added after `since_iso` (incremental sync)."""
    cfg = get_spotify_config()
    playlist_id = parse_spotify_playlist_id(playlist_id_or_url or cfg["default_playlist_id"])
    if not playlist_id:
        raise RuntimeError("Spotify playlist ID is not configured.")

    token = get_valid_access_token()
    pl_res = _spotify_get(
        f"/playlists/{playlist_id}",
        token,
        params={"fields": "id,name,public,owner(display_name,id),external_urls.spotify"},
    )
    if not pl_res.ok:
        raise RuntimeError(f"Spotify playlist fetch failed ({pl_res.status_code}): {pl_res.text}")
    pl_meta = pl_res.json()

    # Spotify Feb 2026 API renamed /playlists/{id}/tracks -> /playlists/{id}/items
    endpoint = f"/playlists/{playlist_id}/items"
    page_res = _spotify_get(endpoint, token, params={"limit": 100, "offset": 0})
    if page_res.status_code == 404:
        endpoint = f"/playlists/{playlist_id}/tracks"
        page_res = _spotify_get(endpoint, token, params={"limit": 100, "offset": 0})
    if not page_res.ok:
        raise RuntimeError(f"Spotify playlist items fetch failed ({page_res.status_code}): {page_res.text}")

    raw_entries: list[dict[str, Any]] = []
    page_data = page_res.json()
    raw_entries.extend(page_data.get("items") or [])
    next_url = page_data.get("next")
    while next_url:
        nxt = _spotify_get(next_url, token)
        if not nxt.ok:
            break
        nxt_data = nxt.json()
        raw_entries.extend(nxt_data.get("items") or [])
        next_url = nxt_data.get("next")

    since_dt = _parse_iso_dt(since_iso)
    all_tracks: list[dict[str, Any]] = []
    new_tracks: list[dict[str, Any]] = []
    max_added_at: str | None = None
    max_added_dt: datetime | None = None

    for entry in raw_entries:
        if not entry:
            continue
        track = entry.get("item") or entry.get("track")
        if not track or not isinstance(track, dict):
            continue
        if track.get("is_local"):
            continue
        track_type = track.get("type")
        if track_type and track_type != "track":
            continue
        track_id = track.get("id")
        title = (track.get("name") or "").strip()
        if not track_id or not title:
            continue

        artists = ", ".join(a.get("name", "") for a in (track.get("artists") or []) if a.get("name")).strip()
        album = track.get("album") or {}
        images = sorted(
            album.get("images") or [],
            key=lambda im: int(im.get("width") or 0),
            reverse=True,
        )
        cover_url = images[0].get("url", "") if images else ""
        added_at = entry.get("added_at") or ""
        added_dt = _parse_iso_dt(added_at)

        if added_dt and (max_added_dt is None or added_dt > max_added_dt):
            max_added_dt = added_dt
            max_added_at = added_at

        record = {
            "spotify_track_id": track_id,
            "title": title,
            "artist": artists or "Unknown Artist",
            "album": album.get("name", ""),
            "duration_ms": int(track.get("duration_ms") or 0),
            "added_at": added_at,
            "cover_url": cover_url,
            "spotify_url": (track.get("external_urls") or {}).get("spotify", ""),
        }
        all_tracks.append(record)
        if since_dt is None or (added_dt is not None and added_dt > since_dt):
            new_tracks.append(record)

    return {
        "playlist": {
            "id": pl_meta.get("id", playlist_id),
            "name": pl_meta.get("name", playlist_id),
            "public": pl_meta.get("public"),
            "owner": (pl_meta.get("owner") or {}).get("display_name", ""),
            "url": (pl_meta.get("external_urls") or {}).get("spotify", ""),
        },
        "total_items": len(all_tracks),
        "new_tracks": new_tracks,
        "all_tracks": all_tracks,
        "max_added_at": max_added_at,
    }
