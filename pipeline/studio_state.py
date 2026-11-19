"""Persistent queue state and background workers for Beat Score Studio automation."""

from __future__ import annotations

import re
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import JobInterrupted, OUTPUT_DIR, WORK_DIR, read_json, run_cmd, slugify, write_json
from queue_prepare import PREPARE_STEPS
from render_lib import STYLE_TO_COMP, render_work_dir
from spotify_sync import (
    fetch_playlist_tracks,
    get_spotify_config,
    parse_spotify_playlist_id,
)
from step5_enrich import DEFAULT_ACP_MODEL
from env_utils import get_env
from youtube_upload import get_youtube_status, upload_private_video


STATE_DIR = WORK_DIR / ".studio"
QUEUE_PATH = STATE_DIR / "queue.json"

PREPARE_SEQUENCE = tuple(PREPARE_STEPS)
QUEUED_TO_ACTIVE = {
    "queued_prepare": "preparing", "queued_render": "rendering", "queued_upload": "uploading",
}
ACTIVE_TO_QUEUED = {active: queued for queued, active in QUEUED_TO_ACTIVE.items()}
QUEUE_STEPS = {"queued_prepare": "source", "queued_render": "render", "queued_upload": "upload"}


def _queue_for_step(step: str) -> str:
    return {"render": "queued_render", "upload": "queued_upload"}.get(step, "queued_prepare")


def _step_label(step: str) -> str:
    if step in PREPARE_STEPS:
        return PREPARE_STEPS[step].removesuffix(" 중")
    return {"render": "MP4 렌더링", "upload": "YouTube 업로드", "refresh_notes": "ACP 주석"}.get(step, step)


_JA_RE = re.compile(r"[\u3040-\u30ff]")
_KO_RE = re.compile(r"[\uac00-\ud7af]")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _guess_language(title: str, artist: str) -> str:
    sample = f"{title} {artist}"
    if _JA_RE.search(sample):
        return "ja"
    if _KO_RE.search(sample):
        return "ko"
    return "en"


def _default_state() -> dict[str, Any]:
    cfg = get_spotify_config()
    return {
        "version": 1,
        "spotify_sync": {
            "playlist_id": cfg.get("default_playlist_id", ""),
            "playlist_name": "",
            "last_synced_at": None,
            "last_checked_at": None,
            "last_error": None,
            "seen_track_ids": [],
        },
        "tracks": {},
    }


class StudioQueueManager:
    """Thread-safe queue manager + background prepare and render/upload workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._prepare_event = threading.Event()
        self._render_event = threading.Event()
        self._stop_event = threading.Event()
        self._workers_started = False
        self._active_jobs: dict[str, threading.Event] = {}
        self._workers: list[threading.Thread] = []
        self._logs: list[dict[str, str]] = []
        self._state = self._load_or_init_state()
        self.sync_existing_work_dirs()

    def _log(self, msg: str, slug: str | None = None, level: str = "info") -> None:
        entry = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "slug": slug or "-",
            "level": level,
            "message": msg,
        }
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > 120:
                self._logs = self._logs[-120:]

    def _load_or_init_state(self) -> dict[str, Any]:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if QUEUE_PATH.exists():
            try:
                data = read_json(QUEUE_PATH)
                if isinstance(data, dict) and "tracks" in data:
                    data.setdefault("spotify_sync", _default_state()["spotify_sync"])
                    return data
            except Exception:
                pass
        state = _default_state()
        write_json(QUEUE_PATH, state)
        return state

    def _save_state_locked(self) -> None:
        write_json(QUEUE_PATH, self._state)

    def sync_existing_work_dirs(self) -> None:
        """Discover existing work/<slug> folders and recover interrupted worker states."""
        with self._lock:
            changed = False
            tracks = self._state.setdefault("tracks", {})

            # Only recover persisted running jobs at startup, never during a dashboard poll.
            if not self._workers_started:
                for item in tracks.values():
                    status = item.get("status")
                    if status not in ACTIVE_TO_QUEUED:
                        continue
                    queued = ACTIVE_TO_QUEUED[status]
                    item["stage"] = item.get("stage") or QUEUE_STEPS[queued]
                    action = item.pop("pending_action", None)
                    item["status"] = {"pause": "paused", "cancel": "cancelled"}.get(action, queued)
                    item["step_message"] = "서버 재시작 후 작업 지점 복구"
                    changed = True

            if WORK_DIR.exists():
                for child in sorted(WORK_DIR.iterdir()):
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    slug = child.name
                    if slug in tracks:
                        continue
                    score_path = child / "score.json"
                    meta_path = child / "meta.json"
                    timeline_path = child / "timeline.json"
                    if not (score_path.exists() or timeline_path.exists() or meta_path.exists()):
                        continue

                    meta: dict[str, Any] = {}
                    if meta_path.exists():
                        try:
                            meta = read_json(meta_path)
                        except Exception:
                            meta = {}

                    style = str(meta.get("style") or "minimal")
                    rendered_mp4 = OUTPUT_DIR / f"{slug}_{style}.mp4"
                    status = "rendered" if rendered_mp4.exists() else "needs_review"

                    tracks[slug] = {
                        "slug": slug,
                        "title": meta.get("title") or slug,
                        "artist": meta.get("artist") or "Unknown Artist",
                        "album": meta.get("album", ""),
                        "language": meta.get("language", "en"),
                        "style": style,
                        "duration": meta.get("duration", 0),
                        "spotify_track_id": meta.get("spotify_track_id", ""),
                        "spotify_url": meta.get("spotify_url", ""),
                        "added_at": meta.get("added_at", ""),
                        "cover_url": "",
                        "source": "existing_work",
                        "status": status,
                        "step_message": "기존 작업 폴더에서 불러옴",
                        "error": None,
                        "upload_requested": False,
                        "output_path": str(rendered_mp4) if rendered_mp4.exists() else "",
                        "youtube": None,
                        "created_at": _utc_now_iso(),
                        "updated_at": _utc_now_iso(),
                    }
                    changed = True

            if changed:
                self._save_state_locked()

    def start_workers(self) -> None:
        with self._lock:
            if self._workers_started:
                return
            self._workers_started = True
        t_prep = threading.Thread(target=self._prepare_worker_loop, name="beatlyric-prepare", daemon=True)
        t_ren = threading.Thread(target=self._render_worker_loop, name="beatlyric-render-upload", daemon=True)
        self._workers = [t_prep, t_ren]
        for worker in self._workers:
            worker.start()
        self._prepare_event.set()
        self._render_event.set()

    def stop_workers(self) -> None:
        self._stop_event.set()
        self._prepare_event.set()
        self._render_event.set()
        for worker in self._workers:
            worker.join(timeout=2)

    def _update_track(self, slug: str, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            item = self._state["tracks"].get(slug)
            if not item:
                return None
            item.update(fields)
            item["updated_at"] = _utc_now_iso()
            self._save_state_locked()
            return dict(item)

    def _make_unique_slug(self, artist: str, title: str, spotify_track_id: str = "") -> str:
        base = slugify(f"{artist}-{title}" if artist and artist != "Unknown Artist" else title)
        with self._lock:
            tracks = self._state["tracks"]
            if spotify_track_id:
                for existing in tracks.values():
                    if existing.get("spotify_track_id") == spotify_track_id:
                        return existing["slug"]
            cand = base
            idx = 2
            while cand in tracks or (WORK_DIR / cand).exists():
                cand = f"{base}-{idx}"
                idx += 1
            return cand

    def sync_spotify_playlist(
        self,
        playlist_id_or_url: str | None = None,
        mode: str = "incremental",
        limit: int = 5,
        style: str = "minimal",
    ) -> dict[str, Any]:
        """Fetch playlist items from Spotify and enqueue new tracks according to `mode`."""
        with self._lock:
            sync_meta = self._state["spotify_sync"]
            target_pl = parse_spotify_playlist_id(playlist_id_or_url or sync_meta.get("playlist_id") or "")
            if not target_pl:
                target_pl = get_spotify_config().get("default_playlist_id", "")
            since_iso = sync_meta.get("last_synced_at") if mode == "incremental" else None

        res = fetch_playlist_tracks(target_pl, since_iso=since_iso)
        pl_info = res["playlist"]
        all_tracks = res["all_tracks"]
        max_added_at = res["max_added_at"]

        with self._lock:
            sync_meta = self._state["spotify_sync"]
            seen_ids = set(sync_meta.get("seen_track_ids") or [])
            for item in self._state["tracks"].values():
                if item.get("spotify_track_id"):
                    seen_ids.add(item["spotify_track_id"])

            if mode == "baseline_only":
                for tr in all_tracks:
                    seen_ids.add(tr["spotify_track_id"])
                sync_meta["playlist_id"] = pl_info["id"]
                sync_meta["playlist_name"] = pl_info["name"]
                sync_meta["last_synced_at"] = max_added_at or _utc_now_iso()
                sync_meta["last_checked_at"] = _utc_now_iso()
                sync_meta["last_error"] = None
                sync_meta["seen_track_ids"] = sorted(seen_ids)
                self._save_state_locked()
                self._log(f"Spotify 기준 시점 설정 완료: '{pl_info['name']}' ({len(all_tracks)}곡 스킵)")
                return {
                    "playlist": pl_info,
                    "mode": mode,
                    "enqueued": [],
                    "skipped_existing": len(all_tracks),
                    "last_synced_at": sync_meta["last_synced_at"],
                }

            candidates = res["new_tracks"] if mode == "incremental" else list(all_tracks)
            # Sort oldest -> newest by added_at so queue preserves playlist addition order
            candidates.sort(key=lambda x: x.get("added_at") or "")
            unseen = [tr for tr in candidates if tr["spotify_track_id"] not in seen_ids]
            if mode == "latest_n" and limit > 0:
                unseen = unseen[-limit:]

            enqueued: list[dict[str, Any]] = []
            for tr in unseen:
                slug = self._make_unique_slug(tr["artist"], tr["title"], tr["spotify_track_id"])
                lang = _guess_language(tr["title"], tr["artist"])
                record = {
                    "slug": slug,
                    "title": tr["title"],
                    "artist": tr["artist"],
                    "album": tr.get("album", ""),
                    "language": lang,
                    "style": style,
                    "duration": round(tr.get("duration_ms", 0) / 1000.0, 3),
                    "duration_ms": tr.get("duration_ms", 0),
                    "spotify_track_id": tr["spotify_track_id"],
                    "spotify_url": tr.get("spotify_url", ""),
                    "added_at": tr.get("added_at", ""),
                    "cover_url": tr.get("cover_url", ""),
                    "source": "spotify",
                    "status": "queued_prepare",
                    "step_message": "자동 준비 대기 중",
                    "error": None,
                    "upload_requested": True,
                    "output_path": "",
                    "youtube": None,
                    "created_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                }
                self._state["tracks"][slug] = record
                seen_ids.add(tr["spotify_track_id"])
                enqueued.append(record)

            sync_meta["playlist_id"] = pl_info["id"]
            sync_meta["playlist_name"] = pl_info["name"]
            if max_added_at:
                sync_meta["last_synced_at"] = max_added_at
            elif not sync_meta.get("last_synced_at"):
                sync_meta["last_synced_at"] = _utc_now_iso()
            sync_meta["last_checked_at"] = _utc_now_iso()
            sync_meta["last_error"] = None
            sync_meta["seen_track_ids"] = sorted(seen_ids)
            self._save_state_locked()

        if enqueued:
            self._log(f"Spotify 동기화: '{pl_info['name']}'에서 새 곡 {len(enqueued)}개 대기열 추가")
            self._prepare_event.set()
        else:
            self._log(f"Spotify 동기화: '{pl_info['name']}' 새 곡 없음 (총 {res['total_items']}곡)")

        return {
            "playlist": pl_info,
            "mode": mode,
            "enqueued": enqueued,
            "total_playlist_items": res["total_items"],
            "last_synced_at": sync_meta["last_synced_at"],
        }

    def enqueue_manual_track(
        self,
        title: str,
        artist: str = "Unknown Artist",
        source_url: str | None = None,
        language: str | None = None,
        style: str = "minimal",
        upload_to_youtube: bool = True,
    ) -> dict[str, Any]:
        """Queue a manual track (URL or local input search) for background preparation."""
        clean_title = title.strip() or "Untitled Track"
        clean_artist = artist.strip() or "Unknown Artist"
        lang = language or _guess_language(clean_title, clean_artist)
        slug = self._make_unique_slug(clean_artist, clean_title)

        with self._lock:
            record = {
                "slug": slug,
                "title": clean_title,
                "artist": clean_artist,
                "album": "",
                "language": lang,
                "style": style,
                "duration": 0,
                "duration_ms": 0,
                "spotify_track_id": "",
                "spotify_url": "",
                "added_at": _utc_now_iso(),
                "cover_url": "",
                "source": "manual_url" if source_url else "manual",
                "source_url": source_url or "",
                "status": "queued_prepare",
                "step_message": "자동 준비 대기 중",
                "error": None,
                "upload_requested": bool(upload_to_youtube),
                "output_path": "",
                "youtube": None,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
            self._state["tracks"][slug] = record
            self._save_state_locked()

        self._log(f"수동 곡 대기열 추가: {clean_artist} - {clean_title}", slug=slug)
        self._prepare_event.set()
        return record

    def approve_track(
        self,
        slug: str,
        style: str | None = None,
        upload_to_youtube: bool = True,
    ) -> dict[str, Any]:
        """Mark a reviewed track as approved and enqueue it for rendering + YouTube upload."""
        with self._lock:
            if slug not in self._state["tracks"]:
                self.sync_existing_work_dirs()
            item = self._state["tracks"].get(slug)
            if not item:
                raise KeyError(f"Unknown track slug: {slug}")
            if item["status"] not in {"needs_review", "rendered", "uploaded"}:
                raise ValueError("검수나 렌더를 마친 곡만 대기열에 추가할 수 있습니다.")
            chosen_style = style or item.get("style") or "minimal"
            style_changed = chosen_style != (item.get("style") or "minimal")
            item["style"] = chosen_style
            item["upload_requested"] = bool(upload_to_youtube)

            existing_mp4 = Path(item["output_path"]) if item.get("output_path") else (OUTPUT_DIR / f"{slug}_{chosen_style}.mp4")
            if upload_to_youtube and not style_changed and existing_mp4.exists() and not item.get("youtube") and not item.get("output_is_clip"):
                item["status"] = "queued_upload"
                item["output_path"] = str(existing_mp4)
                item["step_message"] = "유튜브 비공개 업로드 대기열 추가됨"
            else:
                item["status"] = "queued_render"
                item["step_message"] = "렌더링 대기열 추가됨"
            item["render_options"] = {}
            item["stage"] = QUEUE_STEPS[item["status"]]
            item["error"] = None
            item["updated_at"] = _utc_now_iso()
            self._save_state_locked()
            snapshot = dict(item)

        self._log("검수 완료 → 렌더링/업로드 대기열 추가", slug=slug)
        self._render_event.set()
        return snapshot

    def enqueue_render(
        self, slug: str, style: str | None = None, output: Path | None = None,
        clip_start: float | None = None, clip_end: float | None = None,
        score: dict[str, Any] | None = None, timeline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.sync_existing_work_dirs()
        with self._lock:
            item = self._state["tracks"].get(slug)
            if not item:
                raise KeyError(f"Unknown track slug: {slug}")
            if item["status"] not in {"needs_review", "rendered", "uploaded"}:
                raise ValueError("현재 곡에 진행 중이거나 대기 중인 작업이 있습니다.")
            chosen_style = style or item.get("style") or "minimal"
            if chosen_style not in STYLE_TO_COMP:
                raise ValueError(f"지원하지 않는 영상 스타일입니다: {chosen_style}")
            if score and timeline:
                write_json(WORK_DIR / slug / "score.json", score)
                write_json(WORK_DIR / slug / "timeline.json", timeline)
            result = self._update_track(
                slug, status="queued_render", stage="render", style=chosen_style,
                upload_requested=False, step_message="MP4 렌더링 대기", error=None,
                render_options={"output": str(output) if output else None,
                                "clip_start": clip_start, "clip_end": clip_end},
            )
        self._render_event.set()
        return result

    def control_track(self, slug: str, action: str) -> dict[str, Any]:
        """Pause, resume or cancel only the selected track."""
        with self._lock:
            item = self._state["tracks"].get(slug)
            if not item:
                raise KeyError(f"Unknown track slug: {slug}")
            status = item["status"]
            stage = item.get("stage") or QUEUE_STEPS.get(status, "source")
            label = _step_label(stage)
            if action == "resume" and status == "paused":
                queued = _queue_for_step(stage)
                updates = {"status": queued, "step_message": f"{label} · 재개 대기", "error": None}
            elif action in {"pause", "cancel"} and status in QUEUED_TO_ACTIVE:
                updates = {
                    "status": "paused" if action == "pause" else "cancelled",
                    "stage": stage,
                    "step_message": f"{label} · " + ("일시정지됨" if action == "pause" else "중단됨"),
                }
            elif action in {"pause", "cancel"} and status in ACTIVE_TO_QUEUED and slug in self._active_jobs:
                if item.get("pending_action") == "cancel":
                    raise ValueError("작업을 중단 중입니다.")
                updates = {
                    "pending_action": action,
                    "step_message": f"{label} · " + ("일시정지 처리 중" if action == "pause" else "중단 처리 중"),
                }
                self._active_jobs[slug].set()
            elif action == "cancel" and status == "paused":
                updates = {"status": "cancelled", "step_message": f"{label} · 중단됨"}
            else:
                raise ValueError("현재 상태에서 이 작업을 실행할 수 없습니다.")
            snapshot = self._update_track(slug, **updates)
            self._prepare_event.set()
            self._render_event.set()
            return snapshot

    def retry_track(self, slug: str) -> dict[str, Any]:
        """Requeue a failed or cancelled track at its first incomplete step."""
        with self._lock:
            item = self._state["tracks"].get(slug)
            if not item:
                raise KeyError(f"Unknown track slug: {slug}")
            if item["status"] not in {"failed", "cancelled"}:
                raise ValueError("실패했거나 중단된 곡만 다시 실행할 수 있습니다.")
            stage = item.get("stage")
            if not stage:
                # Older records only retained descriptive status messages.
                message = item.get("step_message", "")
                if message == "렌더/업로드 실패":
                    output = item.get("output_path")
                    stage = "upload" if output and Path(output).exists() and item.get("upload_requested") else "render"
                else:
                    stage = "source"
            result = self._update_track(
                slug, status=_queue_for_step(stage), stage=stage, error=None, step_message="재시도 대기",
            )
        self._prepare_event.set()
        self._render_event.set()
        return result

    def enrich_track_with_acp(self, slug: str, model_id: str | None = None) -> dict[str, Any]:
        """Queue a new annotation pass for a prepared track."""
        chosen_model = model_id or get_env("AGY_ACP_MODEL", DEFAULT_ACP_MODEL) or DEFAULT_ACP_MODEL
        with self._lock:
            item = self._state["tracks"].get(slug)
            if not item or not (WORK_DIR / slug).is_dir():
                raise FileNotFoundError(f"Track work directory not found: {slug}")
            if item["status"] not in {"needs_review", "rendered", "uploaded"}:
                raise ValueError("검수 또는 렌더를 마친 곡에만 주석을 다시 요청할 수 있습니다.")
            self._update_track(
                slug, status="queued_prepare", stage="refresh_notes", notes_return_status=item["status"],
                notes_model=chosen_model, step_message="ACP 주석 생성 대기", error=None,
            )
        self._prepare_event.set()
        return {"slug": slug, "model": chosen_model}

    @staticmethod
    def _remove_path_with_retry(path: Path, is_dir: bool) -> None:
        last_exc: PermissionError | None = None
        for attempt in range(3):
            try:
                if is_dir:
                    shutil.rmtree(path)
                else:
                    path.unlink()
                return
            except FileNotFoundError:
                return
            except PermissionError as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.15)
        if last_exc is not None:
            raise last_exc

    def delete_track(self, slug: str, delete_output: bool = False) -> dict[str, Any]:
        """Safely delete a project work directory, queue entry, and optional rendered MP4s."""
        raw = str(slug or "").strip()
        if not raw or raw.startswith(".") or "/" in raw or "\\" in raw or Path(raw).name != raw:
            raise ValueError(f"유효하지 않은 프로젝트 이름입니다: {slug}")

        work_root = WORK_DIR.resolve()
        work_dir = (WORK_DIR / raw).resolve()
        if work_dir.parent != work_root:
            raise ValueError(f"유효하지 않은 프로젝트 경로입니다: {slug}")

        deleted_work_dir = False
        deleted_outputs: list[str] = []

        with self._lock:
            item = self._state["tracks"].get(raw)
            if item is None and not work_dir.exists():
                raise FileNotFoundError(f"프로젝트를 찾을 수 없습니다: {raw}")

            if (item and item.get("status") in {"preparing", "rendering", "uploading"}):
                raise RuntimeError("현재 준비·렌더링·업로드·주석 생성이 진행 중인 프로젝트는 삭제할 수 없습니다.")

            if work_dir.exists():
                self._remove_path_with_retry(work_dir, is_dir=True)
                deleted_work_dir = True

            if delete_output:
                out_root = OUTPUT_DIR.resolve()
                seen_paths: set[Path] = set()
                candidates: list[Path] = []
                if item and item.get("output_path"):
                    candidates.append(Path(str(item["output_path"])))
                if OUTPUT_DIR.exists():
                    candidates.extend(sorted(OUTPUT_DIR.glob(f"{raw}_*.mp4")))

                for cand in candidates:
                    try:
                        resolved = cand.resolve()
                    except OSError:
                        continue
                    if resolved in seen_paths or resolved.parent != out_root or not resolved.is_file():
                        continue
                    seen_paths.add(resolved)
                    self._remove_path_with_retry(resolved, is_dir=False)
                    deleted_outputs.append(resolved.name)

            self._state["tracks"].pop(raw, None)
            self._save_state_locked()

        self._log(
            f"프로젝트 삭제 완료 (work={deleted_work_dir}, mp4={len(deleted_outputs)}개)",
            slug=raw,
        )
        return {
            "slug": raw,
            "deleted_work_dir": deleted_work_dir,
            "deleted_outputs": deleted_outputs,
        }

    def mark_track_rendered(self, slug: str, output_path: Path, style: str | None = None) -> None:
        """Update queue state when a user triggers an immediate render inside the studio."""
        with self._lock:
            item = self._state["tracks"].get(slug)
            if not item:
                return
            if style:
                item["style"] = style
            item["output_path"] = str(output_path)
            if item.get("status") in {"needs_review", "failed", "queued_prepare", "queued_render"}:
                item["status"] = "rendered"
                item["step_message"] = f"MP4 렌더링 완료 ({output_path.name})"
                item["error"] = None
                item["updated_at"] = _utc_now_iso()
                self._save_state_locked()

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        self.sync_existing_work_dirs()
        with self._lock:
            tracks_list = [dict(item) for item in self._state["tracks"].values()]
            tracks_list.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
            return {
                "spotify_sync": dict(self._state["spotify_sync"]),
                "tracks": tracks_list,
                "logs": list(self._logs[-40:]),
            }

    def _prepare_worker_loop(self) -> None:
        self._worker_loop(("queued_prepare",), self._prepare_event)

    def _render_worker_loop(self) -> None:
        self._worker_loop(("queued_render", "queued_upload"), self._render_event)

    def _worker_loop(self, statuses: tuple[str, ...], wake: threading.Event) -> None:
        while not self._stop_event.is_set():
            wake.clear()
            job = self._claim_job(statuses)
            if job is None:
                wake.wait(timeout=5.0)
            else:
                self._run_job(*job)

    def _claim_job(self, statuses: tuple[str, ...]) -> tuple[dict[str, Any], threading.Event] | None:
        with self._lock:
            candidates = [item for item in self._state["tracks"].values() if item["status"] in statuses]
            if not candidates:
                return None
            item = min(candidates, key=lambda track: track.get("created_at") or "")
            queued = item["status"]
            event = threading.Event()
            snapshot = self._update_track(item["slug"], status=QUEUED_TO_ACTIVE[queued],
                                          stage=item.get("stage") or QUEUE_STEPS[queued], error=None)
            self._active_jobs[item["slug"]] = event
            self._log(f"{snapshot['stage']} 작업 시작", slug=item["slug"])
            return snapshot, event

    def _finish_job(self, slug: str, stage: str, interrupted: bool = False, **fields: Any) -> None:
        """Release the worker and commit the next stage or the requested stop under one lock."""
        with self._lock:
            item = self._state["tracks"][slug]
            action = item.pop("pending_action", None)
            if interrupted or fields.get("status") in QUEUED_TO_ACTIVE or fields.get("status") == "failed":
                if action in {"pause", "cancel"}:
                    stopped_message = "일시정지됨" if action == "pause" else "중단됨"
                    fields.update(status="paused" if action == "pause" else "cancelled",
                                  step_message=f"{_step_label(stage)} · {stopped_message}", error=None)
                elif interrupted:
                    fields.update(status=_queue_for_step(stage), step_message="작업 재개 대기")
            self._update_track(slug, stage=stage, **fields)
            self._active_jobs.pop(slug, None)
            if fields.get("status") != "failed":
                self._log(fields["step_message"], slug=slug)

    def _run_job(self, item: dict[str, Any], event: threading.Event) -> None:
        slug, stage = item["slug"], item["stage"]
        work_dir = WORK_DIR / slug

        def interrupted() -> bool:
            return event.is_set() or self._stop_event.is_set()

        try:
            if stage == "refresh_notes":
                task_path = work_dir / ".queue-task.json"
                write_json(task_path, item)
                self._update_track(slug, step_message="ACP 곡 해설 및 학습 노트 생성 중")
                run_cmd([sys.executable, "-X", "utf8", str(Path(__file__).with_name("queue_prepare.py")),
                         str(task_path), stage], interrupted=interrupted)
                self._finish_job(slug, stage, status=item["notes_return_status"], step_message="ACP 주석 완료")
                return

            if stage in PREPARE_STEPS:
                task_path = work_dir / ".queue-task.json"
                write_json(task_path, item)
                for step in PREPARE_SEQUENCE[PREPARE_SEQUENCE.index(stage):]:
                    # Commit the step pointer before starting its process.
                    with self._lock:
                        if interrupted():
                            raise JobInterrupted()
                        self._update_track(slug, stage=step, step_message=PREPARE_STEPS[step])
                    stage = step
                    proc = run_cmd([sys.executable, "-X", "utf8", str(Path(__file__).with_name("queue_prepare.py")),
                             str(task_path), step], interrupted=interrupted)
                    if step == "words" and proc is not None:
                        summary = proc.stdout.strip().splitlines()[-1:] or ["단어 정렬 완료"]
                        self._update_track(slug, word_alignment_summary=summary[0])
                        self._log(summary[0], slug=slug)
                    next_index = PREPARE_SEQUENCE.index(step) + 1
                    if next_index < len(PREPARE_SEQUENCE):
                        stage = PREPARE_SEQUENCE[next_index]
                        self._update_track(slug, stage=stage)
                meta = read_json(work_dir / "meta.json")
                with self._lock:
                    word_summary = self._state["tracks"][slug].get("word_alignment_summary")
                message = f"검수 대기 · {word_summary}" if word_summary else "검수 대기"
                self._finish_job(slug, stage, status="needs_review", duration=meta.get("duration", 0),
                                 step_message=message)
                return

            style = item.get("style") or "minimal"
            default_output = OUTPUT_DIR / f"{slug}_{style}.mp4"
            if stage == "render":
                self._update_track(slug, step_message=f"MP4 렌더링 중 ({style})")
                options = item.get("render_options") or {}
                clip_start, clip_end = options.get("clip_start"), options.get("clip_end")
                is_clip = clip_start is not None or clip_end is not None
                output = Path(options["output"]) if options.get("output") else None
                output = render_work_dir(work_dir, style=style, output=output,
                                         clip_start=clip_start, clip_end=clip_end, interrupted=interrupted)
                next_stage = "upload" if item.get("upload_requested") else "render"
                self._finish_job(slug, next_stage, output_path=str(output), output_is_clip=is_clip,
                                 status="queued_upload" if item.get("upload_requested") else "rendered",
                                 step_message="업로드 대기" if item.get("upload_requested") else "MP4 렌더링 완료")
                self._render_event.set()
                return

            if stage != "upload":
                raise ValueError(f"Unknown work stage: {stage}")
            if interrupted():
                raise JobInterrupted()
            yt = get_youtube_status()
            if not yt["configured"] or not yt["connected"]:
                self._finish_job(slug, stage, status="rendered",
                                 step_message="렌더 완료 · YouTube OAuth 연결 필요")
                return
            output = Path(item.get("output_path") or default_output)
            self._update_track(slug, step_message="YouTube 비공개 업로드 중")
            result = upload_private_video(work_dir, output, interrupted=interrupted)
            self._finish_job(slug, stage, status="uploaded", youtube=result,
                             step_message="YouTube 비공개 업로드 완료")
        except JobInterrupted:
            self._finish_job(slug, stage, interrupted=True)
        except Exception as exc:
            self._finish_job(slug, stage, status="failed", step_message="작업 실패",
                             error=f"{exc}\n{traceback.format_exc(limit=3)}")
            self._log(f"{stage} 실패: {exc}", slug=slug, level="error")
