"""Persistent queue state and background workers for Beat Score Studio automation."""

from __future__ import annotations

import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_resolver import resolve_audio_for_track
from common import OUTPUT_DIR, WORK_DIR, read_json, slugify, write_json
from render_lib import render_work_dir
from spotify_sync import (
    fetch_playlist_tracks,
    get_spotify_config,
    parse_spotify_playlist_id,
)
from step1_ingest import run_ingest
from step2_align import run_align
from step3_spectrum import extract_spectrum
from step4_score import ensure_score_file
from step5_enrich import (
    DEFAULT_ACP_MODEL,
    enrich_with_agy_acp,
    enrich_with_gemini_api,
    export_llm_prompt_bundle,
    is_agy_acp_available,
    resolve_gemini_api_key,
)
from env_utils import get_env
from youtube_upload import get_youtube_status, upload_private_video


STATE_DIR = WORK_DIR / ".studio"
QUEUE_PATH = STATE_DIR / "queue.json"

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

            # Recover interrupted states from previous server shutdown
            recover_map = {
                "preparing": "queued_prepare",
                "rendering": "queued_render",
                "uploading": "queued_upload",
            }
            for item in tracks.values():
                st = item.get("status")
                if st in recover_map:
                    item["status"] = recover_map[st]
                    item["step_message"] = "서버 재시작 후 대기열 복구됨"
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
        t_prep.start()
        t_ren.start()
        self._prepare_event.set()
        self._render_event.set()

    def stop_workers(self) -> None:
        self._stop_event.set()
        self._prepare_event.set()
        self._render_event.set()

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
            chosen_style = style or item.get("style") or "minimal"
            style_changed = chosen_style != (item.get("style") or "minimal")
            item["style"] = chosen_style
            item["upload_requested"] = bool(upload_to_youtube)

            existing_mp4 = Path(item["output_path"]) if item.get("output_path") else (OUTPUT_DIR / f"{slug}_{chosen_style}.mp4")
            if upload_to_youtube and not style_changed and existing_mp4.exists() and not item.get("youtube"):
                item["status"] = "queued_upload"
                item["step_message"] = "유튜브 비공개 업로드 대기열 추가됨"
            else:
                item["status"] = "queued_render"
                item["step_message"] = "렌더링 대기열 추가됨"
            item["error"] = None
            item["updated_at"] = _utc_now_iso()
            self._save_state_locked()
            snapshot = dict(item)

        self._log("검수 완료 → 렌더링/업로드 대기열 추가", slug=slug)
        self._render_event.set()
        return snapshot

    def retry_track(self, slug: str) -> dict[str, Any]:
        """Retry a failed track from the failed stage or prepare stage."""
        with self._lock:
            if slug not in self._state["tracks"]:
                self.sync_existing_work_dirs()
            item = self._state["tracks"].get(slug)
            if not item:
                raise KeyError(f"Unknown track slug: {slug}")
            work_dir = WORK_DIR / slug
            has_score = (work_dir / "score.json").exists() and (work_dir / "master_audio.m4a").exists()
            has_mp4 = bool(item.get("output_path") and Path(item["output_path"]).exists())

            if has_mp4 and item.get("upload_requested") and not item.get("youtube"):
                next_status = "queued_upload"
                msg = "유튜브 업로드 재시도 대기"
            elif has_score:
                next_status = "queued_render"
                msg = "렌더링 재시도 대기"
            else:
                next_status = "queued_prepare"
                msg = "자동 준비 재시도 대기"

            item["status"] = next_status
            item["error"] = None
            item["step_message"] = msg
            item["updated_at"] = _utc_now_iso()
            self._save_state_locked()
            snapshot = dict(item)

        self._prepare_event.set()
        self._render_event.set()
        self._log(f"재시도 요청 ({snapshot['status']})", slug=slug)
        return snapshot

    def enrich_track_with_acp(self, slug: str, model_id: str | None = None) -> dict[str, Any]:
        """Run Step 5 enrichment via scoped `agy acp` in a background thread."""
        work_dir = WORK_DIR / slug
        if not work_dir.exists():
            raise FileNotFoundError(f"Track work directory not found: {slug}")
        chosen_model = model_id or get_env("AGY_ACP_MODEL", DEFAULT_ACP_MODEL) or DEFAULT_ACP_MODEL

        def _worker() -> None:
            try:
                self._update_track(slug, step_message=f"agy acp ({chosen_model}) 학습 노트 생성 중...")
                self._log(f"agy acp 학습 노트 생성 시작 ({chosen_model})", slug=slug)
                res = enrich_with_agy_acp(work_dir, model_id=chosen_model)
                self._update_track(
                    slug,
                    step_message=f"agy acp ({chosen_model}) {res['updated_lines']}줄 주석 완료",
                    error=None,
                )
                self._log(f"agy acp 주석 완료 ({res['updated_lines']}줄)", slug=slug)
            except Exception as exc:
                self._update_track(slug, step_message="agy acp 주석 실패", error=str(exc))
                self._log(f"agy acp 주석 실패: {exc}", slug=slug, level="error")

        threading.Thread(target=_worker, name=f"acp-enrich-{slug}", daemon=True).start()
        return {"slug": slug, "model": chosen_model}

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
            tracks_list = list(self._state["tracks"].values())
            tracks_list.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
            return {
                "spotify_sync": dict(self._state["spotify_sync"]),
                "tracks": tracks_list,
                "logs": list(self._logs[-40:]),
            }

    def _next_track_with_statuses(self, statuses: tuple[str, ...]) -> dict[str, Any] | None:
        with self._lock:
            candidates = [
                t for t in self._state["tracks"].values()
                if t.get("status") in statuses
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda x: x.get("created_at") or "")
            return dict(candidates[0])

    def _prepare_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._next_track_with_statuses(("queued_prepare",))
            if not item:
                self._prepare_event.clear()
                self._prepare_event.wait(timeout=5.0)
                continue
            self._run_prepare_job(item)

    def _run_prepare_job(self, item: dict[str, Any]) -> None:
        slug = item["slug"]
        work_dir = WORK_DIR / slug
        style = item.get("style") or "minimal"
        lang = item.get("language") or "en"

        try:
            self._update_track(slug, status="preparing", error=None, step_message="1/5 음원·커버 확보 중 (input/ 또는 yt-dlp)...")
            self._log(f"준비 시작: {item['artist']} - {item['title']}", slug=slug)

            resolved = resolve_audio_for_track(
                title=item["title"],
                artist=item["artist"],
                duration_ms=item.get("duration_ms") or None,
                slug=slug,
                cover_url=item.get("cover_url") or None,
                work_dir=work_dir,
                source_url=item.get("source_url") or None,
            )

            self._update_track(slug, step_message="2/5 마스터 오디오 & 배경 이미지 생성 중...")
            extra_meta = {
                "spotify_track_id": item.get("spotify_track_id", ""),
                "spotify_url": item.get("spotify_url", ""),
                "album": item.get("album", ""),
                "added_at": item.get("added_at", ""),
            }
            meta = run_ingest(
                input_path=resolved["audio_path"],
                work_dir=work_dir,
                fps=30,
                style=style,
                language=lang,
                title_override=item["title"],
                artist_override=item["artist"],
                cover_override_path=resolved.get("cover_path"),
                extra_meta=extra_meta,
            )

            self._update_track(slug, step_message="3/5 가사 검색 · Demucs 보컬 분리 · CTC 정렬 중...")
            run_align(
                work_dir=work_dir,
                meta=meta,
                lyrics_file=None,
                use_demucs=True,
                force_neural_align=True,
                aligner="ctc",
            )

            self._update_track(slug, step_message="4/5 스펙트럼 & 박자 악보(0.5박) 생성 중...")
            extract_spectrum(
                wav_path=work_dir / "analysis.wav",
                out_path=work_dir / "spectrum.json",
                fps=30,
                num_bars=28,
            )
            score = ensure_score_file(work_dir, force=True)
            export_llm_prompt_bundle(work_dir)

            api_key = resolve_gemini_api_key(None)
            acp_model = get_env("AGY_ACP_MODEL", DEFAULT_ACP_MODEL) or DEFAULT_ACP_MODEL
            enrich_note = "프롬프트 생성됨 (GEMINI_API_KEY / agy 미설정)"
            if is_agy_acp_available():
                self._update_track(slug, step_message=f"5/5 agy acp ({acp_model}) 발음·해석·학습 노트 생성 중...")
                enrich_with_agy_acp(work_dir, model_id=acp_model)
                enrich_note = f"agy acp ({acp_model}) 학습 노트 완료"
            elif api_key:
                self._update_track(slug, step_message="5/5 Gemini API 발음·해석·학습 노트 생성 중...")
                enrich_with_gemini_api(work_dir, api_key=api_key)
                enrich_note = "Gemini API 학습 노트 완료"

            self._update_track(
                slug,
                status="needs_review",
                duration=meta.get("duration", item.get("duration", 0)),
                step_message=f"검수 대기 (BPM {score['grid']['bpm']}, {len(score['lines'])}줄 · {enrich_note})",
                error=None,
            )
            self._log("준비 완료 → 검수 대기 목록 이동", slug=slug)
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            self._update_track(
                slug,
                status="failed",
                step_message="자동 준비 실패",
                error=f"{exc}\n{tb}",
            )
            self._log(f"준비 실패: {exc}", slug=slug, level="error")

    def _render_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._next_track_with_statuses(("queued_render", "queued_upload"))
            if not item:
                self._render_event.clear()
                self._render_event.wait(timeout=5.0)
                continue
            self._run_render_or_upload_job(item)

    def _run_render_or_upload_job(self, item: dict[str, Any]) -> None:
        slug = item["slug"]
        work_dir = WORK_DIR / slug
        style = item.get("style") or "minimal"

        try:
            out_path = Path(item["output_path"]) if item.get("output_path") else (OUTPUT_DIR / f"{slug}_{style}.mp4")

            if item["status"] == "queued_render":
                self._update_track(slug, status="rendering", error=None, step_message=f"Remotion 1080p MP4 렌더링 중 ({style})...")
                self._log(f"MP4 렌더링 시작 ({style})", slug=slug)
                out_path = render_work_dir(work_dir, style=style, output=out_path)
                self._update_track(slug, output_path=str(out_path))
                self._log(f"MP4 렌더링 완료: {out_path.name}", slug=slug)

            if item.get("upload_requested"):
                yt_status = get_youtube_status()
                if not yt_status["configured"] or not yt_status["connected"]:
                    msg = (
                        "렌더 완료 (YouTube client_secrets.json 미설정)"
                        if not yt_status["configured"]
                        else "렌더 완료 (YouTube OAuth 로그인 필요)"
                    )
                    self._update_track(
                        slug,
                        status="rendered",
                        step_message=msg,
                        error=None,
                    )
                    self._log(msg, slug=slug)
                    return

                self._update_track(slug, status="uploading", error=None, step_message="YouTube 비공개(private) 업로드 중...")
                self._log("YouTube 비공개 업로드 시작", slug=slug)
                yt_res = upload_private_video(work_dir, out_path)
                self._update_track(
                    slug,
                    status="uploaded",
                    youtube=yt_res,
                    step_message=f"YouTube 비공개 업로드 완료 ({yt_res['video_id']})",
                    error=None,
                )
                self._log(f"YouTube 업로드 완료: {yt_res['url']}", slug=slug)
            else:
                self._update_track(
                    slug,
                    status="rendered",
                    step_message=f"MP4 렌더링 완료 ({out_path.name})",
                    error=None,
                )
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            self._update_track(
                slug,
                status="failed",
                step_message="렌더/업로드 실패",
                error=f"{exc}\n{tb}",
            )
            self._log(f"렌더/업로드 실패: {exc}", slug=slug, level="error")
