"""Run one restartable preparation step in a child process owned by the queue."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from audio_resolver import resolve_audio_for_track
from common import read_json, write_json
from env_utils import get_env
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


from word_alignment import run_automatic_word_alignment


PREPARE_STEPS = {
    "source": "1/6 음원·커버 확보 중",
    "ingest": "2/6 마스터 오디오와 배경 생성 중",
    "align": "3/6 가사 검색, 보컬 분리 및 초벌 정렬 중",
    "score": "4/6 스펙트럼과 가사 줄 경계 생성 중",
    "words": "5/6 확정된 줄 안에서 단어를 보컬에 맞추는 중",
    "enrich": "6/6 발음·번역·곡 해설 생성 중",
}


def run_step(work_dir: Path, step: str, item: dict[str, Any]) -> None:
    source_path = work_dir / ".queue-source.json"
    if step == "source":
        source = resolve_audio_for_track(
            title=item["title"],
            artist=item["artist"],
            duration_ms=item.get("duration_ms") or None,
            slug=item["slug"],
            cover_url=item.get("cover_url") or None,
            work_dir=work_dir,
            source_url=item.get("source_url") or None,
        )
        write_json(source_path, {key: str(value) if isinstance(value, Path) else value
                                for key, value in source.items()})
    elif step == "ingest":
        source = read_json(source_path)
        cover = source.get("cover_path")
        run_ingest(
            input_path=Path(source["audio_path"]),
            work_dir=work_dir,
            fps=30,
            style=item.get("style") or "minimal",
            language=item.get("language") or "en",
            title_override=item["title"],
            artist_override=item["artist"],
            cover_override_path=Path(cover) if cover else None,
            extra_meta={key: item.get(key, "") for key in
                        ("spotify_track_id", "spotify_url", "album", "added_at")},
        )
    elif step == "align":
        run_align(
            work_dir=work_dir, meta=read_json(work_dir / "meta.json"), lyrics_file=None,
            use_demucs=True, force_neural_align=True, aligner="ctc",
        )
    elif step == "score":
        extract_spectrum(
            wav_path=work_dir / "analysis.wav", out_path=work_dir / "spectrum.json",
            fps=30, num_bars=28,
        )
        ensure_score_file(work_dir, force=True)
    elif step == "words":
        run_automatic_word_alignment(work_dir)
    elif step == "refresh_notes":
        enrich_with_agy_acp(work_dir, model_id=item["notes_model"])
    elif step == "enrich":
        export_llm_prompt_bundle(work_dir)
        api_key = resolve_gemini_api_key(None)
        if is_agy_acp_available():
            enrich_with_agy_acp(work_dir, model_id=get_env("AGY_ACP_MODEL", DEFAULT_ACP_MODEL))
        elif api_key:
            enrich_with_gemini_api(work_dir, api_key=api_key)
    else:
        raise ValueError(f"Unknown prepare step: {step}")


if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    run_step(input_path.parent, sys.argv[2], read_json(input_path))
