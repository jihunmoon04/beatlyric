"""Step 5: LLM linguistic enrichment (pronunciation, Korean translation, learning comments).

All timing remains purely acoustic/musical (Demucs + CTC/Whisper + beat score).
The LLM intervenes ONLY in this step to populate linguistic annotations while preserving
bar/beat coordinates, word timings, and `whole_line` flags.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from acp_client import DEFAULT_ACP_MODEL, ScopedAgyAcpSession, find_agy_executable
from common import read_json, write_json
from env_utils import get_env
from step4_score import compile_score_to_timeline, ensure_score_file


def _normalize_annotation_map(
    annotations: list[dict[str, Any]] | dict[str, Any],
) -> dict[int, dict[str, Any]]:
    if isinstance(annotations, dict):
        items = annotations.get("lines") or annotations.get("annotations") or []
    else:
        items = annotations
    by_id: dict[int, dict[str, Any]] = {}
    for item in items:
        if "id" in item:
            by_id[int(item["id"])] = item
    return by_id


def _merge_line_annotations(lines: list[dict[str, Any]], by_id: dict[int, dict[str, Any]]) -> int:
    updated = 0
    for ln in lines:
        ann = by_id.get(int(ln["id"]))
        if not ann:
            continue
        if "pronunciation" in ann:
            ln["pronunciation"] = str(ann["pronunciation"] or "")
        if "translation" in ann:
            ln["translation"] = str(ann["translation"] or "")
        if "comment" in ann:
            ln["comment"] = ann["comment"]
        if "whole_line" in ann:
            ln["whole_line"] = bool(ann["whole_line"])
        updated += 1
    return updated


def apply_annotations_to_work_dir(
    work_dir: Path,
    annotations: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """Merge LLM linguistic annotations by line ID while keeping beat/word timings intact."""
    by_id = _normalize_annotation_map(annotations)
    score = ensure_score_file(work_dir)

    updated = _merge_line_annotations(score.get("lines", []), by_id)
    write_json(work_dir / "score.json", score)

    initial_path = work_dir / "score_initial.json"
    if initial_path.exists():
        initial_score = read_json(initial_path)
        _merge_line_annotations(initial_score.get("lines", []), by_id)
        write_json(initial_path, initial_score)

    compiled = compile_score_to_timeline(score)
    write_json(work_dir / "timeline.json", compiled)
    return {"updated_lines": updated, "timeline": compiled}


def _extract_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _build_enrichment_prompt(song_label: str, lang: str, payload_lines: list[dict[str, Any]]) -> str:
    return (
        f"You are creating Korean language-learning study notes for the song '{song_label}' (primary language: {lang}).\n"
        "Do NOT inspect external files or run shell commands. Return ONLY a raw JSON array (no markdown prose).\n"
        "For EVERY item in the input array, keep the exact same integer `id` and return a JSON array of objects with:\n"
        "- `id`: integer\n"
        "- `pronunciation`: natural Korean Hangul phonetic transcription matching how it is sung (empty string \"\" for instrumental markers like ♪ ...)\n"
        "- `translation`: natural idiomatic Korean translation (short Korean label for instrumental markers)\n"
        "- `comment`: either `null` OR an object `{ \"tag\": \"IDIOM\"|\"SLANG\"|\"CULTURE\"|\"GRAMMAR\", \"phrase\": \"short key expression\", \"meaning\": \"concise Korean meaning\", \"tip\": \"one-sentence Korean nuance/usage tip\" }`. Provide helpful `comment` objects on roughly 30-45% of notable lines and `null` on repetitive chorus lines or instrumentals.\n\n"
        f"Input lines JSON:\n{json.dumps(payload_lines, ensure_ascii=False)}"
    )


def enrich_with_gemini_api(
    work_dir: Path,
    api_key: str,
    model: str = "gemini-2.5-flash",
    batch_size: int = 25,
) -> dict[str, Any]:
    """Generate Hangul pronunciation, Korean translation, and learning notes via Gemini API."""
    score = ensure_score_file(work_dir)
    meta = score.get("meta", {})
    lines = score.get("lines", [])
    song_label = f"{meta.get('title', work_dir.name)} - {meta.get('artist', '')}".strip(" -")
    lang = meta.get("language", "en")

    collected: list[dict[str, Any]] = []
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )

    for start_idx in range(0, len(lines), batch_size):
        batch = lines[start_idx : start_idx + batch_size]
        payload_lines = [
            {"id": ln["id"], "original": ln["original"]}
            for ln in batch
        ]
        prompt = _build_enrichment_prompt(song_label, lang, payload_lines)
        req_body = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error ({exc.code}): {err_body}") from exc

        text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = _extract_json_payload(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("lines") or parsed.get("annotations") or []
        if isinstance(parsed, list):
            collected.extend(parsed)

    return apply_annotations_to_work_dir(work_dir, collected)


def export_llm_prompt_bundle(work_dir: Path) -> Path:
    """Write a ready-to-paste Agent prompt file work/<slug>/llm_prompt.md for key-less workflows."""
    score = ensure_score_file(work_dir)
    meta = score.get("meta", {})
    raw_items = [
        {"id": ln["id"], "original": ln["original"]}
        for ln in score.get("lines", [])
    ]
    out_path = work_dir / "llm_prompt.md"
    content = (
        f"# LLM 가사 학습 주석 생성 프롬프트 ({meta.get('title', work_dir.name)})\n\n"
        "AI 에이전트(채팅창) 또는 외부 LLM에게 아래 JSON을 넘겨\n"
        "`pronunciation`, `translation`, `comment`가 채워진 JSON을 받은 뒤 다음 명령으로 적용하세요:\n\n"
        "```bash\n"
        f"uv run python pipeline/run.py enrich {work_dir.name} --from-json annotations.json\n"
        "```\n\n"
        "*(타이밍·박자·단어 오프셋은 `score.json`의 값이 그대로 보존되며 LLM이 건드리지 않습니다.)*\n\n"
        "## 요청 프롬프트 & 입력 JSON\n\n"
        "각 항목의 `id`를 유지한 JSON 배열(`[{\"id\": 0, \"pronunciation\": \"...\", \"translation\": \"...\", \"comment\": null | {\"tag\": \"IDIOM\", \"phrase\": \"...\", \"meaning\": \"...\", \"tip\": \"...\"}}]`)로 응답해 주세요:\n\n"
        "```json\n"
        f"{json.dumps(raw_items, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )
    out_path.write_text(content, encoding="utf-8")
    return out_path


def enrich_with_agy_acp(
    work_dir: Path,
    model_id: str | None = None,
    batch_size: int = 30,
    session_timeout_sec: float = 240.0,
) -> dict[str, Any]:
    """Generate linguistic annotations via scoped `agy acp` inside `work_dir`."""
    score = ensure_score_file(work_dir)
    meta = score.get("meta", {})
    lines = score.get("lines", [])
    song_label = f"{meta.get('title', work_dir.name)} - {meta.get('artist', '')}".strip(" -")
    lang = meta.get("language", "en")
    chosen_model = model_id or get_env("AGY_ACP_MODEL", DEFAULT_ACP_MODEL) or DEFAULT_ACP_MODEL
    timeout_val = float(get_env("AGY_ACP_TIMEOUT_SEC", str(session_timeout_sec)) or session_timeout_sec)

    collected: list[dict[str, Any]] = []
    with ScopedAgyAcpSession(
        work_dir=work_dir,
        model_id=chosen_model,
        session_timeout_sec=timeout_val,
        allow_work_dir_writes=False,
    ) as session:
        for start_idx in range(0, len(lines), batch_size):
            batch = lines[start_idx : start_idx + batch_size]
            payload_lines = [
                {"id": ln["id"], "original": ln["original"]}
                for ln in batch
            ]
            prompt = _build_enrichment_prompt(song_label, lang, payload_lines)
            reply_text = session.prompt_text(prompt, timeout_sec=min(120.0, timeout_val))
            parsed = _extract_json_payload(reply_text)
            if isinstance(parsed, dict):
                parsed = parsed.get("lines") or parsed.get("annotations") or []
            if isinstance(parsed, list):
                collected.extend(parsed)

    write_json(work_dir / "annotations_acp.json", collected)
    return apply_annotations_to_work_dir(work_dir, collected)


def resolve_gemini_api_key(explicit_key: str | None = None) -> str | None:
    return explicit_key or get_env("GEMINI_API_KEY") or get_env("GOOGLE_API_KEY") or None


def is_agy_acp_available() -> bool:
    return find_agy_executable() is not None
