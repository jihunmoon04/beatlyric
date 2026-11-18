"""Step 5: LLM linguistic enrichment (pronunciation, Korean translation, learning comments).

All timing remains purely acoustic/musical (Demucs + CTC/Whisper + beat score).
The LLM intervenes ONLY in this step to populate linguistic annotations while preserving
bar/beat coordinates, word timings, and `whole_line` flags.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import read_json, write_json
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
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


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
        prompt = (
            f"You are creating Korean language-learning study notes for the song '{song_label}' (primary language: {lang}).\n"
            "For EVERY item in the input array, keep the exact same integer `id` and return a JSON array of objects with:\n"
            "- `id`: integer\n"
            "- `pronunciation`: natural Korean Hangul phonetic transcription matching how it is sung (empty string \"\" for instrumental markers like ♪ ...)\n"
            "- `translation`: natural idiomatic Korean translation (short Korean label for instrumental markers)\n"
            "- `comment`: either `null` OR an object `{ \"tag\": \"IDIOM\"|\"SLANG\"|\"CULTURE\"|\"GRAMMAR\", \"phrase\": \"short key expression\", \"meaning\": \"concise Korean meaning\", \"tip\": \"one-sentence Korean nuance/usage tip\" }`. Provide helpful `comment` objects on roughly 30-45% of notable lines and `null` on repetitive chorus lines or instrumentals.\n\n"
            f"Input lines JSON:\n{json.dumps(payload_lines, ensure_ascii=False)}"
        )
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


def resolve_gemini_api_key(explicit_key: str | None = None) -> str | None:
    return explicit_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
