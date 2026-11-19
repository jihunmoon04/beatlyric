"""CLI orchestrator for prepare, score quantization, interactive edit studio, and render."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path

from common import WORK_DIR, slugify
from render_lib import STYLE_TO_COMP, render_work_dir, resolve_work_dir
from step1_ingest import run_ingest
from step2_align import run_align
from step3_spectrum import extract_spectrum
from step4_score import ensure_score_file
from step5_enrich import (
    DEFAULT_ACP_MODEL,
    apply_annotations_to_work_dir,
    enrich_with_agy_acp,
    enrich_with_gemini_api,
    export_llm_prompt_bundle,
    is_agy_acp_available,
    resolve_gemini_api_key,
)
from studio_server import serve_forever
from studio_state import StudioQueueManager
from word_alignment import run_automatic_word_alignment


def cmd_prepare(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    slug = slugify(args.name or input_path.stem)
    work_dir = WORK_DIR / slug

    print(f"[1/5] Ingesting '{input_path.name}' -> {work_dir}")
    meta = run_ingest(
        input_path=input_path,
        work_dir=work_dir,
        fps=args.fps,
        style=args.style,
        language=args.lang,
    )

    lyrics_path = Path(args.lyrics).resolve() if args.lyrics else None
    print(f"[2/5] Aligning lyrics (lang={args.lang}, demucs={not args.no_demucs}, aligner={args.aligner})")
    llm_payload = run_align(
        work_dir=work_dir,
        meta=meta,
        lyrics_file=lyrics_path,
        use_demucs=not args.no_demucs,
        whisper_model=args.whisper_model,
        force_neural_align=args.force_align,
        aligner=args.aligner,
    )

    print(f"[3/5] Extracting audio spectrum ({args.fps} fps, 28 bars)")
    extract_spectrum(
        wav_path=work_dir / "analysis.wav",
        out_path=work_dir / "spectrum.json",
        fps=args.fps,
        num_bars=28,
    )

    print("[4/5] Estimating BPM and fixing lyric line boundaries")
    score = ensure_score_file(work_dir, force=True)
    print("[5/5] Aligning words inside fixed lyric lines")
    run_automatic_word_alignment(work_dir, isolate_vocals=not args.no_demucs)
    export_llm_prompt_bundle(work_dir)
    grid = score["grid"]

    print("\n--- PREPARE COMPLETE ---")
    print(f"Track work dir : {work_dir}")
    print(f"Detected grid  : {grid['bpm']} BPM, t0={grid['t0']}s ({len(llm_payload['lines'])} lines)")
    print(f"LLM Prompt     : {work_dir / 'llm_prompt.md'}")
    print(f"Enrich step    : uv run python pipeline/run.py enrich {work_dir.name} --from-json <annotations.json>")
    print(f"Then tune/check: uv run python pipeline/run.py edit {work_dir.name}")
    print(f"Or render      : uv run python pipeline/run.py render {work_dir.name} --style {args.style}")


def cmd_enrich(args: argparse.Namespace) -> None:
    work_dir = resolve_work_dir(args.target)
    if args.from_json:
        ann_path = Path(args.from_json).resolve()
        res = apply_annotations_to_work_dir(work_dir, _load_json_file(ann_path))
        print(f"Merged {res['updated_lines']} annotated lines into {work_dir / 'score.json'} & timeline.json")
        return

    backend = args.backend
    api_key = resolve_gemini_api_key(args.api_key)

    if backend == "acp" or (backend == "auto" and is_agy_acp_available() and not args.api_key):
        print(f"[ACP] Running scoped `agy acp` ({args.acp_model}) inside {work_dir}...")
        res = enrich_with_agy_acp(
            work_dir,
            model_id=args.acp_model,
            session_timeout_sec=args.timeout,
        )
        print(f"Enriched {res['updated_lines']} lines via agy acp ({args.acp_model}) and closed session.")
        return

    if api_key and backend in {"auto", "gemini"}:
        print(f"[LLM] Calling Gemini ({args.model}) for pronunciation, translation & learning notes...")
        res = enrich_with_gemini_api(work_dir, api_key=api_key, model=args.model)
        print(f"Enriched {res['updated_lines']} lines into {work_dir / 'score.json'} & timeline.json")
        return

    prompt_path = export_llm_prompt_bundle(work_dir)
    print(
        f"No --from-json, agy CLI, or GEMINI_API_KEY available.\n"
        f"Exported Agent prompt bundle to: {prompt_path}\n"
        f"Ask your AI agent to fill the annotations JSON, then run:\n"
        f"  uv run python pipeline/run.py enrich {work_dir.name} --from-json <annotations.json>"
    )


def _load_json_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_auto(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    slug = slugify(args.name or input_path.stem)
    work_dir = WORK_DIR / slug

    print(f"[1/6] Ingesting '{input_path.name}' -> {work_dir}")
    meta = run_ingest(
        input_path=input_path,
        work_dir=work_dir,
        fps=args.fps,
        style=args.style,
        language=args.lang,
    )

    lyrics_path = Path(args.lyrics).resolve() if args.lyrics else None
    print(f"[2/6] Aligning lyrics (lang={args.lang}, demucs={not args.no_demucs}, aligner={args.aligner})")
    run_align(
        work_dir=work_dir,
        meta=meta,
        lyrics_file=lyrics_path,
        use_demucs=not args.no_demucs,
        whisper_model=args.whisper_model,
        force_neural_align=True,
        aligner=args.aligner,
    )

    print(f"[3/6] Extracting audio spectrum ({args.fps} fps, 28 bars)")
    extract_spectrum(
        wav_path=work_dir / "analysis.wav",
        out_path=work_dir / "spectrum.json",
        fps=args.fps,
        num_bars=28,
    )

    print("[4/6] Estimating BPM and fixing lyric line boundaries")
    ensure_score_file(work_dir, force=True)
    print("[5/6] Aligning words inside fixed lyric lines")
    run_automatic_word_alignment(work_dir, isolate_vocals=not args.no_demucs)
    export_llm_prompt_bundle(work_dir)

    print("[6/6] Linguistic enrichment (pronunciation / translation / learning comments)")
    api_key = resolve_gemini_api_key(args.api_key)
    if args.annotations:
        ann = _load_json_file(Path(args.annotations).resolve())
        res = apply_annotations_to_work_dir(work_dir, ann)
        print(f"      Applied {res['updated_lines']} annotated lines from '{args.annotations}'")
    elif is_agy_acp_available() and not args.api_key:
        res = enrich_with_agy_acp(work_dir, model_id=args.acp_model)
        print(f"      Enriched {res['updated_lines']} lines via scoped agy acp ({args.acp_model})")
    elif api_key:
        res = enrich_with_gemini_api(work_dir, api_key=api_key, model=args.model)
        print(f"      Enriched {res['updated_lines']} lines via Gemini ({args.model})")
    else:
        print(f"      No API key / annotations file — exported {work_dir / 'llm_prompt.md'} for agent enrichment.")

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    output_path = Path(args.output).resolve() if args.output else None
    serve_forever(
        work_dir=work_dir,
        host=args.host,
        port=args.port,
        style=args.style,
        output=output_path,
    )


def cmd_score(args: argparse.Namespace) -> None:
    work_dir = resolve_work_dir(args.target)
    score = ensure_score_file(work_dir, bpm=args.bpm, t0=args.t0, force=args.force)
    grid = score["grid"]
    print(f"Score saved to {work_dir / 'score.json'} (BPM={grid['bpm']}, t0={grid['t0']}s, lines={len(score['lines'])})")


def cmd_edit(args: argparse.Namespace) -> None:
    work_dir = resolve_work_dir(args.target)
    output_path = Path(args.output).resolve() if args.output else None
    serve_forever(
        work_dir=work_dir,
        host=args.host,
        port=args.port,
        style=args.style,
        output=output_path,
    )


def cmd_studio(args: argparse.Namespace) -> None:
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve_forever(
        work_dir=None,
        host=args.host,
        port=args.port,
        style=args.style,
        output=None,
    )


def cmd_render(args: argparse.Namespace) -> None:
    work_dir = resolve_work_dir(args.target)
    output_path = Path(args.output).resolve() if args.output else None
    final_mp4 = render_work_dir(
        work_dir=work_dir,
        style=args.style,
        output=output_path,
        crf=args.crf,
        gl=args.gl,
        keep_silent=args.keep_silent,
        clip_start=args.clip_start,
        clip_end=args.clip_end,
    )
    print(f"\nDONE! Final video saved to: {final_mp4}")


def cmd_delete(args: argparse.Namespace) -> None:
    raw = str(args.target).strip().rstrip("/\\")
    cand_path = Path(raw)
    slug = cand_path.name if (cand_path.is_dir() or "/" in raw or "\\" in raw) else resolve_work_dir(raw).name
    if not args.yes:
        ans = input(f"Delete project '{slug}'? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Cancelled.")
            return
    res = StudioQueueManager().delete_track(slug, delete_output=bool(args.delete_output))
    outputs = ", ".join(res["deleted_outputs"]) if res["deleted_outputs"] else "none"
    print(f"Deleted project '{res['slug']}' (work_dir_removed={res['deleted_work_dir']}, outputs_removed={outputs})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lyric Study Video Automation Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Ingest audio/video, align lyrics, extract spectrum, build beat score")
    p_prep.add_argument("input", help="Path to audio (.mp3/.flac/.wav) or video (.mp4/.mkv)")
    p_prep.add_argument("--lyrics", help="Optional path to plain .txt or timed .lrc lyrics")
    p_prep.add_argument("--style", choices=list(STYLE_TO_COMP), default="minimal")
    p_prep.add_argument("--lang", default="en", help="Original song language code (e.g., en, ja, es)")
    p_prep.add_argument("--fps", type=int, default=30)
    p_prep.add_argument("--name", help="Optional custom work directory slug")
    p_prep.add_argument("--no-demucs", action="store_true", help="Skip Demucs vocal isolation")
    p_prep.add_argument("--whisper-model", default="medium", help="Whisper model size for stable-ts")
    p_prep.add_argument("--force-align", action="store_true", help="Run neural alignment even if LRC exists")
    p_prep.add_argument("--aligner", choices=["ctc", "stable-ts"], default="ctc", help="Neural aligner backend")
    p_prep.set_defaults(func=cmd_prepare)

    p_score = sub.add_parser("score", help="Estimate BPM/t0 and quantize timeline into 0.5-beat score.json")
    p_score.add_argument("target", help="Work directory path (work/<slug>) or track slug")
    p_score.add_argument("--bpm", type=float, help="Override detected BPM")
    p_score.add_argument("--t0", type=float, help="Override detected Bar 1 Beat 1 offset (seconds)")
    p_score.add_argument("--force", action="store_true", help="Overwrite existing score.json")
    p_score.set_defaults(func=cmd_score)

    p_edit = sub.add_parser("edit", help="Launch interactive Beat Score Studio UI in browser")
    p_edit.add_argument("target", help="Work directory path (work/<slug>) or track slug")
    p_edit.add_argument("--host", default="127.0.0.1")
    p_edit.add_argument("--port", type=int, default=8765)
    p_edit.add_argument("--style", choices=list(STYLE_TO_COMP), help="Default template style for render button")
    p_edit.add_argument("--output", help="Default output .mp4 path when rendering from studio")
    p_edit.set_defaults(func=cmd_edit)

    p_studio = sub.add_parser("studio", help="Launch Multi-Track Automation Dashboard & Beat Score Studio")
    p_studio.add_argument("--host", default="127.0.0.1")
    p_studio.add_argument("--port", type=int, default=8765)
    p_studio.add_argument("--style", choices=list(STYLE_TO_COMP), default="minimal", help="Default template style")
    p_studio.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    p_studio.set_defaults(func=cmd_studio)

    p_ren = sub.add_parser("render", help="Render timeline.json via Remotion and mux master audio")
    p_ren.add_argument("target", help="Work directory path (work/<slug>) or track slug")
    p_ren.add_argument("--style", choices=list(STYLE_TO_COMP), help="Override template style")
    p_ren.add_argument("--output", help="Custom output .mp4 path")
    p_ren.add_argument("--crf", type=int, default=18, help="Video quality CRF (lower = higher quality)")
    p_ren.add_argument("--gl", help="Remotion Chromium GL backend (e.g., swiftshader on headless Linux)")
    p_ren.add_argument("--keep-silent", action="store_true", help="Keep intermediate silent MP4")
    p_ren.add_argument("--clip-start", type=float, default=None, help="Optional clip start time in seconds")
    p_ren.add_argument("--clip-end", type=float, default=None, help="Optional clip end time in seconds")
    p_ren.set_defaults(func=cmd_render)

    p_enr = sub.add_parser("enrich", help="Merge LLM annotations JSON or auto-generate via agy acp / Gemini API")
    p_enr.add_argument("target", help="Work directory path (work/<slug>) or track slug")
    p_enr.add_argument("--from-json", help="Path to annotations JSON produced by AI agent")
    p_enr.add_argument("--backend", choices=["auto", "acp", "gemini"], default="auto", help="Enrichment backend")
    p_enr.add_argument("--acp-model", default=DEFAULT_ACP_MODEL, help="Model ID for agy acp")
    p_enr.add_argument("--timeout", type=float, default=240.0, help="Hard timeout (seconds) before terminating agy acp session")
    p_enr.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    p_enr.add_argument("--model", default="gemini-2.5-flash", help="Gemini API model name")
    p_enr.set_defaults(func=cmd_enrich)

    p_auto = sub.add_parser("auto", help="End-to-end: prepare + vocal beat score + LLM enrich + launch Studio")
    p_auto.add_argument("input", help="Path to audio (.mp3/.flac/.wav) or video (.mp4/.mkv)")
    p_auto.add_argument("--lyrics", help="Optional path to plain .txt or timed .lrc lyrics")
    p_auto.add_argument("--annotations", help="Optional pre-generated LLM annotations JSON path")
    p_auto.add_argument("--acp-model", default=DEFAULT_ACP_MODEL, help="Model ID for agy acp")
    p_auto.add_argument("--api-key", help="Optional Gemini API key (or set GEMINI_API_KEY env var)")
    p_auto.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    p_auto.add_argument("--style", choices=list(STYLE_TO_COMP), default="minimal")
    p_auto.add_argument("--lang", default="en", help="Original song language code (e.g., en, ja, es)")
    p_auto.add_argument("--fps", type=int, default=30)
    p_auto.add_argument("--name", help="Optional custom work directory slug")
    p_auto.add_argument("--no-demucs", action="store_true", help="Skip Demucs vocal isolation")
    p_auto.add_argument("--whisper-model", default="medium", help="Whisper model size for stable-ts")
    p_auto.add_argument("--aligner", choices=["ctc", "stable-ts"], default="ctc")
    p_auto.add_argument("--host", default="127.0.0.1")
    p_auto.add_argument("--port", type=int, default=8765)
    p_auto.add_argument("--output", help="Default output .mp4 path when rendering from studio")
    p_auto.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    p_auto.set_defaults(func=cmd_auto)

    p_del = sub.add_parser("delete", help="Delete a project work directory and optional rendered MP4s")
    p_del.add_argument("target", help="Work directory path (work/<slug>) or track slug")
    p_del.add_argument("--delete-output", action="store_true", help="Also delete rendered output/<slug>_*.mp4 files")
    p_del.add_argument("-y", "--yes", action="store_true", help="Skip interactive confirmation prompt")
    p_del.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
