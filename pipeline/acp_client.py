"""Scoped Agent Client Protocol (ACP) stdio client for `agy acp`.

Runs `agy acp --sandbox` strictly inside a single target working directory,
pins the requested model (default: `gemini-4-hydrogen-medium`), enforces a
hard wall-clock timeout, and guarantees `session/close` + process termination
on completion, error, or timeout.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_ACP_MODEL = "gemini-4-hydrogen-medium"


def find_agy_executable() -> str | None:
    """Locate `agy` binary in PATH or standard Windows/Unix install locations."""
    env_override = os.environ.get("AGY_PATH", "").strip()
    if env_override and Path(env_override).is_file():
        return env_override

    in_path = shutil.which("agy")
    if in_path:
        return in_path

    candidates = [
        Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe",
        Path.home() / ".local" / "bin" / "agy",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def _is_path_within(candidate: str, root_dir: Path) -> bool:
    try:
        p = Path(candidate)
        if not p.is_absolute():
            p = root_dir / p
        resolved = p.resolve()
        root_resolved = root_dir.resolve()
        return resolved == root_resolved or root_resolved in resolved.parents
    except Exception:
        return False


def _extract_tool_paths(tool_call: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for loc in tool_call.get("locations") or []:
        if isinstance(loc, dict) and loc.get("path"):
            paths.append(str(loc["path"]))
    raw_input = tool_call.get("rawInput") or {}
    if isinstance(raw_input, dict):
        for key in ("path", "file_path", "target_file", "AbsolutePath", "TargetFile", "DirectoryPath"):
            val = raw_input.get(key)
            if isinstance(val, str) and val.strip():
                paths.append(val.strip())
    return paths


class ScopedAgyAcpSession:
    """Context-managed `agy acp` session locked to `work_dir` with hard timeouts."""

    def __init__(
        self,
        work_dir: Path,
        model_id: str = DEFAULT_ACP_MODEL,
        session_timeout_sec: float = 240.0,
        allow_work_dir_writes: bool = False,
    ) -> None:
        self.work_dir = work_dir.resolve()
        self.model_id = model_id
        self.session_timeout_sec = float(session_timeout_sec)
        self.allow_work_dir_writes = allow_work_dir_writes

        self._proc: subprocess.Popen[str] | None = None
        self._rx_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._next_id = 1
        self._session_id: str | None = None
        self._deadline: float = 0.0

    def __enter__(self) -> "ScopedAgyAcpSession":
        agy_exe = find_agy_executable()
        if not agy_exe:
            raise FileNotFoundError(
                "agy CLI executable not found. Install Antigravity CLI and ensure `agy` is available."
            )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._deadline = time.monotonic() + self.session_timeout_sec

        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        self._proc = subprocess.Popen(
            [agy_exe, "acp", "--sandbox"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.work_dir),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self._reader_thread = threading.Thread(
            target=self._stdout_reader_loop,
            name="agy-acp-reader",
            daemon=True,
        )
        self._reader_thread.start()

        try:
            self._rpc_call(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {},
                    "clientInfo": {"name": "beatlyric-pipeline", "version": "0.1.0"},
                },
                timeout_sec=20.0,
            )
            sess_res = self._rpc_call(
                "session/new",
                {
                    "cwd": str(self.work_dir),
                    "mcpServers": [],
                },
                timeout_sec=25.0,
            )
            self._session_id = str(sess_res["sessionId"])

            available_ids = {
                m.get("modelId")
                for m in (sess_res.get("models") or {}).get("availableModels", [])
                if isinstance(m, dict)
            }
            if available_ids and self.model_id not in available_ids:
                raise RuntimeError(
                    f"Requested ACP model '{self.model_id}' not available. Available: {sorted(available_ids)}"
                )

            self._rpc_call(
                "session/set_model",
                {
                    "sessionId": self._session_id,
                    "modelId": self.model_id,
                },
                timeout_sec=15.0,
            )
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _stdout_reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for raw_line in self._proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    self._rx_queue.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finally:
            self._rx_queue.put(None)

    def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin or self._proc.poll() is not None:
            raise RuntimeError("agy acp process is not running")
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _choose_permission_outcome(self, params: dict[str, Any]) -> dict[str, Any]:
        """Strictly deny shell execution and any paths outside `self.work_dir`."""
        options = params.get("options") or []
        tool_call = params.get("toolCall") or {}
        kind = str(tool_call.get("kind") or "").lower()

        allow_opt = next((o for o in options if o.get("kind") in ("allow_once", "allow_always")), None)
        reject_opt = next((o for o in options if o.get("kind") in ("reject_once", "reject_always")), None)

        allowed = False
        if kind == "read":
            paths = _extract_tool_paths(tool_call)
            allowed = bool(paths) and all(_is_path_within(p, self.work_dir) for p in paths)
        elif kind in ("edit", "write") and self.allow_work_dir_writes:
            paths = _extract_tool_paths(tool_call)
            allowed = bool(paths) and all(_is_path_within(p, self.work_dir) for p in paths)
        else:
            # Deny execute/command/network/outside-workspace requests
            allowed = False

        chosen = allow_opt if allowed and allow_opt else reject_opt
        if chosen and chosen.get("optionId"):
            return {"outcome": {"outcome": "selected", "optionId": chosen["optionId"]}}
        return {"outcome": {"outcome": "cancelled"}}

    def _handle_server_request(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "session/request_permission":
            result = self._choose_permission_outcome(params)
            self._send_json({"jsonrpc": "2.0", "id": req_id, "result": result})
        else:
            self._send_json({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unsupported client method: {method}"},
            })

    def _rpc_call(
        self,
        method: str,
        params: dict[str, Any],
        timeout_sec: float = 90.0,
        collect_chunks: list[str] | None = None,
    ) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._send_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })

        turn_deadline = min(self._deadline, time.monotonic() + timeout_sec)
        while True:
            remaining = turn_deadline - time.monotonic()
            if remaining <= 0:
                if self._session_id and method == "session/prompt":
                    try:
                        self._send_json({
                            "jsonrpc": "2.0",
                            "method": "session/cancel",
                            "params": {"sessionId": self._session_id},
                        })
                    except Exception:
                        pass
                raise TimeoutError(f"agy acp '{method}' timed out")

            try:
                msg = self._rx_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if self._proc and self._proc.poll() is not None:
                    raise RuntimeError(f"agy acp exited unexpectedly ({self._proc.returncode})")
                continue

            if msg is None:
                raise RuntimeError("agy acp stdout closed unexpectedly")

            if msg.get("id") == req_id and ("result" in msg or "error" in msg):
                if "error" in msg and msg["error"]:
                    raise RuntimeError(f"agy acp error in '{method}': {msg['error']}")
                return msg.get("result") or {}

            if "method" in msg and "id" in msg:
                self._handle_server_request(msg)
                continue

            if msg.get("method") == "session/update" and collect_chunks is not None:
                upd = (msg.get("params") or {}).get("update") or {}
                if upd.get("sessionUpdate") == "agent_message_chunk":
                    content = upd.get("content") or {}
                    if content.get("type") == "text" and content.get("text"):
                        collect_chunks.append(str(content["text"]))

    def prompt_text(self, text: str, timeout_sec: float = 120.0) -> str:
        """Send a single prompt turn inside `self.work_dir` and return the text response."""
        if not self._session_id:
            raise RuntimeError("ACP session not initialized")
        chunks: list[str] = []
        self._rpc_call(
            "session/prompt",
            {
                "sessionId": self._session_id,
                "prompt": [{"type": "text", "text": text}],
            },
            timeout_sec=timeout_sec,
            collect_chunks=chunks,
        )
        return "".join(chunks).strip()

    def close(self) -> None:
        """Close the ACP session and forcefully terminate the subprocess if still alive."""
        proc = self._proc
        if proc is None:
            return

        try:
            if self._session_id and proc.poll() is None:
                self._deadline = time.monotonic() + 3.0
                try:
                    self._rpc_call(
                        "session/close",
                        {"sessionId": self._session_id},
                        timeout_sec=2.5,
                    )
                except Exception:
                    pass
        finally:
            self._session_id = None
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass

            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    if platform.system() == "Windows":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                            check=False,
                        )
                    else:
                        proc.kill()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
            self._proc = None
