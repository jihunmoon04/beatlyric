"""Open the BeatLyric dashboard from a Windows shortcut."""

import ctypes
import http.client
import json
import msvcrt
import os
from pathlib import Path
import socket
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8765
DASHBOARD = f"http://{HOST}:{PORT}/dashboard"
STARTUP_TIMEOUT = 45


def dashboard_ready():
    connection = http.client.HTTPConnection(HOST, PORT, timeout=0.5)
    try:
        connection.request("GET", "/api/dashboard")
        response = connection.getresponse()
        if response.status != 200:
            return False
        state = json.load(response)
        return isinstance(state, dict) and "spotify_sync" in state and isinstance(state.get("tracks"), list)
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def port_in_use():
    try:
        with socket.create_connection((HOST, PORT), timeout=0.3):
            return True
    except OSError:
        return False


def start_studio(log_path):
    python = PROJECT / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise RuntimeError(f"BeatLyric의 Python 실행 파일을 찾을 수 없습니다.\n{python}")

    with log_path.open("wb") as log:
        return subprocess.Popen(
            [str(python), "-X", "utf8", "-u", "pipeline/run.py", "studio",
             "--host", HOST, "--port", str(PORT), "--no-browser"],
            cwd=PROJECT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )


def ensure_dashboard():
    if dashboard_ready():
        return

    studio_dir = PROJECT / "work" / ".studio"
    studio_dir.mkdir(parents=True, exist_ok=True)
    log_path = studio_dir / "launcher-server.log"
    deadline = time.monotonic() + STARTUP_TIMEOUT

    # Two clicks while the server is starting must share a single launch.
    with (studio_dir / "launcher.lock").open("a+b") as lock:
        while time.monotonic() < deadline:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("BeatLyric이 시작 중입니다. 잠시 후 다시 열어 주세요.")

        if dashboard_ready():
            return

        process = None if port_in_use() else start_studio(log_path)
        while time.monotonic() < deadline:
            if dashboard_ready():
                return
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"BeatLyric 서버가 시작 중 종료됐습니다.\n로그 {log_path}")
            time.sleep(0.25)

        if process is None:
            raise RuntimeError(f"{PORT} 포트에서 BeatLyric 응답을 확인하지 못했습니다.")
        raise RuntimeError(f"BeatLyric 서버가 제시간에 응답하지 않았습니다.\n로그 {log_path}")


def main():
    try:
        ensure_dashboard()
        os.startfile(DASHBOARD)
    except (OSError, RuntimeError) as exc:
        ctypes.windll.user32.MessageBoxW(None, str(exc), "BeatLyric", 0x10)


if __name__ == "__main__":
    main()
