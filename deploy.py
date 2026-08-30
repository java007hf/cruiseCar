#!/usr/bin/env python3
"""Deploy the CruiseCar Python control server.

What it does, in order:
  1. git pull --ff-only  (skip with --no-pull)
  2. kill any running `control_server.server` process (SIGTERM, then SIGKILL)
  3. start the server detached, or exec into it (--no-daemon)

Usage:
  python3 deploy.py            # pull + restart (server detached into background)
  python3 deploy.py --no-pull  # restart only, skip git pull
  python3 deploy.py --check    # just print whether the server is running
  python3 deploy.py --no-daemon  # replace this process with the server
                                #   (use when launched in a managed background
                                #   runner, e.g. `run_in_background`)

The server needs no pip dependencies (pure stdlib). It must run in
`full` deployment mode, which is what CRUISECAR_DEPLOYMENT=full sets.
After start/check, the script prints execution status, local port status,
and user-facing URLs directly to the command line. Set CRUISECAR_PUBLIC_HOST
to your public IP or domain to make those URLs concrete instead of
`http://<server-ip>:...`.
"""
from __future__ import annotations

import argparse
import platform
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = os.getenv("CRUISECAR_REPO_DIR", str(Path(__file__).resolve().parent))
SERVER_DIR = os.path.join(REPO_DIR, "server")
LOG_FILE = os.path.join(SERVER_DIR, "cruisecar-server.log")
SERVER_MODULE = "control_server.server"
PROC_PATTERN = SERVER_MODULE
ENV_DEPLOYMENT = "full"
# Optional env file (gitignored) holding TURN / other runtime secrets that the
# server process should inherit on every (re)start. Lines are KEY=VALUE.
ENV_FILE = os.path.join(SERVER_DIR, ".env")
DEFAULT_PORTS = {
    "control relay": int(os.getenv("CRUISECAR_CONTROL_PORT", "42110")),
    "WebRTC signaling": int(os.getenv("CRUISECAR_WEBRTC_PORT", "42112")),
    "manager-api": int(os.getenv("CRUISECAR_MANAGER_PORT", "8088")),
    "manager-web": int(os.getenv("CRUISECAR_MANAGER_WEB_PORT", "8089")),
    "MCP server": int(os.getenv("CRUISECAR_XIAOZHI_MCP_PORT", "8090")),
}


def _run(cmd, cwd=None):
    print(f"[deploy] $ {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def _server_host_for_users() -> str:
    return os.getenv("CRUISECAR_PUBLIC_HOST") or os.getenv("PUBLIC_HOST") or "<server-ip>"


def _print_access_info() -> None:
    host = _server_host_for_users()
    print("[deploy] access information:", flush=True)
    print(f"[deploy]   manager web:  http://{host}:{DEFAULT_PORTS['manager-web']}/", flush=True)
    print(f"[deploy]   web sender:   http://{host}:{DEFAULT_PORTS['manager-web']}/send/", flush=True)
    print(f"[deploy]   manager api:  http://{host}:{DEFAULT_PORTS['manager-api']}/", flush=True)
    print(f"[deploy]   health check:  http://{host}:{DEFAULT_PORTS['manager-api']}/health", flush=True)
    print(f"[deploy]   Android server account mode uses {host}:{DEFAULT_PORTS['control relay']} for TCP control", flush=True)
    print(f"[deploy]   WebRTC signaling port: {DEFAULT_PORTS['WebRTC signaling']}", flush=True)
    print(f"[deploy]   xiaozhi MCP endpoint: http://{host}:{DEFAULT_PORTS['MCP server']}/mcp", flush=True)
    if host == "<server-ip>":
        print("[deploy]   Set CRUISECAR_PUBLIC_HOST=your.domain.or.ip to print concrete public URLs.", flush=True)


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _print_port_status() -> None:
    print("[deploy] local port status:", flush=True)
    for name, port in DEFAULT_PORTS.items():
        state = "listening" if _is_port_open("127.0.0.1", port) else "not listening"
        print(f"[deploy]   {port:<5} {name}: {state}", flush=True)


def _any_service_port_open() -> bool:
    return any(_is_port_open("127.0.0.1", port) for port in DEFAULT_PORTS.values())


def git_pull() -> None:
    print("[deploy] pulling latest code ...", flush=True)
    _run(["git", "fetch", "--all"], cwd=REPO_DIR)
    rc = _run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
    if rc != 0:
        print("[deploy] WARNING: git pull failed / not a fast-forward; "
              "continuing with current code", file=sys.stderr, flush=True)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR
        ).decode().strip()
        print(f"[deploy] now at commit {head}", flush=True)
    except Exception:
        pass


def _running_pids():
    if platform.system().lower().startswith("win"):
        return _running_pids_windows()
    try:
        out = subprocess.check_output(["pgrep", "-f", PROC_PATTERN]).decode().split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids = []
    for pid_s in out:
        try:
            pids.append(int(pid_s))
        except ValueError:
            pass
    return pids


def _running_pids_windows():
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*-m control_server.server*' } | "
        "ForEach-Object { $_.ProcessId }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            stderr=subprocess.DEVNULL,
        ).decode().split()
    except Exception:
        return []
    pids = []
    for pid_s in out:
        try:
            pids.append(int(pid_s))
        except ValueError:
            pass
    return pids


def kill_old() -> None:
    print("[deploy] stopping any existing server process ...", flush=True)
    my_pid = os.getpid()
    killed = []
    for pid in _running_pids():
        if pid == my_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass

    if not killed:
        print("[deploy] no existing server process found", flush=True)
        return

    print(f"[deploy] sent SIGTERM to pids: {killed}", flush=True)
    # Give them a moment to shut down gracefully.
    for pid in killed:
        for _ in range(20):  # up to ~2s
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
    # Force-kill anything still alive.
    for pid in killed:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"[deploy] force-killed pid {pid}", flush=True)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(1)


def load_env_file(path: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file (no shell expansion). Missing -> {}."""
    if not os.path.isfile(path):
        return {}
    parsed: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def start_server(daemonize: bool) -> int:
    print("[deploy] starting server ...", flush=True)
    env = dict(os.environ)
    env.update(load_env_file(ENV_FILE))
    env["CRUISECAR_DEPLOYMENT"] = ENV_DEPLOYMENT

    if daemonize:
        stdout = open(LOG_FILE, "a", encoding="utf-8")
        stdin = open(os.devnull)
        proc = subprocess.Popen(
            [sys.executable, "-m", SERVER_MODULE],
            cwd=SERVER_DIR,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stdout,
            start_new_session=True,  # equivalent to setsid: own session/process group
            close_fds=True,
        )
        print(f"[deploy] server launched as pid {proc.pid} (detached)", flush=True)
        return proc.pid

    _print_access_info()
    print("[deploy] entering foreground server process; press Ctrl+C to stop", flush=True)
    # exec mode: replace this process image with the server. Does not return.
    os.chdir(SERVER_DIR)
    os.execvpe(sys.executable, [sys.executable, "-m", SERVER_MODULE], env)
    return 0  # unreachable


def status() -> list[int]:
    pids = _running_pids()
    # Filter out the deploy.py launcher itself if it shows up.
    real = []
    for pid in pids:
        try:
            if platform.system().lower().startswith("win"):
                cmd = ""
            else:
                cmd = subprocess.check_output(["cat", f"/proc/{pid}/cmdline"]).decode()
        except Exception:
            cmd = ""
        if "deploy.py" not in cmd:
            real.append(pid)
    if real:
        print("[deploy] server running, pids:", real, flush=True)
    else:
        print("[deploy] server NOT running", flush=True)
    _print_port_status()
    return real


def print_start_result() -> None:
    pids = status()
    if _any_service_port_open():
        print("[deploy] result: service is reachable locally", flush=True)
    elif pids:
        print("[deploy] result: process is running, but ports are not listening yet", flush=True)
        print("[deploy]         wait a few seconds, then run: python3 deploy.py --check", flush=True)
    else:
        print("[deploy] result: service failed to start", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull code and restart CruiseCar server.")
    ap.add_argument("--no-pull", action="store_true", help="skip git pull")
    ap.add_argument("--check", action="store_true", help="only report running status")
    ap.add_argument("--no-daemon", action="store_true",
                    help="exec into the server (for managed background runners)")
    args = ap.parse_args()

    print(f"[deploy] repo dir: {REPO_DIR}", flush=True)
    print(f"[deploy] server dir: {SERVER_DIR}", flush=True)
    print(f"[deploy] deployment mode: {ENV_DEPLOYMENT}", flush=True)

    if args.check:
        status()
        _print_access_info()
        return

    if not args.no_pull:
        git_pull()
    kill_old()

    if args.no_daemon:
        start_server(daemonize=False)  # replaces this process; never returns
    else:
        start_server(daemonize=True)
        time.sleep(2)
        print_start_result()
        _print_access_info()


if __name__ == "__main__":
    main()
