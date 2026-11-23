"""Build agent DSL source in an isolated child process (§22); only IR JSON comes back.

Layers, each enough to stop A1-style code execution on its own:
  1. the AST allow-list (static_check) still runs inside the child;
  2. the child has no network (its own network namespace, when the OS allows it),
     runs as an unprivileged user (when we are root), cannot write files
     (RLIMIT_FSIZE=0), cannot fork (RLIMIT_NPROC=0, which root would ignore), and
     has capped CPU, memory and wall time, an empty environment and a temp cwd;
  3. the parent never runs agent code: it re-validates the returned IR with
     judge.ir_check, which replays it through the DSL.

shortcut: process + namespace isolation, not a container. Run the judge inside
runner/sandbox/Dockerfile (read-only fs, --network none) for real agent runs.
"""
from __future__ import annotations

import functools
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from aliensilicon.ir import Kernel, Program

from .ir_check import load_kernel, load_program
from .static_check import Rejected

ROOT = Path(__file__).resolve().parent.parent
CPU_SECONDS = 10
MEMORY_BYTES = 2 << 30
MAX_OUTPUT_BYTES = 8_000_000


NOBODY = 65534


@functools.cache
def _prefix() -> tuple[str, ...]:
    """Command prefix giving the child its own network namespace and, if we are root, a non-root uid.

    As root: `unshare -n` then `setpriv` down to nobody (root ignores RLIMIT_NPROC, nobody does not).
    As a normal user: `unshare -rn` (uid 0 inside the namespace, but not in the real one).
    """
    unshare, setpriv = shutil.which("unshare"), shutil.which("setpriv")
    if os.geteuid() == 0 and unshare and setpriv:
        cand = (unshare, "-n", setpriv, f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--")
    elif unshare:
        cand = (unshare, "-rn")
    else:
        return ()
    try:
        ok = subprocess.run([*cand, "true"], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return cand if ok else ()


def isolation() -> dict[str, bool]:
    """What this machine's sandbox actually provides; recorded with every judge result."""
    p = _prefix()
    unprivileged = any("setpriv" in part for part in p) or os.geteuid() != 0
    return {"network_namespace": bool(p), "unprivileged_uid": unprivileged}


def _limits() -> None:  # runs in the child between fork and exec
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_isolated(code: str, stdin: str = "", timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """Run trusted Python `code` (which may handle untrusted data) in the isolated child."""
    # NPROC is set by the child itself after unshare has exec'd python, or unshare could not start it.
    prelude = ("import resource, sys\nresource.setrlimit(resource.RLIMIT_NPROC, (0, 0))\n"
               f"sys.path.insert(0, {str(ROOT)!r})\n")
    with tempfile.TemporaryDirectory() as cwd:
        os.chmod(cwd, 0o755)  # readable by the unprivileged child; it cannot write anyway
        return subprocess.run(
            [*_prefix(), sys.executable, "-I", "-S", "-c", prelude + code],
            input=stdin, capture_output=True, text=True, timeout=timeout, cwd=cwd,
            env={"PATH": "/usr/bin:/bin"}, preexec_fn=_limits, check=False,
        )


_WORKER = "from judge.sandbox_worker import main\nmain()\n"


def build_in_sandbox(source: str, timeout: float = 30) -> dict[str, Kernel | Program]:
    """Same contract as static_check.build_submission, but agent code never runs in this process."""
    try:
        proc = run_isolated(_WORKER, stdin=source, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Rejected("build timed out") from None
    out = proc.stdout
    if proc.returncode != 0 or len(out) > MAX_OUTPUT_BYTES:
        raise Rejected(f"build crashed (exit {proc.returncode}): {proc.stderr.strip()[-300:]}")
    try:
        msg: dict[str, Any] = json.loads(out)
    except json.JSONDecodeError:
        raise Rejected("build produced no valid result") from None
    if "rejected" in msg:
        raise Rejected(msg["rejected"])
    result: dict[str, Kernel | Program] = {}
    for name, text in msg["ok"].items():
        kind = json.loads(text).get("type")
        result[name] = load_kernel(text) if kind == "kernel" else load_program(text)
    return result
