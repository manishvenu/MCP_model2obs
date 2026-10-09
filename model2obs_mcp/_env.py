"""Interpreter split: run model2obs work in the library's own Python.

The MCP server process never imports model2obs / dask / pandas. Every
library call is a subprocess:

    MODEL2OBS_PYTHON  model2obs_mcp/_worker.py  <op>  <json args>

which prints exactly one JSON document on stdout.

The model2obs conda env relies on ``etc/conda/activate.d/*.sh`` to export
CROCOLAKE_PATH, DART_ROOT_PATH, TUTORIAL_DATA_PATH, CROCOLAKE_OBS_CONV_PATH,
PYTHONPATH (for DART's convert_crocolake_obs) and to ``module load nco``
(ncks). The server is not launched from an activated env, so when
MODEL2OBS_PYTHON lives inside a conda env we source those scripts once in a
throwaway bash, diff the environment, cache the delta, and merge it into
every worker's environment.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().with_name("_worker.py")

# Output paths that the server refuses to write into (in addition to
# $CROCOLAKE_PATH). Extend with MODEL2OBS_MCP_PROTECTED_PATHS (colon-separated).
DEFAULT_PROTECTED = ["/glade/campaign"]

_SPLIT = "__MODEL2OBS_MCP_SPLIT__"
_ACTIVATION_CACHE: dict[str, dict] = {}

# Lmod init scripts tried when `module` is not already a shell function
# (e.g. the server was started with a scrubbed environment).
_MODULE_INITS = [
    "/etc/profile.d/z00_modules.sh",
    "$LMOD_PKG/init/bash",
    "/usr/share/lmod/lmod/init/bash",
]


def model2obs_python() -> str:
    """Python interpreter of the model2obs env (MODEL2OBS_PYTHON, else sys.executable)."""
    return os.environ.get("MODEL2OBS_PYTHON") or sys.executable


def conda_prefix_for(python: str) -> Path | None:
    """Return the conda env prefix that `python` lives in, or None."""
    prefix = Path(python).expanduser().parent.parent
    if (prefix / "conda-meta").is_dir():
        return prefix
    return None


def _parse_env0(blob: str) -> dict[str, str]:
    out = {}
    for item in blob.split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def activation_info(python: str | None = None, refresh: bool = False) -> dict:
    """Source the env's activate.d/*.sh once and return (and cache) the env delta.

    Returns {"prefix", "scripts", "vars" (name -> value that changed), "error"}.
    """
    python = python or model2obs_python()
    if python in _ACTIVATION_CACHE and not refresh:
        return _ACTIVATION_CACHE[python]

    info: dict = {"prefix": None, "scripts": [], "vars": {}, "error": None}
    prefix = conda_prefix_for(python)
    if prefix is None:
        info["error"] = (
            f"{python} is not inside a conda env (no conda-meta/); "
            "no activate.d scripts sourced"
        )
        _ACTIVATION_CACHE[python] = info
        return info
    info["prefix"] = str(prefix)
    act_dir = prefix / "etc" / "conda" / "activate.d"
    scripts = sorted(act_dir.glob("*.sh")) if act_dir.is_dir() else []
    info["scripts"] = [str(s) for s in scripts]

    module_init = " ".join(
        f'[ -f "{p}" ] && . "{p}" >/dev/null 2>&1 && type module >/dev/null 2>&1 && break;'
        for p in _MODULE_INITS
    )
    sources = "".join(f". {shlex.quote(str(s))} >/dev/null 2>&1; " for s in scripts)
    script = (
        "env -0; "
        f"printf '\\0{_SPLIT}\\0'; "
        f"export CONDA_PREFIX={shlex.quote(str(prefix))}; "
        f"export CONDA_DEFAULT_ENV={shlex.quote(prefix.name)}; "
        f'export PATH={shlex.quote(str(prefix / "bin"))}:"$PATH"; '
        f"if ! type module >/dev/null 2>&1; then for _i in 1; do {module_init} done; fi; "
        f"{sources}"
        "env -0"
    )
    try:
        res = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=120
        )
        before_blob, _, after_blob = res.stdout.partition(f"\0{_SPLIT}\0")
        before, after = _parse_env0(before_blob), _parse_env0(after_blob)
        delta = {
            k: v
            for k, v in after.items()
            if before.get(k) != v
            and not k.startswith("BASH_FUNC_")
            and k not in ("_", "SHLVL", "PWD", "OLDPWD")
        }
        info["vars"] = delta
        if not after:
            info["error"] = f"activation shell produced no env (rc={res.returncode}): {res.stderr[-500:]}"
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"failed to source activate.d scripts: {exc}"
    _ACTIVATION_CACHE[python] = info
    return info


def worker_env(python: str | None = None) -> dict[str, str]:
    """os.environ + the conda activation delta + worker settings."""
    env = dict(os.environ)
    env.update(activation_info(python)["vars"])
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def protected_paths(env: dict[str, str] | None = None) -> list[str]:
    env = env or worker_env()
    paths = list(DEFAULT_PROTECTED)
    if env.get("CROCOLAKE_PATH"):
        paths.append(env["CROCOLAKE_PATH"])
    extra = os.environ.get("MODEL2OBS_MCP_PROTECTED_PATHS", "")
    paths += [p for p in extra.split(":") if p]
    return [os.path.realpath(os.path.expandvars(os.path.expanduser(p))) for p in paths]


def worker_cmd(op: str, args: dict, python: str | None = None) -> list[str]:
    return [python or model2obs_python(), str(WORKER), op, json.dumps(args)]


class WorkerError(RuntimeError):
    pass


def run_worker(op: str, args: dict | None = None, cwd: str | None = None,
               timeout: float | None = 1800) -> dict:
    """Run one worker op and return its JSON result (raises WorkerError on failure)."""
    python = model2obs_python()
    cmd = worker_cmd(op, args or {}, python)
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, env=worker_env(python),
            cwd=cwd, timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise WorkerError(f"Cannot launch MODEL2OBS_PYTHON={python}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkerError(f"worker op '{op}' timed out after {timeout}s") from exc

    payload = None
    out = res.stdout.strip()
    if out:
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            try:
                payload = json.loads(out.splitlines()[-1])
            except json.JSONDecodeError:
                payload = None
    if res.returncode != 0 or payload is None:
        msg = f"worker op '{op}' failed (exit {res.returncode})"
        if isinstance(payload, dict) and "error" in payload:
            msg += f": {payload['error']}\n--- traceback ---\n{payload.get('traceback', '')[-3000:]}"
        else:
            msg += f"\n--- stdout ---\n{res.stdout[-1500:]}"
        msg += f"\n--- stderr (tail) ---\n{res.stderr[-1500:]}"
        raise WorkerError(msg)
    return payload
