"""Run the model2obs comparison (foreground / detached / PBS) and track it."""

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from model2obs_mcp import _env


def _sidecar(config_path: Path, suffix: str) -> Path:
    # Kept next to the config, NOT in the logs folder: run(clear_output=True)
    # empties the logs folder (except model2obs.log) at start-up.
    return config_path.parent / f"{config_path.stem}.{suffix}"


def _write_pbs_script(config_path: Path, cmd: list[str], pbs_project: str, pbs_queue: str,
                      pbs_walltime: str, pbs_ncpus: int, pbs_mem: str) -> tuple[Path, Path]:
    """Write the qsub script; returns (script_path, pbs_log_path)."""
    python = _env.model2obs_python()
    act = _env.activation_info(python)
    script_file = _sidecar(config_path, "mcp_pbs.sh")
    log_file = _sidecar(config_path, "mcp_pbs.log")

    lines = [
        "#!/bin/bash -l",
        "#PBS -N model2obs",
        f"#PBS -A {pbs_project}",
        f"#PBS -q {pbs_queue}",
        f"#PBS -l select=1:ncpus={pbs_ncpus}:mem={pbs_mem}",
        f"#PBS -l walltime={pbs_walltime}",
        "#PBS -j oe",
        f"#PBS -o {log_file}",
        "",
        "# A PBS job starts with a fresh environment: re-create the model2obs conda",
        "# env activation (activate.d exports DART/CrocoLake paths, PYTHONPATH and",
        "# runs `module load nco`), then pin the values seen at submit time.",
    ]
    if act["prefix"]:
        prefix = shlex.quote(act["prefix"])
        lines += [
            f"export CONDA_PREFIX={prefix}",
            f'export PATH={prefix}/bin:"$PATH"',
            f"for f in {prefix}/etc/conda/activate.d/*.sh; do . \"$f\" >/dev/null 2>&1 || true; done",
        ]
    keep = ["DART_ROOT_PATH", "CROCOLAKE_PATH", "CROCOLAKE_OBS_CONV_PATH", "TUTORIAL_DATA_PATH",
            "MODEL2OBS_PATH", "NCAR_SHARED_DATA_ROOT", "PYTHONPATH", "PROJ_DATA"]
    env_now = _env.worker_env(python)
    for k in keep:
        if env_now.get(k):
            lines.append(f"export {k}={shlex.quote(env_now[k])}")
    extra = os.environ.get("MODEL2OBS_MCP_PROTECTED_PATHS")
    if extra:
        lines.append(f"export MODEL2OBS_MCP_PROTECTED_PATHS={shlex.quote(extra)}")
    lines += [
        "export PYTHONUNBUFFERED=1 MPLBACKEND=Agg",
        "command -v ncks >/dev/null || echo 'WARNING: ncks not on PATH (needed for multi-snapshot model files)'",
        f"cd {shlex.quote(str(config_path.parent))}",
        " ".join(shlex.quote(c) for c in cmd),
        "",
    ]
    script_file.write_text("\n".join(lines))
    script_file.chmod(0o755)
    return script_file, log_file


def run_comparison(
    config_path: str,
    parallel: bool = False,
    trim_obs: bool = True,
    clear_output: bool = False,
    parquet_only: bool = False,
    no_matching: bool = False,
    force_obs_time: bool = False,
    background: Union[bool, str] = False,
    pbs_project: Optional[str] = None,
    pbs_queue: str = "casper",
    pbs_walltime: str = "01:00:00",
    pbs_ncpus: int = 4,
    pbs_mem: str = "32GB",
) -> dict:
    """
    Run the model2obs model-vs-observation comparison for a config.

    Equivalent to `WorkflowModelObs.from_config_file(cfg).run(...)` (CLI:
    `perfect-model-obs -c cfg --trim ...`): for each model snapshot it finds
    the obs_seq file whose time range fits the time_window, optionally trims it
    to the model domain (convex hull of ocean_geometry), runs DART's
    perfect_model_obs to interpolate the model to the obs locations, then
    merges everything into parquet_folder/model-obs-*.parquet with columns
    time, longitude, latitude, vertical, type, interpolated_model, obs,
    obs_err_var, interpolated_model_QC, difference (= obs - model),
    abs_difference, squared_difference, normalized_difference, log_likelihood.

    Always validates the config first (validate_config with the same flags) and
    refuses to start if there are errors. The library is run in a subprocess
    with CWD = the config file's directory (its constructor deletes
    ./perfect_model_obs.log and may create ./netcdf_output there; if
    netcdf_output_folder is unset it is pinned to <config dir>/netcdf_output).

    IMPORTANT: the shared DART perfect_model_obs is a Casper build. Run on a
    Casper login node (fine for a handful of pairs) or with background="pbs"
    on the casper queue. From a Derecho login node use
    pbs_queue="casper@casper-pbs".

    Modes (background=...):
      False   foreground: blocks until done and returns the result. Only for
              small runs (a few pairs, < a few minutes).
      True    detached subprocess on this host (start_new_session); returns at
              once. stdout/stderr -> <config stem>.mcp_run.log next to the config.
      "pbs"   writes <config stem>.mcp_pbs.sh next to the config (re-sources the
              conda env's activate.d, exports DART/CrocoLake paths, runs the
              worker) and qsubs it; returns the job id. Job output ->
              <config stem>.mcp_pbs.log.
    In every mode the worker writes <config stem>.mcp_status.json next to the
    config ({"status": running|done|failed, ...}); poll get_run_status.

    Reruns: the library requires empty output folders. clear_output=True
    deletes the files in parquet_folder, input_nml_bck, trimmed_obs_folder,
    output_folder, netcdf_output_folder and the logs folder, and rmtree's
    tmp_folder, before running -- only set it when you mean to discard the
    previous results.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.
    parallel : bool
        One thread per model file (library's ThreadPoolExecutor). Default False.
    trim_obs : bool
        Trim obs to the model domain before perfect_model_obs (default True;
        requires trimmed_obs_folder). Note the CLI's default is False.
    clear_output : bool
        Empty all output folders first (destructive, see above).
    parquet_only : bool
        Skip perfect_model_obs; only rebuild parquet from existing obs_seq_*.out.
    no_matching : bool
        Skip time matching and pair sorted model files with sorted obs files 1:1.
    force_obs_time : bool
        Stamp the model state with the obs_seq time instead of the model time.
    background : bool or "pbs"
        See Modes above.
    pbs_project : str
        PBS account (required for background="pbs").
    pbs_queue : str
        PBS queue (default "casper"; "casper@casper-pbs" from Derecho).
    pbs_walltime : str
        hh:mm:ss (default "01:00:00").
    pbs_ncpus : int
        CPUs to request (default 4; raise with parallel=True and many files).
    pbs_mem : str
        Memory to request (default "32GB").
    """
    cfg = Path(config_path).expanduser().resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"config not found: {cfg}")

    validation = _env.run_worker(
        "validate_config",
        dict(config_path=str(cfg), trim_obs=trim_obs, clear_output=clear_output, parquet_only=parquet_only),
        timeout=600,
    )
    if not validation["valid"]:
        return {"status": "refused", "reason": "config failed validation",
                "errors": validation["errors"], "warnings": validation["warnings"]}

    status_file = _sidecar(cfg, "mcp_status.json")
    args = dict(config_path=str(cfg), status_file=str(status_file), trim_obs=trim_obs,
                no_matching=no_matching, force_obs_time=force_obs_time, parquet_only=parquet_only,
                clear_output=clear_output, parallel=parallel)
    cmd = _env.worker_cmd("run", args)
    common = {"config_path": str(cfg), "status_file": str(status_file),
              "warnings": validation["warnings"],
              "log_file": validation["info"].get("log_file")}

    if background == "pbs":
        if not pbs_project:
            raise ValueError("pbs_project is required for background='pbs'")
        script, pbs_log = _write_pbs_script(cfg, cmd, pbs_project, pbs_queue, pbs_walltime, pbs_ncpus, pbs_mem)
        status_file.write_text(json.dumps({"status": "queued", "script": str(script),
                                           "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
        res = subprocess.run(["qsub", str(script)], capture_output=True, text=True)
        if res.returncode != 0:
            status_file.write_text(json.dumps({"status": "failed", "error": f"qsub: {res.stderr.strip()}"}))
            raise RuntimeError(f"qsub failed: {res.stderr.strip()}")
        jobid = res.stdout.strip()
        status_file.write_text(json.dumps({"status": "queued", "pbs_jobid": jobid, "script": str(script)}))
        return {"status": "queued", "pbs_jobid": jobid, "script_file": str(script), "pbs_log": str(pbs_log),
                **common, "hint": "Poll get_run_status(config_path)."}

    if background:
        run_log = _sidecar(cfg, "mcp_run.log")
        status_file.write_text(json.dumps({"status": "starting"}))
        with open(run_log, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, cwd=str(cfg.parent),
                                    env=_env.worker_env(), start_new_session=True)
        return {"status": "running", "pid": proc.pid, "run_log": str(run_log), **common,
                "hint": "Poll get_run_status(config_path)."}

    try:
        result = _env.run_worker("run", args, cwd=str(cfg.parent), timeout=None)
    except _env.WorkerError as exc:
        return {"status": "failed", "error": str(exc)[-4000:], **common,
                "hint": "get_run_status(config_path) shows per-pair DART logs."}
    result.update(common)
    return result


def get_run_status(config_path: str, tail: int = 30) -> dict:
    """
    Report progress / outcome of a model2obs run for a config.

    Reads (all read-only):
      - <config stem>.mcp_status.json next to the config, written by
        run_comparison's worker: status (queued/starting/running/done/failed),
        flags, pid/host or PBS job id, timings, result counts, and the error +
        traceback on failure. For a "running" status on this host it checks the
        pid is still alive (a dead pid means the worker was killed, e.g. OOM);
        for PBS runs it includes `qstat -x <jobid>`.
      - the tail of the library's run log (config log_file, model2obs.log)
        and any warning lines in it
      - every pair_summary_NNNN.log in the logs folder, parsed into one row per
        model/obs pair: model file, time used, obs_seq file, obs submitted to
        DART, DART exit code, successful / failed interpolations
      - for pairs where DART failed, ERROR/FATAL lines from
        perfect_model_obs_NNNN.log
      - counts of obs_seq_*.out / perfect_output_*.nc in output_folder and of
        all / good (QC<=2) / failed rows in parquet_folder.

    Parameters
    ----------
    config_path : str
        Path to the YAML config used for the run.
    tail : int
        Number of run-log lines to return (default 30).
    """
    cfg = Path(config_path).expanduser().resolve()
    return _env.run_worker(
        "run_status",
        {"config_path": str(cfg), "status_file": str(_sidecar(cfg, "mcp_status.json")), "tail": tail},
        timeout=300,
    )
