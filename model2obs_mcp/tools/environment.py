"""Environment checks and DART observation-type discovery."""

import os
import sys
from typing import Optional

from model2obs_mcp import _env


def check_environment(perfect_model_obs_dir: Optional[str] = None) -> dict:
    """
    Check that everything a model2obs comparison needs is in place.

    Run this first in any session. It reports:
      - which Python runs the library (MODEL2OBS_PYTHON, default: the server's own
        interpreter) and which runs this server
      - whether that Python is inside a conda env, and which of the env's
        etc/conda/activate.d/*.sh scripts were sourced (they export
        DART_ROOT_PATH, CROCOLAKE_PATH, CROCOLAKE_OBS_CONV_PATH, TUTORIAL_DATA_PATH,
        PYTHONPATH and `module load nco`), and the variables they set
      - model2obs version and the versions of pydartdiags/dask/pandas/pyarrow/
        xarray/matplotlib/cartopy/gsw
      - DART: $DART_ROOT_PATH/models/MOM6/work/perfect_model_obs exists, is
        executable and has no unresolved shared libraries (ldd); the
        observations/forward_operators/obs_def_ocean_mod.rst used for
        use_these_obs expansion. NOTE: the shared DART install is a *Casper*
        build -- run comparisons on Casper (login node for tiny runs, or
        run_comparison(background="pbs") on the casper queue).
      - `ncks` on PATH (needed when a model file holds more than one time snapshot)
      - CrocoLake: $CROCOLAKE_PATH readable, _common_metadata present, schema
        columns, number of parquet files (non-recursive listing only)
      - DART's convert_crocolake_obs module importable (used by
        make_obs_seq_from_crocolake)

    Everything is read-only. Returns {"ok", "problems": [...], "server": {...},
    "activation": {...}, "checks": {...}}. If the worker cannot start at all
    (bad MODEL2OBS_PYTHON, model2obs not installed), "ok" is False and
    "worker_error" explains why.

    Parameters
    ----------
    perfect_model_obs_dir : str, optional
        Check this DART work dir instead of $DART_ROOT_PATH/models/MOM6/work.
    """
    python = _env.model2obs_python()
    act = _env.activation_info(python)
    out = {
        "server": {
            "python": sys.executable,
            "MODEL2OBS_PYTHON": os.environ.get("MODEL2OBS_PYTHON"),
            "library_python": python,
            "library_python_source": "MODEL2OBS_PYTHON" if os.environ.get("MODEL2OBS_PYTHON") else "sys.executable (MODEL2OBS_PYTHON unset)",
        },
        "activation": {
            "conda_prefix": act["prefix"],
            "scripts_sourced": act["scripts"],
            "error": act["error"],
            "vars_set": sorted(k for k in act["vars"] if not k.startswith(("_ModuleTable", "__LMOD"))),
        },
        "protected_paths": _env.protected_paths(),
    }
    try:
        res = _env.run_worker("check_env", {"perfect_model_obs_dir": perfect_model_obs_dir}, timeout=300)
    except _env.WorkerError as exc:
        out.update({"ok": False, "problems": ["worker failed to run"], "worker_error": str(exc)})
        return out
    problems = list(res.get("problems", []))
    if act["error"]:
        problems.append(f"activation: {act['error']}")
    out.update({"ok": res.get("ok", False) and not act["error"], "problems": problems, "checks": res.get("checks")})
    return out


def list_obs_types(quantity: Optional[str] = None, rst_path: Optional[str] = None) -> dict:
    """
    List DART ocean observation types, grouped by quantity (QTY_*).

    Parses DART's obs_def_ocean_mod.rst exactly like model2obs does for the
    `use_these_obs` config key (model2obs.utils.config.parse_obs_def_ocean_mod).
    In `use_these_obs` you may list individual types (e.g. ARGO_TEMPERATURE) or
    a whole quantity with the ALL_<FIELD> selector (e.g. ALL_TEMPERATURE expands
    to every type whose quantity is QTY_TEMPERATURE). The returned
    "all_selectors" are the valid ALL_* names.

    Obs_seq files produced from CrocoLake (make_obs_seq_from_crocolake) use the
    types ARGO_*, BOTTLE_* (GLODAP), GLIDER_* (SprayGliders), and plain
    TEMPERATURE / SALINITY for other sources -- all are covered by
    ALL_TEMPERATURE / ALL_SALINITY.

    Parameters
    ----------
    quantity : str, optional
        Filter, e.g. "TEMPERATURE", "QTY_SALINITY" or "ALL_SALINITY" (substring
        match on the quantity name). Default: all quantities.
    rst_path : str, optional
        Explicit path to obs_def_ocean_mod.rst. Default:
        $DART_ROOT_PATH/observations/forward_operators/obs_def_ocean_mod.rst.
    """
    return _env.run_worker("list_obs_types", {"quantity": quantity, "rst_path": rst_path}, timeout=120)
