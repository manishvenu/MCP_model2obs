"""model2obs worker: runs inside the model2obs env (MODEL2OBS_PYTHON).

Usage:  python _worker.py <op> '<json args>'

Prints exactly one JSON document on the real stdout, last. Everything the
library prints (model2obs's ModuleLogger prints, DART converter chatter,
ncks child processes) is redirected to stderr at the file-descriptor level.
On exception prints {"error": ..., "traceback": ...} and exits 1.

Imports only the stdlib at module level; library modules (model2obs, dask,
pandas, pyarrow, pydartdiags, matplotlib, ...) are imported lazily per op.
This file is NOT imported by the MCP server process.
"""

import json
import math
import os
import socket
import sys
import traceback
from datetime import datetime, timezone

# Running `python .../model2obs_mcp/_worker.py` puts the package dir on
# sys.path[0]; drop it so sibling modules can never shadow library imports.
if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

PROTECTED_DEFAULT = ["/glade/campaign"]

OUTPUT_KEYS = ["output_folder", "tmp_folder", "trimmed_obs_folder", "input_nml_bck", "parquet_folder"]

# perfect_model_obs writes QC = 0 on success, 1000 + istatus on failure
# (quality_control_mod.f90:set_input_qc). istatus codes for MOM6 from
# models/MOM6/model_mod.f90.
QC_MEANINGS = {
    0: "success (forward operator ran)",
    1000: "obs type not in assimilate/evaluate list (use_these_obs) - forward operator not run",
    1001: "generic forward-operator failure (istatus 1)",
    1012: "MOM6: NOT_IN_STATE - required quantity not in model_state_variables",
    1013: "MOM6: THICKNESS_NOT_IN_STATE - need use_pseudo_depth: true or h: QTY_LAYER_THICKNESS",
    1014: "MOM6: QUAD_LOCATE_FAILED - horizontal location outside model grid",
    1015: "MOM6: THICKNESS_QUAD_EVALUATE_FAILED",
    1016: "MOM6: QUAD_EVALUATE_FAILED - horizontal interpolation failed",
    1017: "MOM6: QUAD_ON_LAND - surrounding grid cell(s) are land",
    1018: "MOM6: QUAD_ON_BASIN_EDGE - point not in basin / next to land or open boundary",
    1020: "MOM6: OBS_ABOVE_SURFACE",
    1022: "MOM6: OBS_TOO_DEEP - deeper than local model bathymetry/last layer",
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _protected():
    paths = list(PROTECTED_DEFAULT)
    if os.environ.get("CROCOLAKE_PATH"):
        paths.append(os.environ["CROCOLAKE_PATH"])
    paths += [p for p in os.environ.get("MODEL2OBS_MCP_PROTECTED_PATHS", "").split(":") if p]
    return [os.path.realpath(os.path.expandvars(os.path.expanduser(p))) for p in paths]


def _is_protected(path):
    rp = os.path.realpath(path)
    for p in _protected():
        if rp == p or rp.startswith(p.rstrip("/") + "/"):
            return p
    return None


def _assert_writable_target(path, what):
    hit = _is_protected(path)
    if hit:
        raise PermissionError(f"Refusing to write {what} '{path}': it is under protected path '{hit}'")


def _jsonable(obj):
    """Best-effort conversion of numpy/pandas scalars & containers to JSON types."""
    try:
        import numpy as np  # noqa: F401
    except Exception:  # noqa: BLE001
        np = None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return _jsonable(float(obj))
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _jsonable(obj.tolist())
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _rst_path(perfect_model_obs_dir=None):
    if perfect_model_obs_dir:
        return os.path.normpath(os.path.join(
            perfect_model_obs_dir, "../../../observations/forward_operators/obs_def_ocean_mod.rst"))
    root = os.environ.get("DART_ROOT_PATH")
    if not root:
        return None
    return os.path.join(root, "observations/forward_operators/obs_def_ocean_mod.rst")


def _norm_lon(lon):
    """Map a longitude to [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _lon_ranges(lon_min, lon_max):
    """Return list of (lo, hi) ranges in [-180,180], splitting at the dateline."""
    lo, hi = float(lon_min), float(lon_max)
    if hi - lo >= 360:
        return [(-180.0, 180.0)]
    lo_n, hi_n = _norm_lon(lo), _norm_lon(hi)
    if hi_n == -180.0 and hi > lo:
        hi_n = 180.0
    if lo_n <= hi_n:
        return [(lo_n, hi_n)]
    return [(lo_n, 180.0), (-180.0, hi_n)]


def _check_bounds(lat_min, lat_max, lon_min, lon_max, start, end):
    import pandas as pd
    for name, v in (("lat_min", lat_min), ("lat_max", lat_max), ("lon_min", lon_min), ("lon_max", lon_max)):
        if v is None:
            raise ValueError(f"{name} is required")
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(f"need -90 <= lat_min < lat_max <= 90 (got {lat_min}, {lat_max})")
    if start is None or end is None:
        raise ValueError("start and end are required")
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if not t0 < t1:
        raise ValueError(f"start must be before end (got {t0}, {t1})")
    return t0, t1


def _read_config_raw(config_path):
    """model2obs.utils.config.read_config, but tolerant of a missing time_window."""
    import yaml
    from model2obs.utils import config as cu
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} does not contain a YAML mapping")
    cfg = cu.resolve_config_paths(dict(raw), config_path)
    return raw, cfg


# ----------------------------------------------------------------------------
# ops
# ----------------------------------------------------------------------------

def op_check_env(args):
    import importlib
    import importlib.metadata as md
    import shutil
    import subprocess

    checks, problems = {}, []
    checks["python"] = sys.executable
    checks["python_version"] = sys.version.split()[0]
    checks["host"] = socket.gethostname()
    checks["ncar_host"] = os.environ.get("NCAR_HOST")

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:  # noqa: BLE001
            return None

    try:
        import model2obs
        checks["model2obs_version"] = ver("model2obs")
        checks["model2obs_path"] = os.path.dirname(model2obs.__file__)
    except Exception as exc:  # noqa: BLE001
        checks["model2obs_version"] = None
        problems.append(f"model2obs not importable: {exc}")
    checks["packages"] = {p: ver(p) for p in
                          ["pydartdiags", "dask", "pandas", "pyarrow", "xarray", "matplotlib", "cartopy", "gsw", "pyyaml"]}

    env_names = ["DART_ROOT_PATH", "CROCOLAKE_PATH", "CROCOLAKE_OBS_CONV_PATH",
                 "TUTORIAL_DATA_PATH", "MODEL2OBS_PATH", "CONDA_PREFIX"]
    checks["env"] = {k: os.environ.get(k) for k in env_names}
    for k in env_names[:3]:
        if not os.environ.get(k):
            problems.append(f"${k} is not set (normally exported by the model2obs env's activate.d)")

    dart = os.environ.get("DART_ROOT_PATH")
    pmo_dir = args.get("perfect_model_obs_dir") or (os.path.join(dart, "models/MOM6/work") if dart else None)
    pmo = os.path.join(pmo_dir, "perfect_model_obs") if pmo_dir else None
    pmo_info = {"path": pmo, "exists": bool(pmo and os.path.isfile(pmo)),
                "executable": bool(pmo and os.access(pmo, os.X_OK))}
    if pmo_info["exists"]:
        try:
            ldd = subprocess.run(["ldd", pmo], capture_output=True, text=True, timeout=30)
            missing = [l.strip() for l in ldd.stdout.splitlines() if "not found" in l]
            pmo_info["missing_shared_libs"] = missing
            if missing:
                problems.append(f"perfect_model_obs has unresolved shared libs: {missing}")
        except Exception as exc:  # noqa: BLE001
            pmo_info["ldd_error"] = str(exc)
        if "Casper" in pmo and "derecho" in (checks["host"] or "") + (checks["ncar_host"] or ""):
            pmo_info["note"] = "This perfect_model_obs is a Casper build; run comparisons on Casper (or a casper PBS job)."
    else:
        problems.append(f"perfect_model_obs not found at {pmo}")
    checks["perfect_model_obs"] = pmo_info

    rst = _rst_path(pmo_dir)
    checks["obs_def_ocean_mod_rst"] = {"path": rst, "exists": bool(rst and os.path.isfile(rst))}
    if not checks["obs_def_ocean_mod_rst"]["exists"]:
        problems.append(f"obs_def_ocean_mod.rst not found at {rst} (needed for use_these_obs expansion)")

    ncks = shutil.which("ncks")
    checks["ncks"] = ncks
    if not ncks:
        problems.append("ncks not on PATH (needed when a model file has >1 time snapshot; activate.d runs `module load nco`)")

    cl = os.environ.get("CROCOLAKE_PATH")
    cl_info = {"path": cl}
    if cl:
        cl_info["readable"] = os.path.isdir(cl) and os.access(cl, os.R_OK | os.X_OK)
        cm = os.path.join(cl, "_common_metadata")
        cl_info["common_metadata"] = os.path.isfile(cm)
        if cl_info["common_metadata"]:
            try:
                import pyarrow.parquet as pq
                cl_info["columns"] = pq.read_schema(cm).names
            except Exception as exc:  # noqa: BLE001
                cl_info["schema_error"] = str(exc)
        if cl_info["readable"]:
            n = 0
            with os.scandir(cl) as it:
                for e in it:
                    if e.name.endswith(".parquet"):
                        n += 1
            cl_info["n_parquet_files"] = n
        else:
            problems.append(f"CROCOLAKE_PATH {cl} not readable")
    checks["crocolake"] = cl_info

    try:
        importlib.import_module("convert_crocolake_obs")
        checks["convert_crocolake_obs"] = "importable"
    except Exception as exc:  # noqa: BLE001
        checks["convert_crocolake_obs"] = f"NOT importable: {exc}"
        problems.append("DART convert_crocolake_obs not importable (needs $CROCOLAKE_OBS_CONV_PATH on PYTHONPATH)")

    return {"ok": not problems, "problems": problems, "checks": checks}


def op_list_obs_types(args):
    from model2obs.utils import config as cu
    rst = args.get("rst_path") or _rst_path(args.get("perfect_model_obs_dir"))
    if not rst:
        raise ValueError("No rst_path given and $DART_ROOT_PATH is unset")
    type_to_qty, qty_to_types = cu.parse_obs_def_ocean_mod(rst)
    q = args.get("quantity")
    if q:
        key = q.upper()
        for pref in ("ALL_", "QTY_"):
            if key.startswith(pref):
                key = key[len(pref):]
        qty_to_types = {k: v for k, v in qty_to_types.items() if key in k[4:]}
    return {
        "rst_path": rst,
        "n_types": sum(len(v) for v in qty_to_types.values()),
        "quantities": {k: sorted(v) for k, v in sorted(qty_to_types.items())},
        "all_selectors": sorted(f"ALL_{k[4:]}" for k in qty_to_types),
    }


def _crocolake_path(args):
    p = args.get("crocolake_path") or os.environ.get("CROCOLAKE_PATH")
    if not p:
        raise ValueError("crocolake_path not given and $CROCOLAKE_PATH unset")
    if not os.path.isdir(p):
        raise FileNotFoundError(f"CrocoLake path {p} not found")
    return p


def op_query_crocolake(args):
    import pandas as pd
    import pyarrow.compute as pc  # noqa: F401
    import pyarrow.parquet as pq

    path = _crocolake_path(args)
    lat_min, lat_max = float(args["lat_min"]), float(args["lat_max"])
    t0, t1 = _check_bounds(lat_min, lat_max, args.get("lon_min"), args.get("lon_max"),
                           args.get("start"), args.get("end"))
    lon_ranges = _lon_ranges(args["lon_min"], args["lon_max"])
    lon_span = sum(h - l for l, h in lon_ranges)
    days = (t1 - t0).total_seconds() / 86400.0
    budget = lon_span * (lat_max - lat_min) * days
    if budget > float(args.get("max_deg2_days", 50000)) and not args.get("allow_large"):
        raise ValueError(
            f"Query box x window = {budget:.0f} deg^2*days exceeds the safety limit "
            f"({args.get('max_deg2_days', 50000)}). Narrow the box/window or pass allow_large=True.")

    schema = pq.read_schema(os.path.join(path, "_common_metadata")).names
    variables = args.get("variables") or ["TEMP", "PSAL"]
    bad = [v for v in variables if v not in schema]
    if bad:
        raise ValueError(f"variables {bad} not in CrocoLake schema {schema}")
    cols = ["DB_NAME", "PLATFORM_NUMBER", "CYCLE_NUMBER", "JULD", "LATITUDE", "LONGITUDE", "PRES"]
    if "DEPTH" in schema:
        cols.append("DEPTH")
    for v in variables:
        for c in (v, v + "_QC", v + "_ERROR"):
            if c in schema and c not in cols:
                cols.append(c)

    filters = []
    for lo, hi in lon_ranges:
        f = [("LATITUDE", ">=", lat_min), ("LATITUDE", "<=", lat_max),
             ("LONGITUDE", ">=", lo), ("LONGITUDE", "<=", hi),
             ("JULD", ">=", t0.to_pydatetime()), ("JULD", "<", t1.to_pydatetime())]
        if args.get("db_names"):
            f.append(("DB_NAME", "in", list(args["db_names"])))
        filters.append(f)

    table = pq.read_table(path, columns=cols, filters=filters)
    df = table.to_pandas()
    out = {
        "crocolake_path": path,
        "filters": {"lat": [lat_min, lat_max], "lon_ranges": lon_ranges,
                    "start": str(t0), "end": str(t1), "db_names": args.get("db_names")},
        "n_rows": int(len(df)),
    }
    if len(df) == 0:
        out["note"] = "No CrocoLake rows in this box/window."
        return out

    df["_prof"] = df["DB_NAME"].astype(str) + "|" + df["PLATFORM_NUMBER"].astype(str) + "|" + df["CYCLE_NUMBER"].astype(str)
    per_db = {}
    for db, g in df.groupby("DB_NAME", observed=True):
        d = {"rows": int(len(g)), "profiles": int(g["_prof"].nunique()),
             "platforms": int(g["PLATFORM_NUMBER"].nunique())}
        for v in variables:
            nn = g[v].notna()
            d[f"{v}_nonnull"] = int(nn.sum())
            strict = nn.copy()
            for suf in ("_QC", "_ERROR"):
                if v + suf in g:
                    strict &= g[v + suf].notna()
            d[f"{v}_with_qc_and_error"] = int(strict.sum())
        per_db[str(db)] = d
    out["per_db_name"] = per_db
    out["profiles_total"] = int(df["_prof"].nunique())
    out["time_range"] = [str(df["JULD"].min()), str(df["JULD"].max())]
    out["lat_range"] = [float(df["LATITUDE"].min()), float(df["LATITUDE"].max())]
    out["lon_range"] = [float(df["LONGITUDE"].min()), float(df["LONGITUDE"].max())]
    out["pres_range_dbar"] = [float(df["PRES"].min()), float(df["PRES"].max())] if df["PRES"].notna().any() else None
    out["rows_missing_PRES"] = int(df["PRES"].isna().sum())
    if "DEPTH" in df and df["DEPTH"].notna().any():
        out["depth_range_m"] = [float(df["DEPTH"].min()), float(df["DEPTH"].max())]
    out["note"] = ("make_obs_seq_from_crocolake with loose=False keeps only values that have both _QC and "
                   "_ERROR (see *_with_qc_and_error); loose=True keeps every non-null value.")
    if not args.get("summary_only", True):
        n = max(0, min(int(args.get("max_rows", 20)), 200))
        out["sample"] = _jsonable(df.drop(columns="_prof").head(n).to_dict(orient="records"))
    return _jsonable(out)


def op_make_obs_seq(args):
    import shutil
    import tempfile
    import pandas as pd

    out_path = os.path.abspath(os.path.expanduser(args["out_path"]))
    _assert_writable_target(out_path, "obs_seq file")
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} already exists; refusing to overwrite")
    path = _crocolake_path(args)
    lat_min, lat_max = float(args["lat_min"]), float(args["lat_max"])
    t0, t1 = _check_bounds(lat_min, lat_max, args.get("lon_min"), args.get("lon_max"),
                           args.get("start"), args.get("end"))
    variables = args.get("variables") or ["TEMP", "PSAL"]
    allowed = {"TEMP", "PSAL", "DOXY", "TOT_ALKALINITY", "TCO2", "NITRATE", "SILICATE", "PHOSPHATE"}
    bad = [v for v in variables if v not in allowed]
    if bad:
        raise ValueError(f"variables {bad} not supported by convert_crocolake_obs ({sorted(allowed)})")

    from convert_crocolake_obs import ObsSequence  # DART converter (needs CROCOLAKE_OBS_CONV_PATH)

    # Only these columns: the converter treats every other non-QC/ERROR column as an observation.
    selected = ["DB_NAME", "JULD", "LATITUDE", "LONGITUDE", "PRES", "PRES_QC", "PRES_ERROR"]
    for v in variables:
        selected += [v, v + "_QC", v + "_ERROR"]

    db_filters = []
    for lo, hi in _lon_ranges(args["lon_min"], args["lon_max"]):
        base = [("LATITUDE", ">=", lat_min), ("LATITUDE", "<=", lat_max),
                ("LONGITUDE", ">=", lo), ("LONGITUDE", "<=", hi),
                ("PRES", ">", -1e30), ("PRES", "<", 1e30),
                ("JULD", ">=", t0.to_pydatetime()), ("JULD", "<", t1.to_pydatetime())]
        if args.get("db_names"):
            base.append(("DB_NAME", "in", list(args["db_names"])))
        for v in variables:
            db_filters.append(base + [(v, ">", -1e30), (v, "<", 1e30)])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Stage in a temp dir: the converter writes <out>.tmp next to its output,
    # and a stray .tmp inside an obs_seq_in_folder breaks model2obs's '*' glob.
    stage = tempfile.mkdtemp(prefix="m2o_obsseq_")
    try:
        staged = os.path.join(stage, os.path.basename(out_path))
        obs = ObsSequence(path, selected, db_filters,
                          fill_na_qc=args.get("fill_na_qc"), fill_na_error=args.get("fill_na_error"),
                          loose=bool(args.get("loose", False)), obs_seq_out=staged)
        n_rows = int(len(obs.ddf))
        obs.write_obs_seq()
        if not os.path.exists(staged):
            return {"written": False, "out_path": out_path, "n_rows": n_rows,
                    "note": "No observations matched; nothing written."}
        shutil.move(staged, out_path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    num_obs, types = None, []
    with open(out_path, encoding="ascii") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.startswith("num_obs:"):
                num_obs = int(s.split()[1])
                break
            parts = s.split()
            if i >= 3 and len(parts) == 2 and parts[0].isdigit():
                types.append(parts[1])
    return {"written": True, "out_path": out_path, "n_crocolake_rows": n_rows, "num_obs": num_obs,
            "obs_types_in_header": types,
            "time_range": [str(t0), str(t1)],
            "note": ("Salinity is written in kg/kg (PSU*1e-3), as DART expects. "
                     + ("loose=True: values lacking _ERROR get obs error variance -888888, "
                        "which makes normalized_difference/log_likelihood NaN for them." if args.get("loose") else ""))}


def _obs_seq_times(path):
    import pydartdiags.obs_sequence.obs_sequence as obsq
    o = obsq.ObsSequence(path)
    return o.df["time"].min(), o.df["time"].max(), int(len(o.df))


def op_validate_config(args):
    """Static validation of a model2obs YAML config without constructing WorkflowModelObs."""
    import pandas as pd
    from model2obs.utils import config as cu
    from model2obs.model_adapter.registry import create_model_adapter

    config_path = os.path.abspath(os.path.expanduser(args["config_path"]))
    trim_obs = bool(args.get("trim_obs", True))
    clear_output = bool(args.get("clear_output", False))
    parquet_only = bool(args.get("parquet_only", False))
    errors, warnings, info = [], [], {}

    if not os.path.isfile(config_path):
        return {"valid": False, "config_path": config_path, "errors": [f"config file not found: {config_path}"],
                "warnings": [], "info": {}}
    raw, cfg = _read_config_raw(config_path)
    unexpanded = {k: v for k, v in cfg.items() if isinstance(v, str) and "$" in v}
    for k, v in unexpanded.items():
        errors.append(f"{k}: unexpanded environment variable in '{v}' (is the model2obs env activated?)")

    # time window
    if "time_window" not in raw:
        errors.append("time_window is missing (required by read_config)")
    else:
        try:
            cfg = cu.convert_time_window(cfg)
            tw = cfg["time_window"]
            info["time_window"] = tw
            if tw["days"] * 86400 + tw["seconds"] <= 0:
                errors.append("time_window is zero; no obs_seq file can ever match a model snapshot")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"time_window invalid: {exc}")

    # adapter + required keys
    ocean_model = cfg.get("ocean_model")
    adapter = None
    if ocean_model is None:
        errors.append("ocean_model is missing (MOM6 or ROMS_Rutgers)")
    else:
        try:
            adapter = create_model_adapter(ocean_model)
            missing = [k for k in adapter.get_required_config_keys() if k not in cfg]
            if missing:
                errors.append(f"required keys missing for {ocean_model}: {missing}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ocean_model '{ocean_model}': {exc}")
    for k in ("log_file", "tmp_folder"):
        if k not in cfg:
            errors.append(f"{k} is missing: run() reads config['{k}'] with no default")

    # CWD-relative defaults the library would use
    if "input_nml_bck" not in cfg:
        errors.append("input_nml_bck is missing: the library default 'input.nml.backup' is CWD-relative and "
                      "collides with the input.nml.backup FILE the base Namelist writes in CWD; "
                      "run(clear_output=True) also KeyErrors without it. Set an explicit folder.")
    if trim_obs and "trimmed_obs_folder" not in cfg:
        errors.append("trimmed_obs_folder is missing: required with trim_obs=True "
                      "(library default 'trimmed_obs_seq' is CWD-relative; clear_output=True KeyErrors without it)")
    if "netcdf_output_folder" not in cfg:
        warnings.append("netcdf_output_folder not set: the library creates './netcdf_output' in the CWD "
                        "(the MCP runs with CWD = the config's directory)")

    # inputs
    mff = cfg.get("model_files_folder")
    model_files = []
    if mff:
        if not os.path.isdir(mff):
            errors.append(f"model_files_folder does not exist: {mff}")
        else:
            entries = [f for f in sorted(os.listdir(mff)) if os.path.isfile(os.path.join(mff, f))]
            non_nc = [f for f in entries if not f.endswith(".nc")]
            model_files = [os.path.join(mff, f) for f in entries if f.endswith(".nc")]
            if not entries:
                errors.append(f"model_files_folder is empty: {mff}")
            if non_nc:
                errors.append(f"model_files_folder contains non-.nc files (library rejects this): {non_nc[:10]}")
            info["n_model_files"] = len(model_files)
    osf = cfg.get("obs_seq_in_folder")
    obs_files = []
    if osf:
        if not os.path.isdir(osf):
            errors.append(f"obs_seq_in_folder does not exist: {osf}")
        else:
            obs_files = sorted(os.path.join(osf, f) for f in os.listdir(osf)
                               if not f.startswith(".") and os.path.isfile(os.path.join(osf, f)))
            info["n_obs_seq_files"] = len(obs_files)
            if not obs_files:
                errors.append(f"obs_seq_in_folder is empty: {osf}")
            not_obs = []
            for f in obs_files:
                if f.endswith(".tmp"):
                    not_obs.append(os.path.basename(f))
                    continue
                try:
                    with open(f, "rb") as fh:
                        head = fh.read(64)
                    if b"obs_sequence" not in head:
                        not_obs.append(os.path.basename(f))
                except OSError:
                    not_obs.append(os.path.basename(f))
            if not_obs:
                errors.append("obs_seq_in_folder is globbed with '*' and these files do not look like ASCII "
                              f"obs_seq files (stray .tmp / other files break pairing): {not_obs[:10]}")
    if (ocean_model or "").strip().lower() == "mom6":
        for k in ("template_file", "static_file", "ocean_geometry"):
            v = cfg.get(k)
            if v and not os.path.isfile(v):
                errors.append(f"{k} does not exist: {v}")
            elif v and not v.endswith(".nc"):
                errors.append(f"{k} is not a .nc file: {v}")
        sv = cfg.get("model_state_variables")
        if sv is not None:
            if not isinstance(sv, dict):
                errors.append("model_state_variables must be a mapping var: QTY_...")
            elif not cfg.get("use_pseudo_depth", False) and "QTY_LAYER_THICKNESS" not in sv.values():
                errors.append("use_pseudo_depth must be true unless model_state_variables includes "
                              "h: QTY_LAYER_THICKNESS (otherwise DART fails every obs with 1013)")
    elif cfg.get("roms_filename") and not os.path.isfile(cfg["roms_filename"]):
        errors.append(f"roms_filename does not exist: {cfg['roms_filename']}")

    # DART
    pmo_dir = cfg.get("perfect_model_obs_dir")
    if pmo_dir:
        exe = os.path.join(pmo_dir, "perfect_model_obs")
        if not os.path.isfile(exe):
            errors.append(f"perfect_model_obs not found: {exe}")
        elif not os.access(exe, os.X_OK):
            errors.append(f"perfect_model_obs not executable: {exe}")
        rst = _rst_path(pmo_dir)
        if "use_these_obs" in cfg:
            if not os.path.isfile(rst):
                errors.append(f"use_these_obs set but {rst} is missing; the library would silently ignore "
                              "use_these_obs and fall back to the template obs_kind_nml")
            else:
                try:
                    expanded = cu.validate_and_expand_obs_types(list(cfg["use_these_obs"]),
                                                                cu.parse_obs_def_ocean_mod(rst))
                    info["use_these_obs_expanded"] = expanded
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"use_these_obs invalid ({exc}); the library would only warn and silently "
                                  "fall back to the template obs_kind_nml")
        else:
            warnings.append("use_these_obs not set: input.nml template's obs_kind_nml is used as-is")

    # outputs
    out_paths = {k: cfg[k] for k in OUTPUT_KEYS if k in cfg}
    if cfg.get("netcdf_output_folder") and os.path.isabs(str(cfg["netcdf_output_folder"])):
        out_paths["netcdf_output_folder"] = cfg["netcdf_output_folder"]
    log_file = cfg.get("log_file")
    for k, v in out_paths.items():
        hit = _is_protected(v)
        if hit:
            errors.append(f"{k} '{v}' is under protected path '{hit}' (MCP refuses to write there)")
        if os.path.exists(v):
            if not os.path.isdir(v):
                errors.append(f"{k} exists but is not a directory: {v}")
            elif os.listdir(v):
                if k == "parquet_folder" and parquet_only:
                    warnings.append(f"parquet_folder is not empty ({v}); parquet_only will append/overwrite model-obs-*.parquet")
                elif k in ("output_folder", "trimmed_obs_folder") and parquet_only:
                    pass  # parquet_only reads these
                elif clear_output:
                    warnings.append(f"{k} is not empty and will be CLEARED (clear_output=True): {v}")
                else:
                    errors.append(f"{k} is not empty: {v} (library requires empty output folders; "
                                  "use a fresh folder or clear_output=True)")
    if log_file:
        hit = _is_protected(log_file)
        if hit:
            errors.append(f"log_file '{log_file}' is under protected path '{hit}'")
        logs_dir = os.path.dirname(log_file)
        for k, v in out_paths.items():
            vv = os.path.realpath(v).rstrip("/") + "/"
            if (os.path.realpath(logs_dir).rstrip("/") + "/").startswith(vv) and k != "netcdf_output_folder":
                errors.append(f"log_file is inside {k} ({v}); the logs folder is created before path validation, "
                              "so that folder is never empty and run() fails")
        info["log_file"] = log_file
    if parquet_only and cfg.get("output_folder"):
        of = cfg["output_folder"]
        if not (os.path.isdir(of) and any(f.startswith("obs_seq") for f in os.listdir(of))):
            errors.append(f"parquet_only=True but output_folder has no obs_seq*.out files: {of}")

    # optional pairing dry-run
    if args.get("check_pairs") and model_files and obs_files and adapter is not None and "time_window" in info:
        import shutil
        from datetime import timedelta
        tw = timedelta(days=info["time_window"]["days"], seconds=info["time_window"]["seconds"])
        obs_t = {}
        for f in obs_files:
            try:
                obs_t[f] = _obs_seq_times(f)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"cannot read obs_seq {os.path.basename(f)}: {exc}")
        used, pairs, unmatched, multi = set(), [], [], False
        for mf in model_files:
            with adapter.open_dataset_ctx(mf) as ds:
                times = list(ds["time"].values)
            if len(times) > 1:
                multi = True
            for t_id, t in enumerate(times):
                ts = pd.Timestamp(t)
                lo, hi = ts - tw / 2, ts + tw / 2
                hit = None
                for f, (t1, t2, n) in obs_t.items():
                    if f in used:
                        continue
                    if lo <= t1 <= hi and lo <= t2 <= hi:
                        hit = f
                        break
                if hit:
                    used.add(hit)
                    pairs.append({"model_file": os.path.basename(mf), "snapshot": t_id, "time": str(ts),
                                  "obs_seq": os.path.basename(hit), "obs_time_range": [str(obs_t[hit][0]), str(obs_t[hit][1])],
                                  "n_obs": obs_t[hit][2]})
                else:
                    unmatched.append({"model_file": os.path.basename(mf), "snapshot": t_id, "time": str(ts),
                                      "window": [str(lo), str(hi)]})
        info["pairs"] = pairs
        info["unmatched_snapshots"] = unmatched
        info["unused_obs_seq"] = [os.path.basename(f) for f in obs_t if f not in used]
        if not pairs:
            errors.append("time matching found no model-snapshot/obs_seq pairs (whole obs_seq time range must fit "
                          "inside the time_window centred on the snapshot)")
        if multi and not shutil.which("ncks"):
            errors.append("model files have >1 time snapshot so ncks is needed, but ncks is not on PATH")

    return _jsonable({"valid": not errors, "config_path": config_path, "errors": errors,
                      "warnings": warnings, "info": info, "resolved_config": cfg})


def _template_path(template):
    if template and (os.path.sep in template or template.endswith((".yaml", ".yml"))):
        p = os.path.abspath(os.path.expanduser(template))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"template {p} not found")
        return p
    name = template or "MOM6"
    roots = []
    if os.environ.get("MODEL2OBS_PATH"):
        roots.append(os.environ["MODEL2OBS_PATH"])
    try:
        import model2obs
        roots.append(os.path.dirname(os.path.dirname(model2obs.__file__)))
    except Exception:  # noqa: BLE001
        pass
    for r in roots:
        p = os.path.join(r, "tutorials", "config_templates", f"config_template_{name}.yaml")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"config_template_{name}.yaml not found under {roots}")


def op_write_config(args):
    import yaml

    path = os.path.abspath(os.path.expanduser(args["path"]))
    _assert_writable_target(path, "config")
    if os.path.exists(path) and not args.get("overwrite"):
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    values = dict(args.get("values") or {})
    notes = []
    template = args.get("template", "MOM6")
    if template in ("", "none", "None"):
        cfg, tpath = {}, None
    else:
        tpath = _template_path(template or "MOM6")
        with open(tpath, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    root = args.get("output_root")
    if root:
        root = os.path.abspath(os.path.expanduser(os.path.expandvars(root)))
        defaults = {
            "output_folder": os.path.join(root, "out_obs_seq"),
            "input_nml_bck": os.path.join(root, "input_nml_bckp"),
            "trimmed_obs_folder": os.path.join(root, "out_trimmed_obs_seq"),
            "parquet_folder": os.path.join(root, "out_parquet"),
            "tmp_folder": os.path.join(root, "tmp"),
            "netcdf_output_folder": os.path.join(root, "out_netcdf"),
            "log_file": os.path.join(root, "logs", "model2obs.log"),
        }
        for k, v in defaults.items():
            values.setdefault(k, v)
    cfg.update(values)
    cfg.pop("logging", None)  # template documents a nested logging: key that nothing reads

    of = cfg.get("output_folder")
    if "log_file" not in cfg and of:
        cfg["log_file"] = os.path.join(os.path.dirname(os.path.normpath(of)), "logs", "model2obs.log")
        notes.append(f"log_file defaulted to {cfg['log_file']} (must NOT be inside output_folder)")
    if "tmp_folder" not in cfg and of:
        cfg["tmp_folder"] = os.path.join(os.path.dirname(os.path.normpath(of)), "tmp")
        notes.append(f"tmp_folder defaulted to {cfg['tmp_folder']}")
    if "time_window" not in cfg:
        cfg["time_window"] = {"days": 1}
        notes.append("time_window defaulted to {days: 1}")
    cfg.setdefault("ocean_model", "MOM6")

    placeholders = [k for k, v in cfg.items() if isinstance(v, str) and v.startswith("/path/to")]
    if placeholders:
        notes.append(f"still template placeholders (set these): {placeholders}")
    tmpl_outputs = [k for k, v in cfg.items() if isinstance(v, str) and "model2obs_output/workflow_1" in v]
    if tmpl_outputs:
        notes.append(f"still the template's $WORK/model2obs_output/workflow_1 defaults: {tmpl_outputs} "
                     "(pass output_root to place all outputs together)")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (f"# model2obs config written by model2obs-mcp write_config "
              f"({datetime.now().isoformat(timespec='seconds')})\n"
              f"# template: {tpath}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    val = op_validate_config({"config_path": path, "trim_obs": args.get("trim_obs", True)})
    val.pop("resolved_config", None)
    return {"path": path, "template": tpath, "notes": notes, "config": cfg, "validation": val}


def op_preview_namelist(args):
    """Build input.nml in a temp CWD (constructor side effects land there)."""
    import tempfile
    from model2obs.utils import logging_utils
    from model2obs.workflows.workflow_model_obs import WorkflowModelObs

    config_path = os.path.abspath(os.path.expanduser(args["config_path"]))
    with tempfile.TemporaryDirectory(prefix="m2o_preview_") as tmp:
        os.chdir(tmp)
        wf = WorkflowModelObs.from_config_file(config_path, netcdf_output_folder=os.path.join(tmp, "netcdf_output"))
        # library bug: _initialize_model_namelist uses self._logger, which is only
        # created by run() -> _setup_run_logger(); preview_namelist() alone raises.
        if not hasattr(wf, "_logger"):
            wf._logger = logging_utils.get_module_logger("model2obs.workflows.workflow_model_obs")
        nml_file = os.path.join(tmp, "input.nml.preview")
        wf.preview_namelist(nml_file)
        with open(nml_file, encoding="utf-8") as f:
            content = f.read()
        os.chdir("/")
    out = {"config_path": config_path, "namelist": content,
           "note": "Pair-specific entries (input_state_files, obs_seq_in/out_file_name, init_time_*) are filled per pair at run time."}
    if args.get("out_file"):
        p = os.path.abspath(os.path.expanduser(args["out_file"]))
        _assert_writable_target(p, "namelist")
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        out["written_to"] = p
    return out


def _write_status(path, data):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)
    os.replace(tmp, path)


def _parquet_counts(folder):
    import pyarrow.dataset as pads
    import pyarrow.compute as pc
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("model-obs-") and f.endswith(".parquet"))
    if not files:
        return None
    t = pads.dataset(files, format="parquet").to_table(columns=["interpolated_model_QC"])
    qc = t.column("interpolated_model_QC")
    good = int(pc.sum(pc.less_equal(qc, 2)).as_py() or 0)
    return {"n_all": t.num_rows, "n_good": good, "n_failed": t.num_rows - good, "n_parquet_files": len(files)}


def op_run(args):
    from model2obs.workflows.workflow_model_obs import WorkflowModelObs

    config_path = os.path.abspath(os.path.expanduser(args["config_path"]))
    status_file = args.get("status_file")
    flags = {k: bool(args.get(k, d)) for k, d in
             (("trim_obs", True), ("no_matching", False), ("force_obs_time", False),
              ("parquet_only", False), ("clear_output", False), ("parallel", False))}
    status = {"status": "running", "config_path": config_path, "flags": flags, "pid": os.getpid(),
              "host": socket.gethostname(), "pbs_jobid": os.environ.get("PBS_JOBID"),
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_status(status_file, status)
    try:
        os.chdir(os.path.dirname(config_path))
        overrides = {}
        raw, cfg = _read_config_raw(config_path)
        if "netcdf_output_folder" not in raw:
            overrides["netcdf_output_folder"] = os.path.join(os.path.dirname(config_path), "netcdf_output")
        wf = WorkflowModelObs.from_config_file(config_path, **overrides)
        wf.run(**flags)
        counts = _parquet_counts(wf.config["parquet_folder"])
        status.update({"status": "done", "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "results": counts, "parquet_folder": wf.config["parquet_folder"],
                       "output_folder": wf.config["output_folder"], "log_file": wf.config.get("log_file")})
        _write_status(status_file, status)
        return status
    except BaseException as exc:
        status.update({"status": "failed", "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        _write_status(status_file, status)
        raise


def _parse_pair_summary(path):
    row = {"file": os.path.basename(path)}
    keymap = {"Pair number": "pair", "NC file submitted to DART": "model_file", "Original NC file": "original_model_file",
              "Time used": "time_used", "obs_seq.in file": "obs_seq_in", "Obs submitted to DART": "obs_submitted",
              "DART exit code": "dart_exit", "Successfully interpolated obs": "success",
              "Failed interpolations": "failed", "Interpolation counts": "interp_counts"}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if ":" not in line or line.startswith(("=", "-")):
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            if k in keymap:
                row[keymap[k]] = v.strip()
    return row


def op_run_status(args):
    import glob
    import subprocess

    config_path = os.path.abspath(os.path.expanduser(args["config_path"]))
    raw, cfg = _read_config_raw(config_path)
    out = {"config_path": config_path}
    sf = args.get("status_file")
    if sf and os.path.isfile(sf):
        with open(sf, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("traceback"):
            st["traceback"] = st["traceback"][-3000:]
        if st.get("status") == "running" and st.get("pid") and st.get("host") == socket.gethostname():
            try:
                os.kill(int(st["pid"]), 0)
                st["process_alive"] = True
            except OSError:
                st["process_alive"] = False
                st["note"] = "status says running but the process is gone (killed? OOM?)"
        out["status_file"] = sf
        out["status"] = st
    jobid = args.get("pbs_jobid") or (out.get("status") or {}).get("pbs_jobid")
    if jobid:
        try:
            q = subprocess.run(["qstat", "-x", jobid], capture_output=True, text=True, timeout=30)
            out["qstat"] = (q.stdout or q.stderr).strip()[-800:]
        except Exception as exc:  # noqa: BLE001
            out["qstat"] = f"qstat unavailable: {exc}"

    log_file = cfg.get("log_file")
    out["log_file"] = log_file
    if log_file and os.path.isfile(log_file):
        with open(log_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        n = int(args.get("tail", 30))
        out["log_tail"] = "".join(lines[-n:])
        out["log_warnings"] = [l.strip() for l in lines if "Warning" in l or "WARNING" in l][-10:]
        logs_dir = os.path.dirname(log_file)
        pairs = [_parse_pair_summary(p) for p in sorted(glob.glob(os.path.join(logs_dir, "pair_summary_*.log")))]
        out["pairs"] = pairs
        failed_logs = []
        for p in pairs:
            if "FAILED" in p.get("dart_exit", ""):
                num = p.get("pair", "")
                pl = os.path.join(logs_dir, f"perfect_model_obs_{num}.log")
                if os.path.isfile(pl):
                    with open(pl, encoding="utf-8", errors="replace") as f:
                        txt = f.readlines()
                    errs = [l.rstrip() for l in txt if "ERROR" in l or "FATAL" in l]
                    failed_logs.append({"log": pl, "error_lines": errs[:15] or [l.rstrip() for l in txt[-15:]]})
        out["failed_pair_logs"] = failed_logs
        out["n_pmo_logs"] = len(glob.glob(os.path.join(logs_dir, "perfect_model_obs_*.log")))
    elif log_file:
        out["note"] = "log_file does not exist yet (run not started, or a different log_file)"
    of = cfg.get("output_folder")
    if of and os.path.isdir(of):
        out["n_obs_seq_out"] = len(glob.glob(os.path.join(of, "obs_seq_*.out")))
        out["n_perfect_output_nc"] = len(glob.glob(os.path.join(of, "perfect_output_*.nc")))
    pf = cfg.get("parquet_folder")
    if pf and os.path.isdir(pf):
        try:
            out["parquet"] = _parquet_counts(pf)
        except Exception as exc:  # noqa: BLE001
            out["parquet"] = f"unreadable: {exc}"
    return _jsonable(out)


# ---------------------------------------------------------------- results

def _load_results(args, columns=None):
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as pads

    folder = os.path.abspath(os.path.expanduser(args["parquet_folder"]))
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.endswith(".parquet") and not f.startswith(("_", ".")))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {folder}")
    df = pads.dataset(files, format="parquet").to_table(columns=columns).to_pandas()
    n_total = len(df)

    if args.get("obs_type"):
        sel = str(args["obs_type"]).upper()
        if sel.startswith("ALL_"):
            sel = sel[4:]
        df = df[df["type"].astype(str).str.upper() == sel] if (df["type"].astype(str).str.upper() == sel).any() \
            else df[df["type"].astype(str).str.upper().str.contains(sel, regex=False)]
    if args.get("lat_min") is not None:
        df = df[df["latitude"] >= float(args["lat_min"])]
    if args.get("lat_max") is not None:
        df = df[df["latitude"] <= float(args["lat_max"])]
    if args.get("lon_min") is not None and args.get("lon_max") is not None:
        lon = ((df["longitude"].astype(float) + 180.0) % 360.0) - 180.0
        mask = np.zeros(len(df), dtype=bool)
        for lo, hi in _lon_ranges(args["lon_min"], args["lon_max"]):
            mask |= ((lon >= lo) & (lon <= hi)).to_numpy()
        df = df[mask]
    if args.get("depth_min") is not None:
        df = df[df["vertical"] >= float(args["depth_min"])]
    if args.get("depth_max") is not None:
        df = df[df["vertical"] <= float(args["depth_max"])]
    if args.get("start"):
        df = df[pd.to_datetime(df["time"]) >= pd.Timestamp(args["start"])]
    if args.get("end"):
        df = df[pd.to_datetime(df["time"]) < pd.Timestamp(args["end"])]

    df = df.copy()
    if args.get("salinity_to_psu", True) and len(df):
        sal = df["type"].astype(str).str.contains("SALINITY")
        if sal.any():
            for c in ("interpolated_model", "obs", "difference", "abs_difference"):
                if c in df:
                    good = sal & (df["interpolated_model_QC"] <= 2) if c == "interpolated_model" else sal
                    df.loc[good, c] = df.loc[good, c] * 1000.0
            # obs_err_var is left untouched: DART's CrocoLake converter scales PSAL
            # to kg/kg but writes PSAL_ERROR unscaled, so its units are not kg/kg.
            if "squared_difference" in df:
                df.loc[sal, "squared_difference"] = df.loc[sal, "squared_difference"] * 1.0e6
    return folder, n_total, df


def _qc_meaning(code):
    code = int(code)
    if code in QC_MEANINGS:
        return QC_MEANINGS[code]
    if 1 <= code <= 2:
        return "success (DART QC 1-2)"
    if code > 1000:
        return f"forward operator istatus {code - 1000} (see model_mod.f90 for this model)"
    return "DART QC code"


def op_summarize(args):
    import numpy as np
    import pandas as pd

    folder, n_total, df = _load_results(args)
    which = args.get("which", "good")
    good_mask = df["interpolated_model_QC"] <= 2
    out = {"parquet_folder": folder, "n_rows_in_dataset": n_total, "n_after_filters": int(len(df)),
           "n_good": int(good_mask.sum()), "n_failed": int((~good_mask).sum()), "which": which}

    fail = df[~good_mask]
    hist = fail["interpolated_model_QC"].round().astype(int).value_counts().sort_index()
    out["failed_qc_histogram"] = [{"qc": int(k), "count": int(v), "meaning": _qc_meaning(k)} for k, v in hist.items()]
    if len(fail):
        by_type = fail.groupby(["type", fail["interpolated_model_QC"].round().astype(int)]).size()
        out["failed_by_type"] = [{"type": t, "qc": int(q), "count": int(c)} for (t, q), c in by_type.items()]

    if which == "good":
        d = df[good_mask]
    elif which == "failed":
        d = fail
    else:
        d = df
    d = d.copy()

    group_by = list(args.get("group_by") or ["type"])
    if "depth_bin" in group_by:
        bins = args.get("depth_bins") or [0, 10, 50, 100, 200, 500, 1000, 2000, 6000]
        d["depth_bin"] = pd.cut(d["vertical"], bins=bins, right=False)
    elif args.get("depth_bins"):
        d["depth_bin"] = pd.cut(d["vertical"], bins=args["depth_bins"], right=False)
        group_by.append("depth_bin")
    if "day" in group_by:
        d["day"] = pd.to_datetime(d["time"]).dt.strftime("%Y-%m-%d")
    if "month" in group_by:
        d["month"] = pd.to_datetime(d["time"]).dt.strftime("%Y-%m")
    missing = [g for g in group_by if g not in d.columns]
    if missing:
        raise ValueError(f"group_by columns {missing} not available; use columns of the parquet or depth_bin/day/month")

    rows = []
    if which != "failed" and len(d):
        nd = d["normalized_difference"].replace([np.inf, -np.inf], np.nan)
        d = d.assign(_nd=nd)
        for keys, g in d.groupby(group_by, observed=True, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            diff = g["difference"].astype(float)
            r = dict(zip(group_by, [str(k) for k in keys]))
            r.update({
                "n": int(len(g)),
                "bias_obs_minus_model": float(diff.mean()),
                "std": float(diff.std(ddof=1)) if len(g) > 1 else None,
                "rmse": float(np.sqrt((diff ** 2).mean())),
                "mae": float(diff.abs().mean()),
                "mean_obs": float(g["obs"].mean()),
                "mean_model": float(g["interpolated_model"].mean()),
                "mean_normalized_difference": float(g["_nd"].mean()) if g["_nd"].notna().any() else None,
                "rms_normalized_difference": float(np.sqrt((g["_nd"] ** 2).mean())) if g["_nd"].notna().any() else None,
                "depth_range": [float(g["vertical"].min()), float(g["vertical"].max())],
            })
            rows.append(r)
    out["group_by"] = group_by
    out["stats"] = rows
    out["units_note"] = (("difference = obs - model. Salinity values converted from DART kg/kg to PSU (x1000)."
                          if args.get("salinity_to_psu", True) else
                          "difference = obs - model. Salinity left in DART units (kg/kg).")
                         + " normalized_difference = difference / sqrt(obs_err_var) as stored by model2obs (DART "
                         "units). For obs_seq files made by DART's CrocoLake converter, obs_err_var is the raw "
                         "CrocoLake *_ERROR value (instrument uncertainty, not rescaled; PSAL_ERROR stays in PSU "
                         "while salinity is kg/kg), so normalized_difference / log_likelihood are not a "
                         "calibrated misfit measure, and for salinity they are off by ~1e3.")
    return _jsonable(out)


def op_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    kind = args.get("kind", "map")
    field = args.get("field", "difference")
    if kind not in ("map", "profile", "scatter"):
        raise ValueError("kind must be 'map', 'profile' or 'scatter'")
    out_dir = os.path.abspath(os.path.expanduser(args["out_dir"]))
    _assert_writable_target(out_dir, "plot directory")
    os.makedirs(out_dir, exist_ok=True)

    folder, n_total, df = _load_results(args)
    df = df[df["interpolated_model_QC"] <= 2]
    if field not in df.columns:
        raise ValueError(f"field '{field}' not in parquet columns {list(df.columns)}")
    if not len(df):
        raise ValueError("no successfully interpolated rows after filters")
    df = df.assign(lon=((df["longitude"].astype(float) + 180.0) % 360.0) - 180.0)

    use_cartopy = bool(args.get("cartopy", True))
    ccrs = None
    if kind == "map" and use_cartopy:
        try:
            import cartopy.crs as ccrs  # noqa: F811
            import cartopy.feature as cfeature
        except Exception:  # noqa: BLE001
            ccrs = None

    diverging = field in ("difference", "normalized_difference")
    units = lambda t: "PSU" if "SALINITY" in t else ("degC" if "TEMPERATURE" in t else "")  # noqa: E731
    paths = []
    for t, g in df.groupby("type", sort=True):
        t = str(t)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in t)
        u = units(t) if field not in ("normalized_difference", "log_likelihood") else ""
        fname = os.path.join(out_dir, f"{kind}_{field}_{safe}.png")
        if kind == "map":
            agg = g.groupby(["lon", "latitude"], as_index=False)[field].mean()
            vals = agg[field].to_numpy(float)
            if diverging:
                vmax = float(np.nanpercentile(np.abs(vals), 98)) or 1.0
                norm = dict(vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            else:
                norm = dict(vmin=float(np.nanpercentile(vals, 2)), vmax=float(np.nanpercentile(vals, 98)), cmap="viridis")
            fig = plt.figure(figsize=(8, 6))
            if ccrs is not None:
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                sc = ax.scatter(agg["lon"], agg["latitude"], c=vals, s=18, transform=ccrs.PlateCarree(),
                                edgecolors="0.3", linewidths=0.3, **norm)
                try:
                    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
                    ax.coastlines(linewidth=0.6)
                except Exception:  # noqa: BLE001
                    pass
                pad = 1.0
                ax.set_extent([agg["lon"].min() - pad, agg["lon"].max() + pad,
                               agg["latitude"].min() - pad, agg["latitude"].max() + pad], crs=ccrs.PlateCarree())
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6")
                gl.top_labels = gl.right_labels = False
            else:
                ax = fig.add_subplot(1, 1, 1)
                sc = ax.scatter(agg["lon"], agg["latitude"], c=vals, s=18, edgecolors="0.3", linewidths=0.3, **norm)
                ax.set_xlabel("longitude")
                ax.set_ylabel("latitude")
                ax.grid(linewidth=0.3, color="0.85")
            cb = fig.colorbar(sc, ax=ax, shrink=0.8)
            cb.set_label(f"{field} {('[' + u + ']') if u else ''}".strip())
            dr = f"{g['vertical'].min():.0f}-{g['vertical'].max():.0f} m"
            ax.set_title(f"{t}: {field} (mean over depth {dr}), n={len(g)} obs at {len(agg)} locations", fontsize=10)
        elif kind == "profile":
            bins = args.get("depth_bins") or [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 6000]
            fig, ax = plt.subplots(figsize=(5, 7))
            ax.scatter(g[field], g["vertical"], s=4, color="0.75", alpha=0.5, label="obs points", rasterized=True)
            cut = pd.cut(g["vertical"], bins=bins, right=False)
            st = g.groupby(cut, observed=True)[field].agg(["mean", "std", "count"])
            mids = np.array([iv.mid for iv in st.index])
            ax.plot(st["mean"], mids, color="#1f6fb4", linewidth=2, marker="o", markersize=4, label="bin mean")
            ax.fill_betweenx(mids, st["mean"] - st["std"], st["mean"] + st["std"], color="#1f6fb4", alpha=0.15, label="±1 std")
            if diverging:
                ax.axvline(0, color="0.4", linewidth=0.8)
            ax.invert_yaxis()
            ax.set_ylabel("depth [m]")
            ax.set_xlabel(f"{field} {('[' + u + ']') if u else ''}".strip())
            ax.set_title(f"{t}: {field} vs depth (n={len(g)})", fontsize=10)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(linewidth=0.3, color="0.85")
        else:  # scatter obs vs model
            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(g["interpolated_model"], g["obs"], c=g["vertical"], s=5, cmap="viridis_r", rasterized=True)
            lo = float(np.nanmin([g["obs"].min(), g["interpolated_model"].min()]))
            hi = float(np.nanmax([g["obs"].max(), g["interpolated_model"].max()]))
            ax.plot([lo, hi], [lo, hi], color="0.4", linewidth=0.8)
            ax.set_xlabel(f"model {('[' + units(t) + ']') if units(t) else ''}".strip())
            ax.set_ylabel(f"obs {('[' + units(t) + ']') if units(t) else ''}".strip())
            fig.colorbar(sc, ax=ax, shrink=0.8).set_label("depth [m]")
            ax.set_title(f"{t}: obs vs model (n={len(g)})", fontsize=10)
            ax.grid(linewidth=0.3, color="0.85")
        fig.tight_layout()
        fig.savefig(fname, dpi=int(args.get("dpi", 120)))
        plt.close(fig)
        paths.append(fname)
    return {"paths": paths, "kind": kind, "field": field, "n_points": int(len(df)),
            "types": sorted(df["type"].astype(str).unique().tolist()),
            "cartopy_used": bool(ccrs is not None) if kind == "map" else None}


OPS = {
    "check_env": op_check_env,
    "list_obs_types": op_list_obs_types,
    "query_crocolake": op_query_crocolake,
    "make_obs_seq": op_make_obs_seq,
    "validate_config": op_validate_config,
    "write_config": op_write_config,
    "preview_namelist": op_preview_namelist,
    "run": op_run,
    "run_status": op_run_status,
    "summarize": op_summarize,
    "plot": op_plot,
}


def main():
    real_stdout = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # library prints + child processes -> stderr
    sys.stdout = sys.stderr
    try:
        op = sys.argv[1]
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        if op not in OPS:
            raise ValueError(f"unknown op '{op}' (have {sorted(OPS)})")
        result = OPS[op](args)
        real_stdout.write(json.dumps(_jsonable(result)) + "\n")
        real_stdout.flush()
        return 0
    except BaseException as exc:  # noqa: BLE001
        real_stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()}) + "\n")
        real_stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())
