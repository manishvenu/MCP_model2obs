# model2obs-mcp

MCP server for model2obs: compare regional ocean
model output (MOM6, ROMS_Rutgers) with in-situ observations from CrocoLake (Argo, GLODAP bottles,
Spray gliders, ...). DART's `perfect_model_obs` interpolates each model snapshot to the observation
locations. model2obs pairs the files, runs DART and writes a parquet table with `obs`,
`interpolated_model` and `difference` (= obs - model) for each observation. This server adds CrocoLake
queries, obs_seq generation, config writing and validation, run tracking, statistics and static plots.

## Interpreter split

The server process uses only the stdlib, `fastmcp` and `pyyaml`. It never imports model2obs, dask or
pandas. All library work runs as a subprocess:

```
$MODEL2OBS_PYTHON model2obs_mcp/_worker.py <op> '<json args>'   ->  one JSON document on stdout
```

| Env var | Meaning |
|---|---|
| `MODEL2OBS_PYTHON` | Python of the model2obs conda env (default: the server's own `sys.executable`), e.g. `/path/to/conda-envs/model2obs/bin/python` |
| `MODEL2OBS_MCP_PROTECTED_PATHS` | Extra colon-separated paths the server refuses to write into. `/glade/campaign` and `$CROCOLAKE_PATH` are always protected. |

The server is not launched from an activated env. So when `MODEL2OBS_PYTHON` is inside a conda env,
the server sources that env's `etc/conda/activate.d/*.sh` once in a throwaway bash, diffs the
environment, caches the result and merges it into every worker's environment. This provides
`DART_ROOT_PATH`, `CROCOLAKE_PATH`, `CROCOLAKE_OBS_CONV_PATH`, `TUTORIAL_DATA_PATH`, `PYTHONPATH`
(DART's `convert_crocolake_obs`) and `ncks` (via `module load nco`). If `module` is not defined, Lmod is
initialised from `/etc/profile.d/z00_modules.sh`. `check_environment` reports what was sourced and what
is missing.

**The shared DART `perfect_model_obs` is a Casper build.** Run comparisons on Casper: small runs can go
on the login node, larger ones need `run_comparison(background="pbs")` on the `casper` queue. From
Derecho, use `pbs_queue="casper@casper-pbs"`.

## Setup

```bash
pip install -e .            # into any env with Python >= 3.11 (e.g. the CrocoDash env)
```

`.mcp.json`:

```json
{
  "mcpServers": {
    "model2obs": {
      "command": "/path/to/env/bin/python",
      "args": ["/path/to/MCP_model2obs/server.py"],
      "env": {"MODEL2OBS_PYTHON": "/path/to/conda-envs/model2obs/bin/python"}
    }
  }
}
```

You can also run it directly with `model2obs-mcp`, `python -m model2obs_mcp` or `python server.py`.

## Tools

| Tool | Purpose |
|---|---|
| `check_environment()` | Checks the interpreters, activate.d vars, model2obs version, the `perfect_model_obs` binary (with ldd), `obs_def_ocean_mod.rst`, `ncks`, CrocoLake and the converter |
| `list_obs_types(quantity)` | Lists DART ocean obs types by QTY and the valid `ALL_*` selectors for `use_these_obs` |
| `query_crocolake(box, start, end, ...)` | Read-only CrocoLake summary: rows, profiles and platforms per DB_NAME, usable values, time and depth ranges, optional sample rows. Bounds are required. |
| `make_obs_seq_from_crocolake(out_path, box, start, end, ...)` | Writes one obs_seq file with DART's CrocoLake converter. Never overwrites an existing file. |
| `validate_config(config, ...)` | Static validation of a config, with no side effects. `check_pairs=True` dry-runs the time matching. |
| `write_config(path, values, output_root=...)` | Writes a YAML config from the library template and validates it |
| `preview_namelist(config)` | Shows the DART `input.nml` that model2obs would build, using a temp CWD |
| `run_comparison(config, background=False\|True\|"pbs", ...)` | Validates the config, then runs `WorkflowModelObs.run` in the foreground, as a detached process, or as a PBS job |
| `get_run_status(config)` | Shows the status JSON, the run-log tail, a per-pair table from `pair_summary_*.log`, DART errors for failed pairs, and output and parquet counts |
| `summarize_results(parquet_folder, group_by, ...)` | Bias, std, RMSE, MAE and normalized misfit per group (type, depth_bin, day, ...), plus a histogram of failed QC codes |
| `plot_results(parquet_folder, out_dir, kind)` | Static PNGs: `map`, `profile` (vs depth) and `scatter` (obs vs model), one per obs type |

## Workflow example

```
check_environment()
query_crocolake(15, 25, -162, -152, "2023-06-16", "2023-06-19")
make_obs_seq_from_crocolake(".../obs/CL.20230616.out", 15, 25, -162, -152, "2023-06-16", "2023-06-17")
  ... one file per model snapshot (here: per day), alone in the obs folder
write_config(".../config.yaml", {
    "model_files_folder": ".../in_mom6", "obs_seq_in_folder": ".../obs",
    "template_file": ".../hawaii.mom6.h.daily.2023-06-16.nc",
    "static_file": ".../hawaii.mom6.h.static.nc", "ocean_geometry": ".../ocean_geometry.nc",
    "perfect_model_obs_dir": "$DART_ROOT_PATH/models/MOM6/work",
    "use_these_obs": ["ALL_TEMPERATURE", "ALL_SALINITY"], "time_window": {"days": 2}},
  output_root=".../out")
validate_config(".../config.yaml", check_pairs=True)
run_comparison(".../config.yaml")                       # or background=True / "pbs"
get_run_status(".../config.yaml")
summarize_results(".../out/out_parquet", group_by=["type", "depth_bin"])
plot_results(".../out/out_parquet", ".../plots", kind="profile")
```

## Files the server writes

| File | Location | Written by |
|---|---|---|
| `<config stem>.mcp_status.json` | Next to the config | `run_comparison`. Kept out of the logs folder because `clear_output` empties that folder. |
| `<config stem>.mcp_run.log` | Next to the config | `background=True` runs |
| `<config stem>.mcp_pbs.sh` and `<config stem>.mcp_pbs.log` | Next to the config | `background="pbs"` runs |
| `input.nml.backup` and `netcdf_output/` | The config directory | The library itself. Runs use CWD = the config directory, and `netcdf_output_folder` is pinned there when the config leaves it unset. |

## Gotchas handled or reported

- Output folders must be empty. Reruns need `clear_output=True`, which deletes earlier results.
- `obs_seq_in_folder` is globbed with `*`, so any stray file breaks pairing. `validate_config` checks
  that every file in it is an obs_seq file.
- If `use_these_obs` is bad, the library silently falls back to the template's `obs_kind_nml`.
  `validate_config` reports this as an error.
- `use_pseudo_depth` must be true unless `h: QTY_LAYER_THICKNESS` is in the state.
- Salinity is in kg/kg inside DART. `summarize_results` and `plot_results` convert it to PSU.
- QC > 2 means the interpolation failed (`interpolated_model = -888888`). The QC value is
  1000 + DART istatus: 1014 = outside the grid, 1017 = land, 1018 = basin edge, 1022 = too deep.

## Design

Same pattern as the sibling MCP servers: FastMCP, stateless, file-backed, one plain function per tool
with the docstring as the tool description, and no CI.
