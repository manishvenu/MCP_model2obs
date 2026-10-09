"""
model2obs MCP -- model-vs-observation comparison for regional ocean runs.

Wraps model2obs (WorkflowModelObs + DART perfect_model_obs) and CrocoLake.
The server process itself only uses the stdlib + fastmcp; all library work
runs in MODEL2OBS_PYTHON via model2obs_mcp/_worker.py.
"""

from fastmcp import FastMCP

from model2obs_mcp.tools.environment import check_environment, list_obs_types
from model2obs_mcp.tools.crocolake import query_crocolake, make_obs_seq_from_crocolake
from model2obs_mcp.tools.config import validate_config, write_config, preview_namelist
from model2obs_mcp.tools.run import run_comparison, get_run_status
from model2obs_mcp.tools.results import summarize_results, plot_results

INSTRUCTIONS = """\
model2obs MCP server -- compare regional ocean model output (MOM6, ROMS_Rutgers)
with in-situ observations from CrocoLake (Argo, GLODAP bottles, Spray gliders, ...).
DART's perfect_model_obs interpolates each model snapshot to the observation
locations; model2obs pairs files, runs DART and writes a parquet table of
obs, interpolated_model and difference (= obs - model) per observation.

== ENVIRONMENT ==
The library runs in a separate Python: set MODEL2OBS_PYTHON to the model2obs
conda env's python (e.g. /path/to/conda-envs/model2obs/bin/python).
The env's etc/conda/activate.d/*.sh scripts are sourced automatically to get
DART_ROOT_PATH, CROCOLAKE_PATH, CROCOLAKE_OBS_CONV_PATH, TUTORIAL_DATA_PATH,
PYTHONPATH and ncks (module load nco). The shared DART perfect_model_obs is a
CASPER build: run comparisons on Casper (small runs on the login node) or with
run_comparison(background="pbs") on the casper queue.
Start every session with check_environment().

== WORKFLOW ==
1. check_environment()                         -- DART binary, ncks, CrocoLake, converter
2. query_crocolake(box, start, end)            -- is there data? how much?
3. make_obs_seq_from_crocolake(out_path, ...)  -- ONE obs_seq file per model snapshot
   (e.g. one per day for daily output) into a dedicated obs folder that holds
   nothing else. list_obs_types() shows DART type names / ALL_* selectors.
4. write_config(path, values, output_root=...) -- YAML from the library template;
   returns validate_config() of the result. Or validate_config(existing.yaml).
   validate_config(..., check_pairs=True) dry-runs the snapshot/obs time matching.
5. preview_namelist(config)                    -- optional: inspect DART input.nml
6. run_comparison(config, background=False|True|"pbs")
7. get_run_status(config)                      -- per-pair DART results, errors
8. summarize_results(parquet_folder, group_by=["type","depth_bin"])
   plot_results(parquet_folder, out_dir, kind="map"|"profile"|"scatter")

== GOTCHAS ==
- Every output folder must be empty (or absent) before a run; reruns need
  clear_output=True, which DELETES previous outputs -- only on explicit request.
- obs_seq_in_folder is globbed with '*': any stray file breaks pairing.
- An obs_seq file pairs with a snapshot only if its whole time range lies in
  time_window centred on the snapshot time (daily files at 00:00 -> days: 2).
- use_pseudo_depth must be true unless h: QTY_LAYER_THICKNESS is in
  model_state_variables.
- Salinity is kg/kg inside DART; summarize/plot convert to PSU by default.
- QC > 2 = failed interpolation (interpolated_model = -888888); 1000+code gives
  the DART reason (1018 = not in basin, 1017 = land, 1022 = too deep).
- Nothing is ever written under /glade/campaign or $CROCOLAKE_PATH.
"""

mcp = FastMCP("model2obs", instructions=INSTRUCTIONS)

mcp.add_tool(check_environment)
mcp.add_tool(list_obs_types)
mcp.add_tool(query_crocolake)
mcp.add_tool(make_obs_seq_from_crocolake)
mcp.add_tool(validate_config)
mcp.add_tool(write_config)
mcp.add_tool(preview_namelist)
mcp.add_tool(run_comparison)
mcp.add_tool(get_run_status)
mcp.add_tool(summarize_results)
mcp.add_tool(plot_results)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
