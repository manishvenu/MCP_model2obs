"""model2obs YAML config: validate, write, preview the DART namelist."""

from typing import Optional

from model2obs_mcp import _env


def validate_config(
    config_path: str,
    trim_obs: bool = True,
    clear_output: bool = False,
    parquet_only: bool = False,
    check_pairs: bool = False,
) -> dict:
    """
    Validate a model2obs YAML config WITHOUT constructing WorkflowModelObs.

    The library's own validation is spread across the constructor and run()
    and has side effects (creates ./netcdf_output, deletes ./perfect_model_obs.log,
    creates output folders), so this re-implements the checks read-only:

      - read_config semantics: every string value except layer_name /
        ocean_model gets $VAR expansion and is resolved relative to the config
        file's directory; unexpanded $VARs are errors
      - time_window present and non-zero (years/months/weeks/days/hours/
        minutes/seconds; 1 month = 30 d, 1 year = 365 d)
      - required keys for the ocean_model adapter (MOM6: model_files_folder,
        obs_seq_in_folder, output_folder, template_file, static_file,
        ocean_geometry, perfect_model_obs_dir, parquet_folder) plus log_file and
        tmp_folder (read by run() with no default); input_nml_bck and
        trimmed_obs_folder must be explicit (library defaults are CWD-relative
        and buggy)
      - model_files_folder exists and holds only .nc files; obs_seq_in_folder
        exists and every file in it looks like an ASCII obs_seq (it is globbed
        with '*', so stray .tmp/other files break pairing)
      - template/static/ocean_geometry .nc files exist
      - use_pseudo_depth must be true unless model_state_variables has
        h: QTY_LAYER_THICKNESS
      - perfect_model_obs exists and is executable; use_these_obs expands
        against obs_def_ocean_mod.rst (the library silently ignores a bad list)
      - output folders (output_folder, tmp_folder, trimmed_obs_folder,
        input_nml_bck, parquet_folder) must be empty or absent unless
        clear_output=True; log_file must not be inside any of them; nothing may
        be under a protected path (/glade/campaign, $CROCOLAKE_PATH)
      - check_pairs=True additionally dry-runs the library's time matching:
        reads model snapshot times and each obs_seq's time range and reports
        which snapshot pairs with which obs_seq file (slower: parses every
        obs_seq file)

    Returns {"valid", "errors", "warnings", "info", "resolved_config"}.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.
    trim_obs, clear_output, parquet_only : bool
        The run_comparison flags you intend to use (they change which output
        folder states are acceptable).
    check_pairs : bool
        Dry-run the model-snapshot / obs_seq time matching (default False).
    """
    return _env.run_worker(
        "validate_config",
        dict(config_path=config_path, trim_obs=trim_obs, clear_output=clear_output,
             parquet_only=parquet_only, check_pairs=check_pairs),
        timeout=900,
    )


def write_config(
    path: str,
    values: dict,
    template: str = "MOM6",
    output_root: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Write a model2obs YAML config from the library's template plus your values.

    Starts from model2obs's tutorials/config_templates/config_template_<template>.yaml
    (template="MOM6" or "ROMS_Rutgers", or a path to any YAML to use as base,
    or "none" for an empty base), overlays `values` (top-level keys replace the
    template's), then:
      - output_root, if given, fills every output key you did not set:
        output_folder=<root>/out_obs_seq, input_nml_bck=<root>/input_nml_bckp,
        trimmed_obs_folder=<root>/out_trimmed_obs_seq, parquet_folder=<root>/out_parquet,
        tmp_folder=<root>/tmp, netcdf_output_folder=<root>/out_netcdf,
        log_file=<root>/logs/model2obs.log (recommended -- otherwise the
        template's $WORK/model2obs_output/workflow_1/... defaults remain)
      - ensures log_file and tmp_folder exist (next to output_folder, never
        inside it) and time_window exists ({days: 1} if absent)
      - drops the template's unused nested `logging:` key
    Comments from the template are not preserved (yaml.safe_dump). Keys you
    will typically set: model_files_folder, obs_seq_in_folder, template_file,
    static_file, ocean_geometry, perfect_model_obs_dir (e.g.
    "$DART_ROOT_PATH/models/MOM6/work"), use_these_obs (e.g. ["ALL_TEMPERATURE",
    "ALL_SALINITY"]), time_window (e.g. {"days": 2}), model_state_variables,
    use_pseudo_depth, layer_name. $VARs are kept verbatim and expanded at read time.

    Refuses to overwrite an existing file unless overwrite=True, and refuses to
    write under protected paths. Returns {"path", "template", "notes",
    "config", "validation"} where validation is validate_config(path).

    Parameters
    ----------
    path : str
        Where to write the YAML.
    values : dict
        Config keys to set.
    template : str
        "MOM6" (default), "ROMS_Rutgers", a YAML path, or "none".
    output_root : str, optional
        Directory under which all outputs are placed (see above).
    overwrite : bool
        Replace an existing file at `path`.
    """
    return _env.run_worker(
        "write_config",
        dict(path=path, values=values, template=template, output_root=output_root, overwrite=overwrite),
        timeout=600,
    )


def preview_namelist(config_path: str, out_file: Optional[str] = None) -> dict:
    """
    Show the DART input.nml that model2obs would build from a config.

    Constructs WorkflowModelObs and calls its preview_namelist() inside a
    throwaway temporary working directory, so the constructor's side effects
    (./netcdf_output, ./input.nml.backup, deleting ./perfect_model_obs.log)
    land there and are removed. The result reflects model_nml (template_file,
    static_file, ocean_geometry, model_state_variables, use_pseudo_depth,
    layer_name, assimilation period from time_window) and obs_kind_nml
    (expanded use_these_obs). Pair-specific entries (input_state_files,
    obs_seq_in/out_file_name, init_time_days/seconds) are filled per pair at
    run time. Note: the constructor does not create output folders, so this is
    safe to call on a config whose outputs do not exist yet.

    Parameters
    ----------
    config_path : str
        Path to the YAML config.
    out_file : str, optional
        Also write the namelist text to this file.
    """
    return _env.run_worker("preview_namelist", {"config_path": config_path, "out_file": out_file}, timeout=300)
