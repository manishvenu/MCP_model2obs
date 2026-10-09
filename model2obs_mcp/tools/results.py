"""Statistics and static plots from model2obs model-obs parquet output."""

from typing import Optional

from model2obs_mcp import _env


def summarize_results(
    parquet_folder: str,
    group_by: Optional[list[str]] = None,
    which: str = "good",
    obs_type: Optional[str] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    depth_bins: Optional[list[float]] = None,
    salinity_to_psu: bool = True,
) -> dict:
    """
    Model-vs-observation statistics from a model2obs parquet_folder.

    Reads parquet_folder/model-obs-*.parquet (the run_comparison output) and,
    after optional filters, returns per-group statistics of
    difference = obs - interpolated_model:
      n, bias_obs_minus_model (mean difference; positive = model too low),
      std, rmse, mae, mean_obs, mean_model, mean_normalized_difference and
      rms_normalized_difference (difference / sqrt(obs_err_var); RMS ~ 1 means
      misfits are consistent with the stated obs errors), depth_range.

    Always also returns n_good / n_failed and a histogram of the failed
    interpolated_model_QC codes with their meaning. perfect_model_obs writes
    QC 0 on success and 1000 + istatus on failure, e.g. 1000 = obs type not in
    use_these_obs, 1013 = layer thickness not in state (needs use_pseudo_depth),
    1014 = outside model grid, 1017 = on land, 1018 = on basin edge / not in
    basin, 1022 = deeper than the model column. Failed rows carry
    interpolated_model = -888888, so never average them.

    Units: DART stores salinity in kg/kg; with salinity_to_psu=True (default)
    salinity obs/model/difference are multiplied by 1000 (PSU) and their
    squared/variance columns by 1e6. Temperature is degC. vertical is depth in
    metres (positive down). longitude is 0..360 in the parquet; lon filters
    accept -180..180 or 0..360.

    Parameters
    ----------
    parquet_folder : str
        The config's parquet_folder.
    group_by : list[str]
        Columns to group by (default ["type"]). Besides any parquet column you
        can use "depth_bin" (uses depth_bins), "day" or "month".
    which : str
        "good" (QC <= 2, default), "failed" (only the QC histogram is
        meaningful) or "all".
    obs_type : str, optional
        Restrict to one type (exact, e.g. "ARGO_TEMPERATURE") or a substring /
        ALL_ selector ("TEMPERATURE", "ALL_SALINITY").
    lat_min, lat_max, lon_min, lon_max, depth_min, depth_max : float, optional
        Spatial / depth (m) filters.
    start, end : str, optional
        Time filter (start inclusive, end exclusive).
    depth_bins : list[float], optional
        Bin edges in metres; if given, "depth_bin" is added to group_by.
        Default edges when group_by has "depth_bin":
        [0, 10, 50, 100, 200, 500, 1000, 2000, 6000].
    salinity_to_psu : bool
        Convert salinity to PSU (default True).
    """
    return _env.run_worker(
        "summarize",
        dict(parquet_folder=parquet_folder, group_by=group_by, which=which, obs_type=obs_type,
             lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
             depth_min=depth_min, depth_max=depth_max, start=start, end=end,
             depth_bins=depth_bins, salinity_to_psu=salinity_to_psu),
        timeout=1800,
    )


def plot_results(
    parquet_folder: str,
    out_dir: str,
    kind: str = "map",
    obs_type: Optional[str] = None,
    field: str = "difference",
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    depth_bins: Optional[list[float]] = None,
    cartopy: bool = True,
    salinity_to_psu: bool = True,
) -> dict:
    """
    Save static PNG plots of model2obs results (one figure per observation type).

    model2obs's own viz module is ipywidgets-only (notebooks); this produces
    plain matplotlib PNGs instead. Only successfully interpolated rows
    (QC <= 2) are plotted. Kinds:
      "map"      each profile location coloured by the mean of `field` over the
                 selected depth range (use depth_min/depth_max to pick a layer);
                 diverging RdBu_r centred on 0 for difference /
                 normalized_difference. Uses cartopy (land + coastlines) when
                 available and cartopy=True, plain lon/lat axes otherwise.
      "profile"  `field` vs depth: all points in grey plus depth-binned mean
                 and +-1 std band (depth_bins to override the bin edges).
      "scatter"  obs vs interpolated model, coloured by depth, with the 1:1 line.

    Files are written as <out_dir>/<kind>_<field>_<TYPE>.png (existing files
    with the same name are overwritten); out_dir may not be under a protected
    path. Returns {"paths": [...], "types", "n_points", "cartopy_used"}.

    Parameters
    ----------
    parquet_folder : str
        The config's parquet_folder.
    out_dir : str
        Directory for the PNGs (created if needed).
    kind : str
        "map" (default), "profile" or "scatter".
    obs_type : str, optional
        Exact type or substring/ALL_ selector; default: every type, one PNG each.
    field : str
        Column to plot for map/profile: "difference" (default, obs - model),
        "normalized_difference", "abs_difference", "obs", "interpolated_model",
        "log_likelihood".
    depth_min, depth_max, lat_min, lat_max, lon_min, lon_max, start, end : optional
        Filters, as in summarize_results.
    depth_bins : list[float], optional
        Profile bin edges in metres.
    cartopy : bool
        Use cartopy for maps if importable (default True).
    salinity_to_psu : bool
        Convert salinity from DART kg/kg to PSU (default True).
    """
    return _env.run_worker(
        "plot",
        dict(parquet_folder=parquet_folder, out_dir=out_dir, kind=kind, obs_type=obs_type, field=field,
             depth_min=depth_min, depth_max=depth_max, lat_min=lat_min, lat_max=lat_max,
             lon_min=lon_min, lon_max=lon_max, start=start, end=end, depth_bins=depth_bins,
             cartopy=cartopy, salinity_to_psu=salinity_to_psu),
        timeout=1800,
    )
