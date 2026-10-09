"""CrocoLake queries and obs_seq generation (read-only on CrocoLake)."""

from typing import Optional

from model2obs_mcp import _env


def query_crocolake(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start: str,
    end: str,
    db_names: Optional[list[str]] = None,
    variables: Optional[list[str]] = None,
    max_rows: int = 20,
    summary_only: bool = True,
    crocolake_path: Optional[str] = None,
    allow_large: bool = False,
) -> dict:
    """
    Summarise CrocoLake observations in a lat/lon box and time window (read-only).

    CrocoLake ($CROCOLAKE_PATH, ~23 GB of parquet) is read with pyarrow
    predicate push-down on LATITUDE/LONGITUDE/JULD (and DB_NAME), so bounds are
    always required. Use this before make_obs_seq_from_crocolake to see whether
    a region/period has data and how much.

    Returns: total rows; per DB_NAME (ARGO, GLODAP, SprayGliders, ...) the
    number of rows, profiles (unique PLATFORM_NUMBER+CYCLE_NUMBER) and
    platforms, and for each variable the non-null count and the count that also
    has _QC and _ERROR (what make_obs_seq_from_crocolake keeps with
    loose=False); overall time / lat / lon / pressure (dbar) / depth ranges; and
    optionally a small sample of rows.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in degrees north (inclusive).
    lon_min, lon_max : float
        Longitude bounds. CrocoLake stores -180..180; 0..360 input is accepted
        and a box crossing the dateline (lon_min > lon_max after normalising,
        e.g. 170 -> -170) is split into two ranges.
    start, end : str
        Time window, any pandas-parsable timestamp ("2023-06-16",
        "2023-06-16T12:00"). start inclusive, end exclusive.
    db_names : list[str], optional
        Restrict to these DB_NAME values (e.g. ["ARGO"]). Default: all.
    variables : list[str], optional
        CrocoLake variables to count (default ["TEMP", "PSAL"]).
    max_rows : int
        Number of sample rows to return when summary_only=False (capped at 200).
    summary_only : bool
        If True (default) return only counts/ranges, no row sample.
    crocolake_path : str, optional
        Override $CROCOLAKE_PATH.
    allow_large : bool
        The query is refused when box-area x window-length exceeds 50,000
        deg^2*days (e.g. 10x10 deg for 500 days), to protect the login node's
        memory. Set True to bypass.
    """
    return _env.run_worker(
        "query_crocolake",
        dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, start=start, end=end,
             db_names=db_names, variables=variables, max_rows=max_rows, summary_only=summary_only,
             crocolake_path=crocolake_path, allow_large=allow_large),
        timeout=1800,
    )


def make_obs_seq_from_crocolake(
    out_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start: str,
    end: str,
    variables: Optional[list[str]] = None,
    db_names: Optional[list[str]] = None,
    loose: bool = False,
    fill_na_qc: Optional[int] = None,
    fill_na_error: Optional[float] = None,
    crocolake_path: Optional[str] = None,
) -> dict:
    """
    Write ONE DART obs_seq file from CrocoLake observations in a box/time window.

    Uses DART's own converter (observations/obs_converters/CrocoLake/
    convert_crocolake_obs.ObsSequence + write_obs_seq), the same code the
    model2obs tutorials use. Pressure is converted to depth with gsw, longitude
    to 0..360 radians, and salinity from PSU to kg/kg (DART convention).
    Observation types are <SOURCE>_<FIELD> (ARGO_, BOTTLE_ for GLODAP, GLIDER_
    for SprayGliders) or plain TEMPERATURE / SALINITY for other sources.

    model2obs pairs each model snapshot with ONE obs_seq file whose whole time
    range fits inside `time_window` centred on the snapshot time, so make one
    file per model snapshot (e.g. one per day for daily output) and put them
    all -- and nothing else -- in the config's obs_seq_in_folder.

    Safety: refuses to overwrite an existing out_path and refuses to write under
    /glade/campaign or $CROCOLAKE_PATH. The converter's temporary .tmp file is
    staged in a temp dir, so no stray file lands in your obs folder. If no
    observations match, nothing is written ("written": False).

    Parameters
    ----------
    out_path : str
        obs_seq file to create (parent dirs are created).
    lat_min, lat_max, lon_min, lon_max : float
        Box (degrees; lon -180..180 or 0..360, dateline crossing supported).
    start, end : str
        Time window; start inclusive, end exclusive.
    variables : list[str]
        CrocoLake variables to convert (default ["TEMP", "PSAL"]). Supported by
        the converter: TEMP, PSAL, DOXY, TOT_ALKALINITY, TCO2, NITRATE, SILICATE,
        PHOSPHATE (the BGC ones only exist in a BGC CrocoLake).
    db_names : list[str], optional
        Restrict to these DB_NAME values (e.g. ["ARGO"]).
    loose : bool
        False (default): keep only values that have both a QC flag and an error.
        True: keep every non-null value; missing errors are written as -888888,
        which makes normalized_difference / log_likelihood NaN for those obs.
    fill_na_qc, fill_na_error : optional
        Replace missing QC flags / errors with these values before conversion.
    crocolake_path : str, optional
        Override $CROCOLAKE_PATH.
    """
    return _env.run_worker(
        "make_obs_seq",
        dict(out_path=out_path, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
             start=start, end=end, variables=variables, db_names=db_names, loose=loose,
             fill_na_qc=fill_na_qc, fill_na_error=fill_na_error, crocolake_path=crocolake_path),
        timeout=3600,
    )
