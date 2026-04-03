"""Per-TSO SMARD data collector using the www.smard.de GET API.

Fetches per-TSO generation and load data and stores as a single parquet file.
Supports incremental updates and saves after each column for crash resilience.

Usage via update_database.py:
    python update_database.py DE update_smard_v2 hourly
"""

import os

import pandas as pd
import requests

from data_collection_modules.parquet_operations import ParquetOperations

from logger import get_logger
logger = get_logger(__name__)

BASE_URL = "https://www.smard.de/app/chart_data"

# TSO region -> column suffix mapping (matches eu_locations.py conventions)
TSO_SUFFIX = {
    "50Hertz": "_50hz",
    "Amprion": "_ampr",
    "TenneT": "_tenn",
    "TransnetBW": "_tran",
    "Creos": "_lu",
}

# Filter -> column base name mapping
# Clean 1:1 mappings with ENTSO-E
FILTER_COLUMNS = {
    1225: "wind_offshore",
    4067: "wind_onshore",
    4068: "solar",
    410: "load",
    4066: "biomass",
    4071: "gas",
    4069: "hard_coal",
    1223: "lignite",
    4070: "pumped_storage",
    # Merged categories (no ENTSO-E 1:1 equivalent)
    1226: "hydro",         # run_of_river + water_reservoir
    1227: "other_conv",    # oil + other_fossil
    1228: "other_renew",   # other_renewables + geothermal + waste
}

# Filter/region combos known to 404 (from API spike)
KNOWN_MISSING = {
    (1223, "TransnetBW"),   # no lignite in TransnetBW
    (1225, "Amprion"),      # no offshore wind in Amprion
    (1225, "TransnetBW"),   # no offshore wind in TransnetBW
    (1225, "Creos"),        # no offshore wind in Luxembourg
    (4069, "Creos"),        # no hard coal in Luxembourg
    (1223, "Creos"),        # no lignite in Luxembourg
    (4070, "Creos"),        # no pumped storage in Luxembourg
    (1228, "Creos"),        # no other renewables in Luxembourg
}

# Resolution mapping
RESOLUTION_MAP = {
    "hourly": "hour",
    "minutely_15": "quarterhour",
}


def get_timestamps(filter_id, region, resolution="hour"):
    """Fetch available weekly chunk timestamps for a filter/region combo.

    Returns list of millisecond timestamps, or None if 404.
    """
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("timestamps", [])


def get_chunk(filter_id, region, timestamp, resolution="hour"):
    """Fetch one weekly data chunk. Returns list of [timestamp_ms, value] pairs."""
    url = (
        f"{BASE_URL}/{filter_id}/{region}/"
        f"{filter_id}_{region}_{resolution}_{timestamp}.json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("series", [])


def fetch_filter_region(filter_id, region, resolution="hour",
                        after_ts=None):
    """Fetch all data for a single filter/region combo.

    Args:
        filter_id: SMARD filter code
        region: TSO region name (e.g. "50Hertz")
        resolution: "hour" or "quarterhour"
        after_ts: if set, only fetch chunks whose timestamp > this value (ms)

    Returns:
        pd.Series with UTC DatetimeIndex, or None if no data.
    """
    timestamps = get_timestamps(filter_id, region, resolution)
    if timestamps is None:
        return None

    if after_ts is not None:
        # Chunk timestamps mark week start; include the chunk containing the
        # cutoff by subtracting one week so partial-week updates are re-fetched.
        week_ms = 7 * 24 * 3600 * 1000
        timestamps = [ts for ts in timestamps if ts > after_ts - week_ms]

    if not timestamps:
        return None

    all_points = []
    for ts in timestamps:
        try:
            series = get_chunk(filter_id, region, ts, resolution)
            all_points.extend(series)
        except Exception as e:
            logger.warning(f"Failed to fetch chunk {filter_id}/{region}/{ts}: {e}")

    if not all_points:
        return None

    df = pd.DataFrame(all_points, columns=["timestamp_ms", "value"])
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df["value"]


def _merge_column_into_parquet(fname, col_name, series):
    """Merge a single new column/series into an existing parquet file (or create it)."""
    if os.path.isfile(fname):
        df = ParquetOperations.read(fname)
        # Extend index to cover new timestamps before merging
        new_idx = series.index.difference(df.index)
        if len(new_idx) > 0:
            df = df.reindex(df.index.union(new_idx))
        if col_name in df.columns:
            # Update: preserve existing data, overwrite only where new data exists
            df[col_name] = series.combine_first(df[col_name])
        else:
            df[col_name] = series
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
    else:
        df = pd.DataFrame({col_name: series})
        df.index.name = "date"
        df = df.sort_index()

    ParquetOperations.save(df, fname)
    return df.shape


def update_smard_v2(today, data_dir, freq, verbose=True, start_from=None):
    """Main entry point: create or incrementally update the SMARD v2 parquet file.

    Saves progress after each filter/region combo so the process can be
    interrupted and resumed without losing work.

    Args:
        today: current timestamp (pd.Timestamp, UTC)
        data_dir: path to database/DE/smard_v2/
        freq: "hourly" or "minutely_15"
        verbose: log progress
        start_from: if set, only fetch data from this date onward
                    on initial download. Ignored on incremental updates.
    """
    resolution = RESOLUTION_MAP[freq]
    fname = os.path.join(data_dir, f"history_{freq}.parquet")
    os.makedirs(data_dir, exist_ok=True)

    # Determine the after_ts cutoff
    after_ts = None
    if os.path.isfile(fname):
        df_existing = ParquetOperations.read(fname)
        existing_cols = set(df_existing.columns)
        last_idx = df_existing.dropna(how="all").index.max()
        if pd.notna(last_idx):
            overlap = pd.Timedelta(hours=72)
            cutoff = pd.Timestamp(last_idx) - overlap
            after_ts = int(cutoff.timestamp() * 1000)
            if verbose:
                logger.info(f"Incremental update from {cutoff} "
                            f"(existing data to {last_idx}, {len(df_existing)} rows, "
                            f"{len(existing_cols)} cols)")
        del df_existing
    else:
        existing_cols = set()
        if start_from is not None:
            after_ts = int(pd.Timestamp(start_from, tz="UTC").timestamp() * 1000)
            if verbose:
                logger.info(f"Initial download from {start_from}")

    # Build the list of columns to fetch
    combos = []
    for filter_id, col_base in FILTER_COLUMNS.items():
        for region, suffix in TSO_SUFFIX.items():
            col_name = col_base + suffix
            if (filter_id, region) in KNOWN_MISSING:
                continue
            combos.append((filter_id, region, col_name))

    # On initial download, skip columns already in the parquet (resume support)
    if existing_cols and after_ts is None:
        combos = [(f, r, c) for f, r, c in combos if c not in existing_cols]
        if verbose and len(combos) < len(FILTER_COLUMNS) * len(TSO_SUFFIX):
            logger.info(f"Resuming: {len(existing_cols)} cols already present, "
                        f"{len(combos)} remaining")

    # Split into existing (incremental) and new (full history) columns
    new_combos = [(f, r, c) for f, r, c in combos if c not in existing_cols]
    existing_combos = [(f, r, c) for f, r, c in combos if c in existing_cols]
    if new_combos and after_ts is not None and verbose:
        logger.info(f"New columns detected: {[c for _, _, c in new_combos]}. "
                    f"Fetching full history for these.")

    total = len(combos)
    if total == 0:
        if verbose:
            logger.info("All columns already present, nothing to fetch")
        return

    for i, (filter_id, region, col_name) in enumerate(combos, 1):
        if verbose:
            logger.info(f"[{i}/{total}] {col_name}: fetching...")

        # New columns get full history; existing columns get incremental
        ts_cutoff = None if col_name not in existing_cols else after_ts
        series = fetch_filter_region(filter_id, region, resolution, ts_cutoff)

        if series is not None and len(series) > 0:
            shape = _merge_column_into_parquet(fname, col_name, series)
            if verbose:
                logger.info(f"  -> {len(series)} rows, saved ({shape[0]}x{shape[1]} total)")
        else:
            if verbose:
                logger.info(f"  -> no new data")

    if verbose and os.path.isfile(fname):
        df = ParquetOperations.read(fname)
        logger.info(f"Done: {fname}: {df.shape[0]} rows x {df.shape[1]} cols, "
                    f"{df.index.min()} to {df.index.max()}")
