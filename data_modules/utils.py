from typing import Callable

import numpy as np
import pandas as pd

from logger import get_logger
logger = get_logger(__name__)

def handle_nans_with_interpolation(
    df: pd.DataFrame, name: str, log_func: Callable[..., None], max_gap: int = 48
) -> pd.DataFrame:
    """
    Checks each column of the DataFrame for NaNs. Logs gaps > 3 consecutive NaNs,
    raises ValueError for gaps exceeding max_gap. Fills remaining NaNs using
    bi-directional linear interpolation.

    Args:
        max_gap: Maximum allowed consecutive NaN gap (hours). Gaps exceeding this
            indicate a structural data issue rather than a transient collection gap.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        consecutive_nans = (
            df_copy[col].isna().astype(int)
            .groupby((~df_copy[col].isna()).cumsum())
            .cumsum()
        )
        max_consecutive = int(consecutive_nans.max())
        if max_consecutive > max_gap:
            raise ValueError(
                f"Column '{col}' in {name} has {max_consecutive} consecutive NaNs "
                f"(max allowed: {max_gap}). This indicates a structural data gap, "
                f"not a transient collection issue."
            )
        if max_consecutive > 3:
            log_func(
                f"Column '{col}' in {name} contains {max_consecutive} consecutive NaNs."
            )

    df_copy = df_copy.interpolate(method='linear', limit_direction='both', axis=0)
    return df_copy

def fix_broken_periodicity_with_interpolation(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Fixes broken hourly periodicity by adding missing timestamps if fewer than 3 consecutive are missing.
    Raises an error if more than 3 consecutive timestamps are missing.
    Missing values are filled using time-based interpolation.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"The DataFrame {name} must have a datetime index.")

    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    missing_timestamps = expected_index.difference(df.index)

    if missing_timestamps.empty:
        logger.info(f"The DataFrame {name} is already hourly with no missing segments.")
        return df

    # Convert to a Series to check consecutive missing timestamps
    missing_series = pd.Series(missing_timestamps)
    groups = (missing_series.diff() != pd.Timedelta(hours=1)).cumsum()

    # Check if any group has more than 3 missing points
    group_counts = groups.value_counts()
    if (group_counts > 3).any():
        bad_group = group_counts[group_counts > 3].index[0]
        raise ValueError(f"More than 3 consecutive missing timestamps detected: "
                         f"{missing_series[groups == bad_group].values} in {name}")

    # Reindex and interpolate
    fixed_df = df.reindex(expected_index)
    fixed_df = fixed_df.interpolate(method='time')

    logger.info(f"Added and interpolated {len(missing_timestamps)} missing timestamps in {name}.")

    return fixed_df

def validate_dataframe(df: pd.DataFrame, name: str, log_func:Callable[...,None], verbose:bool=False) -> pd.DataFrame:
    """Check for NaNs, missing values, and periodicity in a time-series DataFrame."""

    # Check for NaNs
    if df.isnull().any().any():
        if verbose: logger.error(f"{name} DataFrame contains NaN values.")
        df = handle_nans_with_interpolation(df, name, log_func)

    # Check if index is sorted in ascending order
    if not df.index.is_monotonic_increasing:
        if verbose: logger.error(f"{name} The index is not in ascending order.")
        raise ValueError("Data is not in ascending order.")

    # Check for hourly frequency with no missing segments
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    if not full_range.equals(df.index):
        if verbose: logger.error(f"{name} The data is not hourly or has missing segments.")
        df = fix_broken_periodicity_with_interpolation(df, name)

    return df


def merge_tso_dataframes(
    parts: list[pd.DataFrame],
    label: str = "",
) -> pd.DataFrame:
    """Merge multiple TSO DataFrames on their datetime index using intersection.

    Instead of left-joining onto the first DataFrame (which silently introduces
    NaN when TSOs have different date ranges), this finds the common date range
    across all parts, trims to that range, and inner-joins.

    Any trimming is logged as a warning. Raises ValueError only when there is
    no overlapping range at all. Interior NaN gaps (within the common range) are
    logged but not filled — downstream callers (handle_nans_with_interpolation)
    enforce the max gap limit.

    Args:
        parts: List of DataFrames with DatetimeIndex, one per TSO.
        label: Descriptive label for log messages (e.g. "onshore/history").

    Returns:
        Merged DataFrame covering the common date range of all inputs.
    """
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0].copy()

    # Find common date range (intersection)
    common_start = max(df.index.min() for df in parts)
    common_end = min(df.index.max() for df in parts)

    if common_start > common_end:
        raise ValueError(
            f"No overlapping date range across TSOs for {label}. "
            f"Ranges: {[(df.index.min(), df.index.max()) for df in parts]}"
        )

    # Check each part and log any trimming
    for i, df in enumerate(parts):
        start_trim = (common_start - df.index.min()).total_seconds() / 3600
        end_trim = (df.index.max() - common_end).total_seconds() / 3600

        if start_trim > 0 or end_trim > 0:
            logger.warning(
                f"Trimming TSO {i} for {label}: removing {start_trim:.0f}h from "
                f"start, {end_trim:.0f}h from end to align with common range "
                f"({common_start} to {common_end})."
            )

    # Trim all parts to common range, then inner-join
    trimmed = [df.loc[common_start:common_end] for df in parts]
    result = trimmed[0].copy()
    for df in trimmed[1:]:
        result = result.merge(df, left_index=True, right_index=True, how="inner")

    # Final safety check — no NaN should be introduced by inner join on aligned ranges
    nan_cols = result.columns[result.isna().any()].tolist()
    if nan_cols:
        logger.warning(
            f"NaN values found after merge for {label} in columns: {nan_cols}. "
            f"These are interior gaps within the common range."
        )

    return result

