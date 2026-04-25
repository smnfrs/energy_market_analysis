"""Generate historical EMA forecasts for EP retraining.

Produces day-ahead generation/load forecasts by running trained EMA models
on historical weather data. Two modes:

- backtest: 2025+ using Open-Meteo historical forecast weather (realistic errors)
- hindcast: pre-2025 using Open-Meteo actual weather archive (unrealistically accurate)

The output is a national DE/LU aggregate parquet file with SMARD-compatible column
names, suitable for replacing SMARD's prognostizierte_* columns in EP's training data.

Usage:
    python generate_historical_forecasts.py --mode backtest
    python generate_historical_forecasts.py --mode hindcast --start 2023-01-01 --end 2024-12-31
    python generate_historical_forecasts.py --mode backtest --target wind_onshore_50hz  # single target
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_collection_modules.eu_locations import countries_metadata
from data_modules.data_classes import HistForecastDataset
from data_modules.data_loaders import (
    clean_and_impute,
    compute_gen_load_diff,
    impute_smard_nans,
)
from data_modules.utils import merge_tso_dataframes
from export_national_forecasts import COMPONENTS, CSV_COLUMNS
from forecasting_modules.base_models import instantiate_base_singletarget_forecaster
from forecasting_modules.tasks import BaseModelTasks, EnsembleModelTasks
from forecasting_modules.utils import convert_ensemble_string
from logger import get_logger

logger = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DB_PATH = Path("database/DE/")
FORECASTS_DIR = Path("output/DE/forecasts/")
OUTPUT_DIR = Path("output/DE/historical_forecasts/")
FORECAST_HORIZON = 168  # 7 days × 24 hours
N_HORIZONS = 100  # history length = N_HORIZONS × FORECAST_HORIZON

# Execution order (matches EMA production pipeline)
TARGET_GROUPS = [
    # Group 0: weather-only, no dependencies
    {
        "wind_offshore": ["wind_offshore_50hz", "wind_offshore_tenn"],
        "wind_onshore": [
            "wind_onshore_50hz", "wind_onshore_ampr", "wind_onshore_tenn",
            "wind_onshore_tran", "wind_onshore_lu",
        ],
        "solar": [
            "solar_50hz", "solar_ampr", "solar_tenn", "solar_tran", "solar_lu",
        ],
    },
    # Group 1: depends on wind/solar predictions
    {
        "load": [
            "load_50hz", "load_ampr", "load_tenn", "load_tran", "load_lu",
        ],
    },
    # Group 2: depends on all above
    {
        "gen_load_diff": ["gen_load_diff_delu"],
    },
]

# Maps target type → weather suffix for data loading
WEATHER_SUFFIX = {
    "wind_offshore": "offshore",
    "wind_onshore": "onshore",
    "solar": "solar",
    "load": "cities",
    "gen_load_diff": "cities",
}

# Maps target type → which exogenous prediction targets are needed
EXOG_TARGETS = {
    "load": ["wind_offshore", "wind_onshore", "solar"],
    "gen_load_diff": ["wind_offshore", "wind_onshore", "solar", "load"],
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def get_de_metadata() -> dict:
    """Get DE country metadata from EMA's eu_locations."""
    return [d for d in countries_metadata if d["code"] == "DE"][0]


def load_weather_per_tso(c_dict: dict) -> dict:
    """Load all weather parquet files, organized by suffix and TSO.

    Returns:
        {suffix: {tso_name: {"history": DataFrame, "hist_forecast": DataFrame}}}
    """
    weather = {}
    for suffix in ["offshore", "onshore", "solar", "cities"]:
        weather[suffix] = {}
        for tso_dict in c_dict["regions"]:
            tso_name = tso_dict["TSO"]
            if tso_name == "DE_ALL":
                continue
            dir_ = DB_PATH / "openmeteo" / suffix / tso_name
            if not dir_.is_dir():
                continue
            data = {}
            for source in ["history", "hist_forecast"]:
                path = dir_ / f"{source}_hourly.parquet"
                if path.exists():
                    data[source] = pd.read_parquet(path)
            if data:
                weather[suffix][tso_name] = data
    return weather


def combine_weather_for_suffix(
    weather_data: dict, suffix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine all TSO weather data for a suffix into single history/hist_forecast frames.

    Returns (df_history, df_hist_forecast) with all TSO columns merged.
    """
    history_parts = []
    hist_forecast_parts = []
    for _tso_name, sources in weather_data[suffix].items():
        if "history" in sources:
            history_parts.append(sources["history"])
        if "hist_forecast" in sources:
            hist_forecast_parts.append(sources["hist_forecast"])

    df_history = _merge_parts(history_parts, label=f"{suffix}/history")
    df_hist_forecast = _merge_parts(hist_forecast_parts, label=f"{suffix}/hist_forecast")
    return df_history, df_hist_forecast


def _merge_parts(parts: list[pd.DataFrame], label: str = "") -> pd.DataFrame:
    """Merge TSO DataFrames using common date range intersection."""
    return merge_tso_dataframes(parts, label=label)


def load_smard_targets() -> pd.DataFrame:
    """Load SMARD v2 per-TSO generation and load actuals."""
    path = DB_PATH / "smard_v2" / "history_hourly.parquet"
    df = pd.read_parquet(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = impute_smard_nans(df)
    df = df.join(compute_gen_load_diff(df))
    return df


# ─── Weather slicing for a cutoff ────────────────────────────────────────────

def get_weather_columns_for_target(
    df_weather: pd.DataFrame, c_dict: dict, target_label: str, target_type: str
) -> list[str]:
    """Get the weather column names relevant for a target (matching TSO locations)."""
    suffix = WEATHER_SUFFIX[target_type]
    locations = c_dict["locations"][suffix]

    # Find the TSO for this target
    tso_code = None
    for r in c_dict["regions"]:
        if target_label.endswith(r["suffix"]):
            tso_code = r["TSO"]
            break
    if tso_code is None:
        raise ValueError(f"No region found for target {target_label}")

    # Get location suffixes for this TSO
    if tso_code == "DE_ALL":
        om_suffixes = [loc["suffix"] for loc in locations]
    else:
        om_suffixes = [loc["suffix"] for loc in locations if loc["TSO"] == tso_code]

    # Filter weather columns
    feature_cols = [
        c for c in df_weather.columns if c.endswith(tuple(om_suffixes))
    ]
    return feature_cols


def slice_weather_for_cutoff(
    df_history: pd.DataFrame,
    df_hist_forecast: pd.DataFrame,
    feature_cols: list[str],
    cutoff: pd.Timestamp,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice weather data for a specific cutoff and mode.

    Args:
        df_history: actual weather archive
        df_hist_forecast: historical forecast weather
        feature_cols: columns to select
        cutoff: last timestamp of history (forecast starts at cutoff + 1h)
        mode: "backtest" (use hist_forecast for forecast) or "hindcast" (use history)

    Returns:
        (df_weather_hist, df_weather_forecast) with only feature_cols
    """
    forecast_start = cutoff + pd.Timedelta(hours=1)
    forecast_end = cutoff + pd.Timedelta(hours=FORECAST_HORIZON)

    # History: always use actual weather
    hist_end = cutoff
    hist_start = cutoff - pd.Timedelta(hours=N_HORIZONS * FORECAST_HORIZON)
    df_weather_hist = df_history.loc[hist_start:hist_end, feature_cols]

    # Forecast: depends on mode
    if mode == "backtest":
        df_weather_forecast = df_hist_forecast.loc[
            forecast_start:forecast_end, feature_cols
        ]
    else:  # hindcast
        df_weather_forecast = df_history.loc[
            forecast_start:forecast_end, feature_cols
        ]

    return df_weather_hist, df_weather_forecast


# ─── Model loading ────────────────────────────────────────────────────────────

def get_best_model_info(target: str) -> dict:
    """Load best_model.json for a target."""
    path = FORECASTS_DIR / target / "best_model.json"
    with open(path) as f:
        bm = json.load(f)
    return bm[target]


def get_model_dir(target: str, model_label: str) -> Path:
    """Get the trained model directory path."""
    if "ensemble" in model_label:
        dir_name = convert_ensemble_string(model_label)
    else:
        dir_name = model_label
    return FORECASTS_DIR / target / dir_name / "trained"


def load_model_config(trained_dir: Path) -> tuple[dict, dict]:
    """Load dataset.json and best_parameters.json from a trained model directory.

    Returns:
        (model_dataset_pars, optuna_pars)
    """
    with open(trained_dir / "dataset.json") as f:
        model_dataset_pars = json.load(f)
    with open(trained_dir / "best_parameters.json") as f:
        optuna_pars = json.load(f)

    # Point scalers to pre-fitted files
    model_dataset_pars["target_scaler"] = str(trained_dir / "target_scaler.pkl")
    model_dataset_pars["feature_scaler"] = str(trained_dir / "feature_scaler.pkl")
    model_dataset_pars["verbose"] = False
    model_dataset_pars["copy_input"] = False  # efficiency: we create fresh slices

    return model_dataset_pars, optuna_pars


# ─── Prediction ───────────────────────────────────────────────────────────────

def _load_forecaster(model_name: str, target: str, trained_dir: Path):
    """Instantiate a forecaster and load saved weights.

    We pass empty model_pars since the actual weights come from model.joblib.
    The instantiation just needs to create the right model class wrapper.
    """
    forecaster = instantiate_base_singletarget_forecaster(
        model_name=model_name,
        targets=[target],
        model_pars={},  # weights loaded from joblib, not from params
        verbose=False,
    )
    forecaster.load_model(str(trained_dir / "model.joblib"))
    return forecaster


class TargetPredictor:
    """Loads a trained model once and runs inference on multiple cutoff days."""

    def __init__(self, target: str, model_info: dict):
        self.target = target
        self.model_label = model_info["model_label"]
        self.is_ensemble = "ensemble" in self.model_label

        if self.is_ensemble:
            self._init_ensemble()
        else:
            self._init_base()

    def _init_base(self):
        """Load a single base model."""
        trained_dir = get_model_dir(self.target, self.model_label)
        self.model_dataset_pars, self.optuna_pars = load_model_config(trained_dir)
        self.lags_target = self.optuna_pars.get("lags_target")

        # Instantiate with empty params (model weights loaded from joblib)
        self.forecaster = _load_forecaster(
            self.model_label, self.target, trained_dir,
        )

    def _init_ensemble(self):
        """Load ensemble: meta-model + base models."""
        from forecasting_modules.utils import get_ensemble_name_and_model_names

        meta_model_name, base_model_names = get_ensemble_name_and_model_names(
            self.model_label
        )
        self.meta_model_name = meta_model_name
        self.base_model_names = base_model_names

        # Load meta-model config
        meta_dir = get_model_dir(self.target, self.model_label)
        self.meta_dataset_pars, self.meta_optuna_pars = load_model_config(meta_dir)
        self.use_pred_intervals = self.meta_optuna_pars.get(
            "use_base_models_pred_intervals", False
        )

        # Load meta-model
        self.meta_forecaster = _load_forecaster(
            meta_model_name, self.target, meta_dir,
        )

        # Load each base model
        self.base_predictors = {}
        for bm_name in base_model_names:
            bm_dir = FORECASTS_DIR / self.target / bm_name / "trained"
            bm_dataset_pars, bm_optuna_pars = load_model_config(bm_dir)

            bm_forecaster = _load_forecaster(bm_name, self.target, bm_dir)

            self.base_predictors[bm_name] = {
                "forecaster": bm_forecaster,
                "dataset_pars": bm_dataset_pars,
                "optuna_pars": bm_optuna_pars,
                "lags_target": bm_optuna_pars.get("lags_target"),
            }

    def predict_window(
        self, df_hist: pd.DataFrame, df_forecast: pd.DataFrame
    ) -> pd.Series:
        """Run inference for one cutoff. Returns fitted predictions as a Series.

        Args:
            df_hist: weather features + target column, history up to cutoff
            df_forecast: weather features only, forecast horizon

        Returns:
            pd.Series indexed by forecast timestamps, values in original scale (MW)
        """
        if self.is_ensemble:
            return self._predict_ensemble(df_hist, df_forecast)
        else:
            return self._predict_base(
                df_hist, df_forecast,
                self.model_dataset_pars, self.optuna_pars,
                self.forecaster, self.lags_target,
            )

    def _predict_base(
        self,
        df_hist: pd.DataFrame,
        df_forecast: pd.DataFrame,
        dataset_pars: dict,
        optuna_pars: dict,
        forecaster,
        lags_target,
    ) -> pd.Series:
        """Run base model inference."""
        config = optuna_pars | dataset_pars

        # Create dataset with pre-fitted scalers
        ds = HistForecastDataset(
            df_historic=df_hist, df_forecast=df_forecast, pars=dataset_pars,
        )
        ds.run_preprocess_pipeline(config)

        # Forecast
        result = forecaster.forecast_window(
            ds.exog_forecast, ds.target_hist, lags_target=lags_target,
        )

        # Inverse transform to original scale
        result_inv = ds.inverse_transform_targets(result)
        fitted_col = f"{self.target}_fitted"
        return result_inv[fitted_col]

    def _predict_ensemble(
        self, df_hist: pd.DataFrame, df_forecast: pd.DataFrame
    ) -> pd.Series:
        """Run ensemble inference: base models → stack → meta-model."""
        # 1. Get base model predictions
        base_forecasts = {}
        for bm_name, bm_info in self.base_predictors.items():
            pred = self._predict_base(
                df_hist, df_forecast,
                bm_info["dataset_pars"], bm_info["optuna_pars"],
                bm_info["forecaster"], bm_info["lags_target"],
            )
            base_forecasts[bm_name] = pred

        # 2. Create meta-model dataset (for target scaler + feature engineering)
        meta_config = self.meta_optuna_pars | self.meta_dataset_pars
        meta_ds = HistForecastDataset(
            df_historic=df_hist, df_forecast=df_forecast, pars=self.meta_dataset_pars,
        )
        meta_ds.run_preprocess_pipeline(meta_config)

        # 3. Build meta-features from base model predictions
        # Base predictions need to be in scaled space (matching how meta-model was trained)
        features_to_use = ["fitted"]
        if self.use_pred_intervals:
            features_to_use = ["fitted", "lower", "upper"]

        X_meta = pd.DataFrame(index=meta_ds.forecast_idx)
        for bm_name, bm_info in self.base_predictors.items():
            # Re-run base model to get scaled predictions (with _fitted, _lower, _upper)
            bm_ds_pars = bm_info["dataset_pars"]
            bm_config = bm_info["optuna_pars"] | bm_ds_pars
            bm_ds = HistForecastDataset(
                df_historic=df_hist, df_forecast=df_forecast, pars=bm_ds_pars,
            )
            bm_ds.run_preprocess_pipeline(bm_config)

            bm_result_scaled = bm_info["forecaster"].forecast_window(
                bm_ds.exog_forecast, bm_ds.target_hist,
                lags_target=bm_info["lags_target"],
            )

            for key in features_to_use:
                col = f"{self.target}_{key}"
                X_meta[f"base_{bm_name}_{col}"] = bm_result_scaled[col].values

        # 4. Add meta-model's own exogenous features if any
        if meta_ds.exog_forecast is not None and len(meta_ds.exog_forecast.columns) > 0:
            X_meta = X_meta.merge(
                meta_ds.exog_forecast, left_index=True, right_index=True,
            )

        # 5. Meta-model prediction
        meta_lags = self.meta_optuna_pars.get("lags_target")
        meta_result = self.meta_forecaster.forecast_window(
            X_meta, meta_ds.target_hist, lags_target=meta_lags,
        )

        # 6. Inverse transform
        result_inv = meta_ds.inverse_transform_targets(meta_result)
        fitted_col = f"{self.target}_fitted"
        return result_inv[fitted_col]


# ─── Input construction ──────────────────────────────────────────────────────

def construct_input_for_target(
    target_label: str,
    target_type: str,
    cutoff: pd.Timestamp,
    mode: str,
    weather_history: pd.DataFrame,
    weather_hist_forecast: pd.DataFrame,
    feature_cols: list[str],
    smard_targets: pd.DataFrame,
    upstream_predictions: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build df_hist and df_forecast for a target at a specific cutoff.

    Args:
        target_label: e.g., "wind_onshore_50hz"
        target_type: e.g., "wind_onshore"
        cutoff: last historical timestamp
        mode: "backtest" or "hindcast"
        weather_history: full actual weather history
        weather_hist_forecast: full historical forecast weather
        feature_cols: weather columns for this target
        smard_targets: SMARD generation/load actuals
        upstream_predictions: dict of {exog_target: pd.Series} for dependent targets

    Returns:
        (df_hist, df_forecast) matching extract_from_database() format
    """
    # Slice weather
    df_weather_hist, df_weather_forecast = slice_weather_for_cutoff(
        weather_history, weather_hist_forecast, feature_cols, cutoff, mode,
    )

    # Validate forecast has enough data
    if len(df_weather_forecast) < FORECAST_HORIZON:
        raise ValueError(
            f"Not enough forecast weather data at cutoff {cutoff}: "
            f"got {len(df_weather_forecast)}, need {FORECAST_HORIZON}"
        )
    df_weather_forecast = df_weather_forecast.iloc[:FORECAST_HORIZON]

    # Build df_hist: weather + target
    target_hist = smard_targets.loc[:cutoff, target_label]
    # Align to weather history range
    common_idx = df_weather_hist.index.intersection(target_hist.index)
    df_hist = df_weather_hist.loc[common_idx].copy()
    df_hist[target_label] = target_hist.loc[common_idx]

    # Limit history size (same as extract_from_database)
    max_hist = N_HORIZONS * FORECAST_HORIZON
    if len(df_hist) > max_hist:
        df_hist = df_hist.tail(max_hist)

    # Drop any rows with NaN in target
    df_hist = df_hist.dropna(subset=[target_label])

    # Build df_forecast: weather only (same columns as df_hist minus target)
    df_forecast = df_weather_forecast.copy()

    # Add exogenous features for dependent targets
    if target_type in EXOG_TARGETS and upstream_predictions is not None:
        exog_types = EXOG_TARGETS[target_type]
        _add_exogenous_features(
            df_hist, df_forecast, smard_targets, upstream_predictions,
            exog_types, cutoff,
        )

    # Ensure contiguity: forecast must start exactly 1h after history ends
    expected_forecast_start = df_hist.index[-1] + pd.Timedelta(hours=1)
    if df_forecast.index[0] != expected_forecast_start:
        # Trim history to align
        valid_end = df_forecast.index[0] - pd.Timedelta(hours=1)
        df_hist = df_hist.loc[:valid_end]
        if len(df_hist) == 0:
            raise ValueError(f"No valid history after alignment at cutoff {cutoff}")

    # Ensure forecast is exactly FORECAST_HORIZON rows
    if len(df_forecast) != FORECAST_HORIZON:
        df_forecast = df_forecast.iloc[:FORECAST_HORIZON]

    # Ensure history is divisible by forecast horizon
    remainder = len(df_hist) % FORECAST_HORIZON
    if remainder != 0:
        df_hist = df_hist.iloc[remainder:]

    # Validate: no NaN in forecast
    if df_forecast.isna().any().any():
        nan_cols = df_forecast.columns[df_forecast.isna().any()].tolist()
        logger.warning(f"NaN in forecast data for {target_label}: {nan_cols}")
        df_forecast = df_forecast.ffill().bfill()

    # Validate: no NaN in history weather features
    weather_cols_in_hist = [c for c in df_hist.columns if c != target_label]
    if df_hist[weather_cols_in_hist].isna().any().any():
        df_hist[weather_cols_in_hist] = (
            df_hist[weather_cols_in_hist].ffill().bfill()
        )

    return df_hist, df_forecast


def _add_exogenous_features(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    smard_targets: pd.DataFrame,
    upstream_predictions: dict,
    exog_types: list[str],
    cutoff: pd.Timestamp,
) -> None:
    """Add exogenous generation/load features to df_hist and df_forecast (in-place).

    Replicates the exact column ordering from extract_from_database():
    iterates over exog_types, then DE regions, adding columns in that order.

    For df_hist: uses SMARD actuals (same as production).
    For df_forecast: uses upstream model predictions.
    """
    c_dict = get_de_metadata()
    regions = [r for r in c_dict["regions"] if r["TSO"] != "DE_ALL"]
    forecast_start = cutoff + pd.Timedelta(hours=1)
    forecast_end = cutoff + pd.Timedelta(hours=FORECAST_HORIZON)

    for exog_type in exog_types:
        for tso_dict in regions:
            exog_target = exog_type + tso_dict["suffix"]

            # Skip if target doesn't exist in SMARD data
            if exog_target not in smard_targets.columns:
                continue
            # Skip if no trained model exists (same as extract_from_database)
            best_model_path = FORECASTS_DIR / exog_target / "best_model.json"
            if not best_model_path.exists():
                continue

            # History: SMARD actuals
            hist_vals = smard_targets.loc[df_hist.index, exog_target]
            df_hist[exog_target] = hist_vals.values

            # Forecast: upstream predictions
            if exog_target in upstream_predictions:
                pred = upstream_predictions[exog_target]
                forecast_vals = pred.reindex(df_forecast.index)
                df_forecast[exog_target] = forecast_vals.values
            else:
                # No prediction available — column will be NaN
                df_forecast[exog_target] = np.nan

    # Drop columns that are all-NaN in df_forecast (same as extract_from_database)
    nan_cols = [c for c in df_forecast.columns if df_forecast[c].isna().all()]
    if nan_cols:
        logger.warning(
            f"Dropping {len(nan_cols)} exog columns with no predictions: {nan_cols}"
        )
        df_forecast.drop(columns=nan_cols, inplace=True)
        df_hist.drop(
            columns=[c for c in nan_cols if c in df_hist.columns], inplace=True,
        )


# ─── Orchestration ────────────────────────────────────────────────────────────

def determine_cutoffs(mode: str, start: str | None, end: str | None) -> list[pd.Timestamp]:
    """Determine the list of cutoff timestamps for backtesting.

    Each cutoff is at 07:00 UTC (forecast starts at 08:00 UTC, but the last
    historical hour is 07:00). This matches EMA's production schedule where
    the forecast is generated at ~08:00 UTC using weather data that starts
    at the current hour.

    For simplicity and to maximize day-ahead coverage, we use cutoff = 23:00 UTC
    on day D-1, so the forecast covers D 00:00 to D+6 23:00 and the full day D
    is the "day-ahead" slice.
    """
    if mode == "backtest":
        default_start = "2025-01-08"  # first date with enough weather history
        default_end = "2026-03-06"  # last date with complete forecast weather
    else:  # hindcast
        default_start = "2021-06-01"  # need ~6 months of history for the model
        default_end = "2024-12-31"

    start_date = pd.Timestamp(start or default_start, tz="UTC")
    end_date = pd.Timestamp(end or default_end, tz="UTC")

    # Cutoff at 23:00 UTC on the day BEFORE the target day
    # So predictions for day D start at D 00:00 UTC
    cutoffs = []
    current = start_date - pd.Timedelta(hours=1)  # 23:00 UTC on day before start
    end_cutoff = end_date - pd.Timedelta(hours=1)

    while current <= end_cutoff:
        cutoffs.append(current)
        current += pd.Timedelta(days=1)

    return cutoffs


def _checkpoint_dir(mode: str) -> Path:
    """Return the checkpoint directory for a given mode."""
    return OUTPUT_DIR / f".checkpoints_{mode}"


def _save_checkpoint(
    target_label: str, day_ahead: pd.Series, full: pd.DataFrame, mode: str
) -> None:
    """Save a single target's results to checkpoint files."""
    ckpt = _checkpoint_dir(mode)
    ckpt.mkdir(parents=True, exist_ok=True)
    day_ahead.to_frame(name=target_label).to_parquet(
        ckpt / f"{target_label}_dayahead.parquet"
    )
    full.to_parquet(ckpt / f"{target_label}_full.parquet")


def _load_checkpoints(mode: str) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """Load all existing checkpoint files. Returns (all_predictions, all_full_predictions)."""
    ckpt = _checkpoint_dir(mode)
    predictions = {}
    full_predictions = {}
    if not ckpt.exists():
        return predictions, full_predictions
    for path in sorted(ckpt.glob("*_dayahead.parquet")):
        target = path.name.replace("_dayahead.parquet", "")
        full_path = ckpt / f"{target}_full.parquet"
        if not full_path.exists():
            continue
        df_da = pd.read_parquet(path)
        predictions[target] = df_da[target]
        full_predictions[target] = pd.read_parquet(full_path)
        logger.info(f"  Loaded checkpoint: {target}")
    return predictions, full_predictions


def _run_target(
    target_label: str,
    target_type: str,
    mode: str,
    cutoffs: list[pd.Timestamp],
    weather_history: pd.DataFrame,
    weather_hist_forecast: pd.DataFrame,
    smard_targets: pd.DataFrame,
    upstream_predictions: dict | None,
) -> tuple[str, pd.Series | None, pd.DataFrame | None]:
    """Run all cutoffs for a single target. Designed for use in worker processes.

    Returns (target_label, day_ahead_series, full_predictions) or (target_label, None, None)
    on failure.
    """
    logger.info(f"\n--- {target_label} ---")
    c_dict = get_de_metadata()

    model_info = get_best_model_info(target_label)
    logger.info(f"  [{target_label}] Model: {model_info['model_label']}")

    feature_cols = get_weather_columns_for_target(
        weather_history, c_dict, target_label, target_type,
    )
    logger.info(f"  [{target_label}] Weather features: {len(feature_cols)} columns")

    predictor = TargetPredictor(target_label, model_info)

    predictions = []
    t_start = time.time()
    n_errors = 0

    for i, cutoff in enumerate(cutoffs):
        try:
            df_hist, df_forecast = construct_input_for_target(
                target_label, target_type, cutoff, mode,
                weather_history, weather_hist_forecast,
                feature_cols, smard_targets, upstream_predictions,
            )
            pred = predictor.predict_window(df_hist, df_forecast)
            predictions.append(pred)
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                logger.warning(f"  [{target_label}] Error at cutoff {cutoff}: {e}")
            if n_errors == 3:
                logger.warning(f"  [{target_label}] (suppressing further errors)")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(cutoffs) - i - 1) / rate
            logger.info(
                f"  [{target_label}] {i+1}/{len(cutoffs)} cutoffs done "
                f"({rate:.1f}/s, ~{remaining:.0f}s remaining)"
            )

    elapsed = time.time() - t_start
    logger.info(
        f"  [{target_label}] Completed {len(predictions)}/{len(cutoffs)} cutoffs "
        f"in {elapsed:.1f}s ({n_errors} errors)"
    )

    if not predictions:
        return target_label, None, None

    rows = []
    for pred, cutoff_ts in zip(predictions, cutoffs[:len(predictions)]):
        df_pred = pred.to_frame(name=target_label)
        df_pred["cutoff"] = cutoff_ts
        df_pred["hours_ahead"] = (
            (df_pred.index - cutoff_ts).total_seconds() / 3600
        ).astype(int)
        rows.append(df_pred)
    full = pd.concat(rows)

    day_ahead = full[(full["hours_ahead"] >= 1) & (full["hours_ahead"] <= 24)]
    day_ahead_series = day_ahead[target_label]
    day_ahead_series = day_ahead_series[
        ~day_ahead_series.index.duplicated(keep="last")
    ].sort_index()

    logger.info(
        f"  [{target_label}] Stored {len(day_ahead_series)} day-ahead hours, "
        f"{len(full)} total predictions ({full.index[0]} to {full.index[-1]})"
    )

    del predictor
    gc.collect()

    return target_label, day_ahead_series, full


def run_backtest(
    targets: list[str] | None = None,
    mode: str = "backtest",
    start: str | None = None,
    end: str | None = None,
    dry_run: bool = False,
    max_workers: int = 1,
    clean: bool = False,
    stop_after_group: int | None = None,
) -> pd.DataFrame:
    """Run the full backtest/hindcast pipeline.

    Args:
        targets: specific targets to run (None = all)
        mode: "backtest" or "hindcast"
        start: start date (YYYY-MM-DD)
        end: end date (YYYY-MM-DD)
        dry_run: if True, only process first 3 cutoffs per target
        max_workers: number of parallel worker processes per target group
        clean: if True, delete existing checkpoints before starting
        stop_after_group: if set, stop after completing this group index (0-based)

    Returns:
        National aggregate DataFrame with SMARD-compatible columns
    """
    logger.info(f"Starting {mode} forecast generation (max_workers={max_workers})")

    # Handle clean start
    if clean:
        ckpt = _checkpoint_dir(mode)
        if ckpt.exists():
            shutil.rmtree(ckpt)
            logger.info(f"Deleted existing checkpoints: {ckpt}")

    # Load existing checkpoints
    all_predictions, all_full_predictions = _load_checkpoints(mode)
    if all_predictions:
        logger.info(f"Resuming with {len(all_predictions)} checkpointed targets")

    # Load data
    c_dict = get_de_metadata()
    logger.info("Loading weather data...")
    weather_data = load_weather_per_tso(c_dict)
    logger.info("Loading SMARD target data...")
    smard_targets = load_smard_targets()

    # Pre-combine weather per suffix (avoids repeated merging)
    weather_combined = {}
    for suffix in ["offshore", "onshore", "solar", "cities"]:
        history, hist_forecast = combine_weather_for_suffix(weather_data, suffix)
        weather_combined[suffix] = {
            "history": history,
            "hist_forecast": hist_forecast,
        }
        logger.info(
            f"  {suffix}: history {history.shape}, "
            f"hist_forecast {hist_forecast.shape}"
        )

    # Determine cutoff days
    cutoffs = determine_cutoffs(mode, start, end)
    if dry_run:
        cutoffs = cutoffs[:3]
    logger.info(f"Will process {len(cutoffs)} cutoff days ({cutoffs[0]} to {cutoffs[-1]})")

    # Process target groups in dependency order
    for group_idx, group in enumerate(TARGET_GROUPS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing target group {group_idx}")
        logger.info(f"{'='*60}")

        # Collect all targets for this group that still need processing
        group_targets = []  # (target_label, target_type, suffix)
        for target_type, target_list in group.items():
            if targets is not None:
                target_list = [t for t in target_list if t in targets]
            for target_label in target_list:
                if target_label in all_predictions:
                    logger.info(f"  Skipping {target_label} (checkpointed)")
                    continue
                group_targets.append((target_label, target_type))

        if not group_targets:
            logger.info("  All targets in this group already checkpointed")
            continue

        # Upstream predictions for dependent groups
        upstream = dict(all_predictions) if group_idx > 0 else None

        use_parallel = max_workers > 1 and len(group_targets) > 1

        if use_parallel:
            n_workers = min(max_workers, len(group_targets))
            logger.info(f"  Running {len(group_targets)} targets with {n_workers} workers")
            futures = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for target_label, target_type in group_targets:
                    suffix = WEATHER_SUFFIX[target_type]
                    fut = executor.submit(
                        _run_target,
                        target_label, target_type, mode, cutoffs,
                        weather_combined[suffix]["history"],
                        weather_combined[suffix]["hist_forecast"],
                        smard_targets, upstream,
                    )
                    futures[fut] = target_label

                for fut in as_completed(futures):
                    target_label = futures[fut]
                    try:
                        _, day_ahead_series, full = fut.result()
                    except Exception as e:
                        logger.error(f"  Worker failed for {target_label}: {e}")
                        continue
                    if day_ahead_series is not None:
                        all_predictions[target_label] = day_ahead_series
                        all_full_predictions[target_label] = full
                        _save_checkpoint(target_label, day_ahead_series, full, mode)
                        logger.info(f"  Checkpointed {target_label}")
        else:
            for target_label, target_type in group_targets:
                suffix = WEATHER_SUFFIX[target_type]
                _, day_ahead_series, full = _run_target(
                    target_label, target_type, mode, cutoffs,
                    weather_combined[suffix]["history"],
                    weather_combined[suffix]["hist_forecast"],
                    smard_targets, upstream,
                )
                if day_ahead_series is not None:
                    all_predictions[target_label] = day_ahead_series
                    all_full_predictions[target_label] = full
                    _save_checkpoint(target_label, day_ahead_series, full, mode)
                    logger.info(f"  Checkpointed {target_label}")

        if stop_after_group is not None and group_idx >= stop_after_group:
            logger.info(
                f"\nStopping after group {group_idx} (--stop-after-group {stop_after_group}). "
                f"Checkpoints saved — restart without --clean to continue."
            )
            return pd.DataFrame()

    # Aggregate to national level
    logger.info(f"\n{'='*60}")
    logger.info("Aggregating to national DE/LU level")
    logger.info(f"{'='*60}")
    result = aggregate_national(all_predictions)

    # Save output
    save_output(result, mode, start, end)

    # Save full per-TSO predictions (with horizon metadata) for analysis
    if all_full_predictions:
        save_full_predictions(all_full_predictions, mode)

    # Clean up checkpoints after successful completion
    ckpt = _checkpoint_dir(mode)
    if ckpt.exists():
        shutil.rmtree(ckpt)
        logger.info(f"Cleaned up checkpoints: {ckpt}")

    return result


def aggregate_national(predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """Aggregate per-TSO predictions to national DE/LU level.

    Replicates the aggregation from export_national_forecasts.py.
    Generation columns are clipped to zero (negative MW is non-physical).
    """
    aggregates = {}

    # Base components: sum per-TSO predictions
    for component_name, tso_list in COMPONENTS.items():
        parts = []
        for tso_target in tso_list:
            if tso_target in predictions:
                parts.append(predictions[tso_target])
            else:
                logger.warning(f"Missing prediction for {tso_target}")
        if parts:
            aggregates[component_name] = pd.concat(parts, axis=1).sum(axis=1)
            logger.info(f"  {component_name}: {len(parts)} TSOs summed")
        else:
            logger.warning(f"  {component_name}: no predictions available")

    # gen_load_diff_delu
    if "gen_load_diff_delu" in predictions:
        gld = predictions["gen_load_diff_delu"]
    else:
        gld = None

    # Derived quantities
    if "verbrauch" in aggregates and gld is not None:
        aggregates["gesamt"] = aggregates["verbrauch"] + gld
    if all(k in aggregates for k in ["onshore", "offshore", "photovoltaik"]):
        aggregates["wind_und_photovoltaik"] = (
            aggregates["onshore"] + aggregates["offshore"]
            + aggregates["photovoltaik"]
        )
    if "gesamt" in aggregates and "wind_und_photovoltaik" in aggregates:
        aggregates["sonstige"] = (
            aggregates["gesamt"] - aggregates["wind_und_photovoltaik"]
        )
    if "verbrauch" in aggregates and "wind_und_photovoltaik" in aggregates:
        aggregates["residuallast"] = (
            aggregates["verbrauch"] - aggregates["wind_und_photovoltaik"]
        )

    # Clip generation columns to zero (negative MW is non-physical).
    # Do NOT clip: verbrauch (load), residuallast (legitimately negative).
    clip_keys = [
        "onshore", "offshore", "photovoltaik",
        "wind_und_photovoltaik", "gesamt", "sonstige",
    ]
    for key in clip_keys:
        if key in aggregates:
            n_neg = (aggregates[key] < 0).sum()
            if n_neg > 0:
                logger.info(f"  Clipping {n_neg} negative values in {key}")
            aggregates[key] = aggregates[key].clip(lower=0)

    # Build DataFrame with SMARD-compatible column names
    df = pd.DataFrame()
    for key, csv_col in CSV_COLUMNS.items():
        if key in aggregates:
            df[csv_col] = aggregates[key]

    # Also keep the breakdown columns
    for key, series in aggregates.items():
        if key not in CSV_COLUMNS:
            df[key] = series

    df.index.name = "date_utc"
    return df


def _derive_filename(mode: str, df: pd.DataFrame) -> str:
    """Derive output filename from mode and actual date range in the data."""
    date_min = df.index.min()
    date_max = df.index.max()
    start_str = date_min.strftime("%Y_%m")
    end_str = date_max.strftime("%Y_%m")
    return f"{mode}_{start_str}_to_{end_str}"


def save_output(
    df: pd.DataFrame, mode: str, start: str | None, end: str | None
) -> None:
    """Save output to parquet and CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Add metadata columns
    df["source"] = mode

    fname = _derive_filename(mode, df)

    # Parquet
    pq_path = OUTPUT_DIR / f"{fname}.parquet"
    df.to_parquet(pq_path)
    logger.info(f"Saved {pq_path} ({len(df)} rows, {len(df.columns)} columns)")

    # CSV for inspection
    csv_path = OUTPUT_DIR / f"{fname}.csv"
    df.to_csv(csv_path)
    logger.info(f"Saved {csv_path}")


def save_full_predictions(
    all_full: dict[str, pd.DataFrame], mode: str,
) -> None:
    """Save full per-TSO predictions with cutoff/hours_ahead metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for target, df in all_full.items():
        parts.append(df)
    combined = pd.concat(parts, axis=1)
    # Remove duplicate columns (cutoff, hours_ahead appear per-target)
    # Keep the first occurrence
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Derive filename from date range
    date_min = combined.index.min()
    date_max = combined.index.max()
    start_str = date_min.strftime("%Y_%m")
    end_str = date_max.strftime("%Y_%m")
    fname = f"per_tso_{mode}_{start_str}_to_{end_str}.parquet"

    pq_path = OUTPUT_DIR / fname
    combined.to_parquet(pq_path)
    logger.info(f"Saved per-TSO predictions: {pq_path} ({len(combined)} rows)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate historical EMA forecasts for EP retraining",
    )
    parser.add_argument(
        "--mode", required=True, choices=["backtest", "hindcast"],
        help="backtest: 2025+ forecast weather; hindcast: pre-2025 actual weather",
    )
    parser.add_argument(
        "--start", default=None, help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None, help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--target", default=None,
        help="Single target to process (e.g., wind_onshore_50hz)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only first 3 cutoffs per target (for testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers per target group (default: 1 = sequential)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing checkpoints and start fresh",
    )
    parser.add_argument(
        "--stop-after-group", type=int, default=None,
        help="Stop after completing this group (0=generation, 1=load, 2=gen_load_diff)",
    )
    args = parser.parse_args()

    targets = [args.target] if args.target else None

    run_backtest(
        targets=targets,
        mode=args.mode,
        start=args.start,
        end=args.end,
        dry_run=args.dry_run,
        max_workers=args.workers,
        clean=args.clean,
        stop_after_group=args.stop_after_group,
    )


if __name__ == "__main__":
    main()
