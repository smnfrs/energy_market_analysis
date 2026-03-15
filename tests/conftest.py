"""Shared fixtures for energy market analysis tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def forecast_index():
    """168-hour UTC hourly index starting from a Monday."""
    return pd.date_range("2026-03-09", periods=168, freq="h", tz="UTC")


@pytest.fixture
def mock_forecast_csv(tmp_path, forecast_index):
    """Create a mock forecast CSV matching the pipeline output format."""

    def _create(target_name: str, mean_value: float = 1000.0, noise_std: float = 100.0):
        rng = np.random.default_rng(42)
        n = len(forecast_index)
        fitted = mean_value + rng.normal(0, noise_std, n)
        lower = fitted - 200
        upper = fitted + 200
        actual = mean_value + rng.normal(0, noise_std, n)

        df = pd.DataFrame(
            {
                f"{target_name}_actual": actual,
                f"{target_name}_fitted": fitted,
                f"{target_name}_lower": lower,
                f"{target_name}_upper": upper,
            },
            index=forecast_index,
        )
        df.index.name = "date"
        return df

    return _create


@pytest.fixture
def mock_forecasts_dir(tmp_path, mock_forecast_csv):
    """Create a complete mock output/DE/forecasts/ directory with all TSO targets."""
    base = tmp_path / "output" / "DE" / "forecasts"

    targets = {
        "wind_onshore_50hz": 2000,
        "wind_onshore_ampr": 1500,
        "wind_onshore_tenn": 1800,
        "wind_onshore_tran": 1200,
        "wind_onshore_lu": 50,
        "wind_offshore_50hz": 3000,
        "wind_offshore_tenn": 2500,
        "solar_50hz": 2000,
        "solar_ampr": 1500,
        "solar_tenn": 1800,
        "solar_tran": 1200,
        "solar_lu": 30,
        "load_50hz": 8000,
        "load_ampr": 7000,
        "load_tenn": 9000,
        "load_tran": 5000,
        "load_lu": 400,
        "gen_load_diff_delu": 2000,
    }

    for target_name, mean_val in targets.items():
        target_dir = base / target_name
        model_dir = target_dir / "LightGBM" / "forecast"
        model_dir.mkdir(parents=True)

        # Write forecast CSV
        df = mock_forecast_csv(target_name, mean_value=mean_val)
        df.to_csv(model_dir / "forecast.csv")

        # Write best_model.json (honest rolling CV selection)
        best_model = {
            target_name: {
                "method": "trained",
                "model_label": "LightGBM",
                "avg_rmse": 100.0,
            }
        }
        with open(target_dir / "best_model.json", "w") as f:
            json.dump(best_model, f)

    return base
