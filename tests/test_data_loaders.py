"""Tests for data_modules/data_loaders.py — imputation and derived columns."""

import numpy as np
import pandas as pd
import pytest

from data_modules.data_loaders import (
    compute_gen_load_diff,
    compute_residual_load,
    impute_smard_nans,
)


class TestImputeSmardNans:
    def test_fills_nans_with_mean(self):
        df = pd.DataFrame({
            "solar_50hz": [100.0, 200.0, np.nan, 400.0],
            "load_50hz": [1000.0, 2000.0, 3000.0, 4000.0],
        })
        result = impute_smard_nans(df)
        assert not result.isna().any().any()
        # NaN in solar_50hz filled with mean of [100, 200, 400] = 233.33
        assert abs(result.loc[2, "solar_50hz"] - 233.33) < 1.0

    def test_all_nan_column_unchanged(self):
        df = pd.DataFrame({
            "solar_50hz": [np.nan, np.nan, np.nan],
            "load_50hz": [1000.0, 2000.0, 3000.0],
        })
        result = impute_smard_nans(df)
        assert result["solar_50hz"].isna().all()

    def test_no_nans_unchanged(self):
        df = pd.DataFrame({
            "solar_50hz": [100.0, 200.0, 300.0],
        })
        result = impute_smard_nans(df)
        pd.testing.assert_frame_equal(result, df)


class TestComputeGenLoadDiff:
    def test_basic_computation(self):
        df = pd.DataFrame({
            "solar_50hz": [100.0, 200.0],
            "wind_onshore_50hz": [300.0, 400.0],
            "load_50hz": [500.0, 600.0],
            "load_tenn": [200.0, 300.0],
        }, index=pd.date_range("2026-01-01", periods=2, freq="h"))

        result = compute_gen_load_diff(df)
        assert result.name == "gen_load_diff_delu"
        # gen = solar + wind_onshore = [400, 600]
        # load = load_50hz + load_tenn = [700, 900]
        # diff = [-300, -300]
        expected = pd.Series([-300.0, -300.0], index=df.index, name="gen_load_diff_delu")
        pd.testing.assert_series_equal(result, expected)

    def test_handles_nans_with_min_count(self):
        df = pd.DataFrame({
            "solar_50hz": [100.0, np.nan],
            "load_50hz": [500.0, 600.0],
        }, index=pd.date_range("2026-01-01", periods=2, freq="h"))

        result = compute_gen_load_diff(df)
        # Second row: gen=NaN (min_count=1 returns NaN for all-NaN), load=600
        assert pd.isna(result.iloc[1])


class TestComputeResidualLoad:
    def test_basic_computation(self):
        df = pd.DataFrame({
            "load_50hz": [5000.0],
            "wind_offshore_50hz": [1000.0],
            "wind_onshore_50hz": [1500.0],
            "solar_50hz": [500.0],
        })
        result = compute_residual_load(df, "_50hz")
        assert result.name == "residual_load_50hz"
        assert result.iloc[0] == 2000.0  # 5000 - 1000 - 1500 - 500

    def test_missing_offshore(self):
        """TSOs without offshore wind should still compute residual load."""
        df = pd.DataFrame({
            "load_ampr": [5000.0],
            "wind_onshore_ampr": [1500.0],
            "solar_ampr": [500.0],
        })
        result = compute_residual_load(df, "_ampr")
        assert result.iloc[0] == 3000.0  # 5000 - 1500 - 500 (no offshore)
