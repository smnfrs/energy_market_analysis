"""Tests for export_national_forecasts.py — DE/LU national aggregation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from export_national_forecasts import (
    COMPONENTS,
    CSV_COLUMNS,
    JSON_FILES,
    aggregate_delu,
    export_downloadable,
    export_json,
    load_best_forecast,
)


class TestLoadBestForecast:
    def test_loads_lightgbm_forecast(self, mock_forecasts_dir):
        result = load_best_forecast(mock_forecasts_dir / "wind_onshore_50hz")
        assert "forecast" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["model_label"] == "LightGBM"
        assert len(result["forecast"]) == 168

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_best_forecast(tmp_path / "nonexistent")


class TestAggregateDelu:
    def test_all_components_present(self, mock_forecasts_dir):
        data = aggregate_delu(mock_forecasts_dir)
        for key in ["onshore", "offshore", "photovoltaik", "verbrauch",
                     "gesamt", "wind_und_photovoltaik", "sonstige", "residuallast"]:
            assert key in data, f"Missing component: {key}"

    def test_lu_included_in_sums(self, mock_forecasts_dir):
        """Verify LU targets contribute to the national aggregates."""
        # Load LU-only forecasts
        lu_onshore = load_best_forecast(mock_forecasts_dir / "wind_onshore_lu")
        lu_solar = load_best_forecast(mock_forecasts_dir / "solar_lu")
        lu_load = load_best_forecast(mock_forecasts_dir / "load_lu")

        data = aggregate_delu(mock_forecasts_dir)

        # If we removed LU, the sum would be smaller
        de_only_onshore = sum(
            load_best_forecast(mock_forecasts_dir / d)["forecast"]
            for d in ["wind_onshore_50hz", "wind_onshore_ampr",
                       "wind_onshore_tenn", "wind_onshore_tran"]
        )
        assert data["onshore"]["forecast"].sum() > de_only_onshore.sum()

    def test_sonstige_algebraic_identity(self, mock_forecasts_dir):
        """sonstige = gesamt - wind_und_photovoltaik."""
        data = aggregate_delu(mock_forecasts_dir)
        diff = (
            data["gesamt"]["forecast"]
            - data["wind_und_photovoltaik"]["forecast"]
            - data["sonstige"]["forecast"]
        )
        assert diff.abs().max() < 0.01

    def test_residuallast_algebraic_identity(self, mock_forecasts_dir):
        """residuallast = verbrauch - wind_und_photovoltaik."""
        data = aggregate_delu(mock_forecasts_dir)
        diff = (
            data["verbrauch"]["forecast"]
            - data["wind_und_photovoltaik"]["forecast"]
            - data["residuallast"]["forecast"]
        )
        assert diff.abs().max() < 0.01

    def test_gesamt_equals_load_plus_gld(self, mock_forecasts_dir):
        """gesamt = verbrauch + gen_load_diff_delu."""
        data = aggregate_delu(mock_forecasts_dir)
        gld = load_best_forecast(mock_forecasts_dir / "gen_load_diff_delu")
        diff = (
            data["gesamt"]["forecast"]
            - data["verbrauch"]["forecast"]
            - gld["forecast"]
        )
        assert diff.abs().max() < 0.01

    def test_forecast_length(self, mock_forecasts_dir):
        data = aggregate_delu(mock_forecasts_dir)
        for key, val in data.items():
            assert len(val["forecast"]) == 168, f"{key} has wrong length"


class TestExportJson:
    def test_creates_all_json_files(self, mock_forecasts_dir, tmp_path):
        data = aggregate_delu(mock_forecasts_dir)
        output_dir = tmp_path / "api" / "forecasts"
        export_json(data, output_dir)

        for fname in JSON_FILES.values():
            fpath = output_dir / fname
            assert fpath.exists(), f"Missing JSON: {fname}"

            with open(fpath) as f:
                content = json.load(f)
            assert "metadata" in content
            assert "data" in content
            assert len(content["data"]) == 168
            assert content["metadata"]["units"] == "MW"
            assert content["metadata"]["tso_region"] == "DE_LU"

    def test_json_data_has_required_fields(self, mock_forecasts_dir, tmp_path):
        data = aggregate_delu(mock_forecasts_dir)
        output_dir = tmp_path / "api" / "forecasts"
        export_json(data, output_dir)

        with open(output_dir / "national_wind_onshore.json") as f:
            content = json.load(f)

        record = content["data"][0]
        assert "index" in record
        assert "datetime" in record
        assert "forecast" in record


class TestExportDownloadable:
    def test_creates_csv(self, mock_forecasts_dir, tmp_path):
        data = aggregate_delu(mock_forecasts_dir)
        output_dir = tmp_path / "downloads"
        export_downloadable(data, output_dir)

        fpath = output_dir / "national_forecasts.csv"
        assert fpath.exists()

        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        assert len(df) == 168
        assert len(df.columns) == 8

    def test_csv_column_names_match_smard(self, mock_forecasts_dir, tmp_path):
        data = aggregate_delu(mock_forecasts_dir)
        output_dir = tmp_path / "downloads"
        export_downloadable(data, output_dir)

        df = pd.read_csv(output_dir / "national_forecasts.csv", index_col=0)
        for col in CSV_COLUMNS.values():
            assert col in df.columns, f"Missing CSV column: {col}"
