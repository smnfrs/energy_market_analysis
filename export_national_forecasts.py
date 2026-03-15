"""Export national DE/LU bidding zone forecast aggregates.

Aggregates per-TSO forecasts (including Creos/Luxembourg) to national level.
Outputs JSON files for the dashboard API and a combined CSV for download.

Usage:
    python export_national_forecasts.py [--output-dir deploy/data/DE]
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting_modules.utils import convert_ensemble_string
from logger import get_logger

logger = get_logger(__name__)


# Mapping from national aggregate name to per-TSO forecast directories
COMPONENTS = {
    "onshore": [
        "wind_onshore_50hz", "wind_onshore_ampr",
        "wind_onshore_tenn", "wind_onshore_tran",
        "wind_onshore_lu",
    ],
    "offshore": [
        "wind_offshore_50hz", "wind_offshore_tenn",
    ],
    "photovoltaik": [
        "solar_50hz", "solar_ampr", "solar_tenn", "solar_tran",
        "solar_lu",
    ],
    "verbrauch": [
        "load_50hz", "load_ampr", "load_tenn", "load_tran",
        "load_lu",
    ],
}

# SMARD-style column names for the CSV export
CSV_COLUMNS = {
    "onshore": "prognostizierte_erzeugung_onshore",
    "offshore": "prognostizierte_erzeugung_offshore",
    "photovoltaik": "prognostizierte_erzeugung_photovoltaik",
    "verbrauch": "prognostizierter_verbrauch_gesamt",
    "gesamt": "prognostizierte_erzeugung_gesamt",
    "wind_und_photovoltaik": "prognostizierte_erzeugung_wind_und_photovoltaik",
    "sonstige": "prognostizierte_erzeugung_sonstige",
    "residuallast": "prognostizierter_verbrauch_residuallast",
}

# JSON filenames
JSON_FILES = {
    "onshore": "national_wind_onshore.json",
    "offshore": "national_wind_offshore.json",
    "photovoltaik": "national_solar.json",
    "verbrauch": "national_load.json",
    "gesamt": "national_generation_total.json",
    "wind_und_photovoltaik": "national_wind_solar_combined.json",
    "sonstige": "national_sonstige.json",
    "residuallast": "national_residual_load.json",
}

# Human-readable target names for JSON metadata
TARGET_NAMES = {
    "onshore": "wind_onshore",
    "offshore": "wind_offshore",
    "photovoltaik": "solar",
    "verbrauch": "load",
    "gesamt": "generation",
    "wind_und_photovoltaik": "wind_solar_combined",
    "sonstige": "sonstige",
    "residuallast": "residual_load",
}


def load_best_forecast(target_dir: Path) -> dict:
    """Load the best model's forecast CSV for a target directory.

    Returns dict with keys: forecast (Series), ci_lower (Series), ci_upper (Series),
    model_label (str), metadata from the forecast.
    """
    # Determine best model — prefer best_model.json (honest rolling CV) over
    # best_model_forecast.json (inference on data seen during training → leaked metrics)
    for json_name in ["best_model.json", "best_model_forecast.json"]:
        json_path = target_dir / json_name
        if json_path.exists():
            with open(json_path) as f:
                best_models = json.load(f)
            break
    else:
        raise FileNotFoundError(f"No best_model*.json found in {target_dir}")

    # best_models has one key (the target name) with model_label
    target_name = list(best_models.keys())[0]
    model_label = best_models[target_name]["model_label"]

    # Convert ensemble string to directory name
    if "ensemble" in model_label:
        model_dir = convert_ensemble_string(model_label)
    else:
        model_dir = model_label

    forecast_csv = target_dir / model_dir / "forecast" / "forecast.csv"
    if not forecast_csv.exists():
        raise FileNotFoundError(f"Forecast CSV not found: {forecast_csv}")

    df = pd.read_csv(forecast_csv, index_col=0, parse_dates=True)

    # Columns are like: target_actual, target_fitted, target_lower, target_upper
    fitted_col = f"{target_name}_fitted"
    lower_col = f"{target_name}_lower"
    upper_col = f"{target_name}_upper"

    result = {
        "forecast": df[fitted_col],
        "model_label": model_label,
    }
    if lower_col in df.columns:
        result["ci_lower"] = df[lower_col]
    if upper_col in df.columns:
        result["ci_upper"] = df[upper_col]

    # Load metadata for timestamps
    metadata_path = target_dir / model_dir / "forecast" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            result["metadata"] = json.load(f)

    return result


def aggregate_delu(forecasts_base: Path) -> pd.DataFrame:
    """Aggregate per-TSO forecasts to DE/LU national level.

    Returns DataFrame with columns: forecast, ci_lower, ci_upper for each
    of the 8 national aggregate quantities.
    """
    # Load per-TSO forecasts for each base component
    component_data = {}
    for component_name, tso_dirs in COMPONENTS.items():
        forecasts = []
        ci_lowers = []
        ci_uppers = []
        models_used = []

        for tso_dir_name in tso_dirs:
            target_dir = forecasts_base / tso_dir_name
            if not target_dir.exists():
                logger.warning(f"Missing target directory: {target_dir}")
                continue

            try:
                data = load_best_forecast(target_dir)
                forecasts.append(data["forecast"])
                if "ci_lower" in data:
                    ci_lowers.append(data["ci_lower"])
                if "ci_upper" in data:
                    ci_uppers.append(data["ci_upper"])
                models_used.append(f"{tso_dir_name}:{data['model_label']}")
            except (FileNotFoundError, KeyError) as e:
                logger.warning(f"Skipping {tso_dir_name}: {e}")
                continue

        if not forecasts:
            raise ValueError(f"No forecasts found for component '{component_name}'")

        # Align on common index and sum
        df_fc = pd.concat(forecasts, axis=1).sum(axis=1)
        component_data[component_name] = {"forecast": df_fc}

        # Sum CIs directly (conservative — assumes perfect positive correlation)
        if ci_lowers:
            component_data[component_name]["ci_lower"] = pd.concat(ci_lowers, axis=1).sum(axis=1)
        if ci_uppers:
            component_data[component_name]["ci_upper"] = pd.concat(ci_uppers, axis=1).sum(axis=1)

        logger.info(
            f"  {component_name}: {len(forecasts)} TSOs summed, "
            f"{len(df_fc)} hours, models: {models_used}"
        )

    # Compute derived quantities
    # gesamt = verbrauch + gen_load_diff_delu
    gen_load_diff_dir = forecasts_base / "gen_load_diff_delu"
    if gen_load_diff_dir.exists():
        gld_data = load_best_forecast(gen_load_diff_dir)
        gld_forecast = gld_data["forecast"]

        component_data["gesamt"] = {
            "forecast": component_data["verbrauch"]["forecast"] + gld_forecast
        }
        if "ci_lower" in gld_data and "ci_lower" in component_data["verbrauch"]:
            component_data["gesamt"]["ci_lower"] = (
                component_data["verbrauch"]["ci_lower"] + gld_data["ci_lower"]
            )
        if "ci_upper" in gld_data and "ci_upper" in component_data["verbrauch"]:
            component_data["gesamt"]["ci_upper"] = (
                component_data["verbrauch"]["ci_upper"] + gld_data["ci_upper"]
            )
        logger.info(f"  gesamt: verbrauch + gen_load_diff_delu (model: {gld_data['model_label']})")
    else:
        logger.warning("gen_load_diff_delu not found — gesamt will be unavailable")

    # wind_und_photovoltaik = onshore + offshore + photovoltaik
    component_data["wind_und_photovoltaik"] = {
        "forecast": (
            component_data["onshore"]["forecast"]
            + component_data["offshore"]["forecast"]
            + component_data["photovoltaik"]["forecast"]
        )
    }
    for ci_key in ["ci_lower", "ci_upper"]:
        parts = [component_data[c].get(ci_key) for c in ["onshore", "offshore", "photovoltaik"]]
        if all(p is not None for p in parts):
            component_data["wind_und_photovoltaik"][ci_key] = parts[0] + parts[1] + parts[2]

    # sonstige = gesamt - wind_und_photovoltaik
    if "gesamt" in component_data:
        component_data["sonstige"] = {
            "forecast": (
                component_data["gesamt"]["forecast"]
                - component_data["wind_und_photovoltaik"]["forecast"]
            )
        }
        # For sonstige CI: gesamt_upper - wsp_lower gives sonstige_upper, etc.
        if "ci_upper" in component_data["gesamt"] and "ci_lower" in component_data["wind_und_photovoltaik"]:
            component_data["sonstige"]["ci_upper"] = (
                component_data["gesamt"]["ci_upper"]
                - component_data["wind_und_photovoltaik"]["ci_lower"]
            )
        if "ci_lower" in component_data["gesamt"] and "ci_upper" in component_data["wind_und_photovoltaik"]:
            component_data["sonstige"]["ci_lower"] = (
                component_data["gesamt"]["ci_lower"]
                - component_data["wind_und_photovoltaik"]["ci_upper"]
            )

    # residuallast = verbrauch - wind_und_photovoltaik
    component_data["residuallast"] = {
        "forecast": (
            component_data["verbrauch"]["forecast"]
            - component_data["wind_und_photovoltaik"]["forecast"]
        )
    }
    if "ci_upper" in component_data["verbrauch"] and "ci_lower" in component_data["wind_und_photovoltaik"]:
        component_data["residuallast"]["ci_upper"] = (
            component_data["verbrauch"]["ci_upper"]
            - component_data["wind_und_photovoltaik"]["ci_lower"]
        )
    if "ci_lower" in component_data["verbrauch"] and "ci_upper" in component_data["wind_und_photovoltaik"]:
        component_data["residuallast"]["ci_lower"] = (
            component_data["verbrauch"]["ci_lower"]
            - component_data["wind_und_photovoltaik"]["ci_upper"]
        )

    return component_data


def export_json(component_data: dict, output_dir: Path) -> None:
    """Write per-component JSON files matching the existing deploy API schema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for key, fname in JSON_FILES.items():
        if key not in component_data:
            logger.warning(f"Skipping {fname}: no data for '{key}'")
            continue

        data = component_data[key]
        fc = data["forecast"]

        # Build data array
        records = []
        for i, (dt, val) in enumerate(fc.items()):
            record = {
                "index": i,
                "datetime": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "forecast": float(val),
            }
            if "ci_lower" in data:
                record["ci_lower"] = float(data["ci_lower"].iloc[i])
            if "ci_upper" in data:
                record["ci_upper"] = float(data["ci_upper"].iloc[i])
            records.append(record)

        data_keys = ["forecast"]
        if "ci_lower" in data:
            data_keys.append("ci_lower")
        if "ci_upper" in data:
            data_keys.append("ci_upper")

        json_data = {
            "metadata": {
                "file": fname,
                "data_keys": data_keys,
                "target_name": TARGET_NAMES[key],
                "tso_region": "DE_LU",
                "model_label": "national_aggregate",
                "forecast_datetime": now_str,
                "source": "https://smnfrse.github.io/energy_market_analysis/",
                "forecast_horizon_hours": len(fc),
                "units": "MW",
                "notes": "National DE/LU aggregate. CIs summed directly (conservative, assumes correlated errors).",
            },
            "data": records,
        }

        fpath = output_dir / fname
        with open(fpath, "w") as f:
            json.dump(json_data, f, indent=4)
        logger.info(f"Wrote {fpath} ({len(records)} hours)")


def export_downloadable(component_data: dict, output_dir: Path) -> None:
    """Write combined CSV with all 8 columns for bulk download."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()
    for key, csv_col in CSV_COLUMNS.items():
        if key in component_data:
            df[csv_col] = component_data[key]["forecast"]

    df.index.name = "date"
    fpath = output_dir / "national_forecasts.csv"
    df.to_csv(fpath)
    logger.info(f"Wrote {fpath} ({len(df)} rows, {len(df.columns)} columns)")


def main(output_base: str = "deploy/data/DE"):
    forecasts_base = Path("output/DE/forecasts")
    output_base = Path(output_base)

    logger.info("Aggregating per-TSO forecasts to DE/LU national level...")
    component_data = aggregate_delu(forecasts_base)

    logger.info("Exporting JSON files...")
    export_json(component_data, output_base / "api" / "forecasts")

    logger.info("Exporting downloadable CSV...")
    export_downloadable(component_data, output_base / "downloads")

    # Sanity checks
    for key in ["gesamt", "wind_und_photovoltaik", "sonstige"]:
        if key in component_data:
            fc = component_data[key]["forecast"]
            logger.info(f"  {key}: mean={fc.mean():.0f} MW, min={fc.min():.0f}, max={fc.max():.0f}")

    # Verify algebraic identity: sonstige = gesamt - wind_und_photovoltaik
    if all(k in component_data for k in ["gesamt", "wind_und_photovoltaik", "sonstige"]):
        diff = (
            component_data["gesamt"]["forecast"]
            - component_data["wind_und_photovoltaik"]["forecast"]
            - component_data["sonstige"]["forecast"]
        )
        if diff.abs().max() > 0.01:
            logger.error(f"Algebraic check FAILED: max deviation = {diff.abs().max():.4f}")
        else:
            logger.info("Algebraic check passed: sonstige = gesamt - wind_und_photovoltaik")

    logger.info("Done.")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "deploy/data/DE"
    main(output_dir)
