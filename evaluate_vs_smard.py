"""Evaluate our forecasts against SMARD's official TSO forecasts.

Compares our weather-based national DE/LU forecasts (available days in advance)
against SMARD's official forecasts (published ~18:00 CET, after the day-ahead auction).
Both are evaluated against SMARD actuals.

Usage:
    python evaluate_vs_smard.py [--weeks N]  # N weeks of rolling eval (default 5)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


# ── Map our national aggregates to SMARD columns ────────────────────────────

# SMARD legacy DB columns (from database/DE/smard/history_hourly.parquet)
SMARD_FORECAST_MAP = {
    "wind_onshore":  "wind_onshore_forecasted",
    "wind_offshore": "wind_offshore_forecasted",
    "solar":         "solar_forecasted",
    "load":          "total_grid_load_forecasted",
    "gesamt":        "total_gen_forecasted",
    "sonstige":      "other_gen_forecasted",
    "residuallast":  "residual_load_forecasted",
}

# SMARD v2 per-TSO actual columns to sum for national-level actuals
SMARD_ACTUAL_SUM = {
    "wind_onshore":  ["wind_onshore_50hz", "wind_onshore_ampr", "wind_onshore_tenn",
                      "wind_onshore_tran", "wind_onshore_lu"],
    "wind_offshore": ["wind_offshore_50hz", "wind_offshore_tenn"],
    "solar":         ["solar_50hz", "solar_ampr", "solar_tenn", "solar_tran", "solar_lu"],
    "load":          ["load_50hz", "load_ampr", "load_tenn", "load_tran", "load_lu"],
}

# Our forecast result CSV column naming
OUR_COMPONENTS = {
    "wind_onshore":  ["wind_onshore_50hz", "wind_onshore_ampr", "wind_onshore_tenn",
                      "wind_onshore_tran", "wind_onshore_lu"],
    "wind_offshore": ["wind_offshore_50hz", "wind_offshore_tenn"],
    "solar":         ["solar_50hz", "solar_ampr", "solar_tenn", "solar_tran", "solar_lu"],
    "load":          ["load_50hz", "load_ampr", "load_tenn", "load_tran", "load_lu"],
}


# ── Metrics ──────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    return 200.0 * np.nanmean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])


# ── Load our per-TSO forecasts and aggregate ─────────────────────────────────

def load_our_forecast(target_name: str, tso_dirs: list[str],
                      forecasts_base: Path) -> pd.Series:
    """Load and sum our per-TSO forecast CSVs for a given target."""
    from forecasting_modules.utils import convert_ensemble_string

    series_list = []
    for tso_dir_name in tso_dirs:
        target_dir = forecasts_base / tso_dir_name
        if not target_dir.exists():
            continue

        # Find best model — prefer best_model.json (honest rolling CV)
        for json_name in ["best_model.json", "best_model_forecast.json"]:
            json_path = target_dir / json_name
            if json_path.exists():
                with open(json_path) as f:
                    best_models = json.load(f)
                break
        else:
            continue

        target_key = list(best_models.keys())[0]
        model_label = best_models[target_key]["model_label"]
        if "ensemble" in model_label:
            model_dir = convert_ensemble_string(model_label)
        else:
            model_dir = model_label

        # Load the trained result.csv (has actuals + fitted on historical data)
        result_csv = target_dir / model_dir / "trained" / "result.csv"
        if not result_csv.exists():
            # Fall back to forecast/result.csv
            result_csv = target_dir / model_dir / "forecast" / "result.csv"
        if not result_csv.exists():
            logger.warning(f"No result.csv for {tso_dir_name}")
            continue

        df = pd.read_csv(result_csv, index_col=0, parse_dates=True)
        fitted_col = f"{tso_dir_name}_fitted"
        if fitted_col in df.columns:
            series_list.append(df[fitted_col])
        else:
            logger.warning(f"Column {fitted_col} not in {result_csv}")

    if not series_list:
        return pd.Series(dtype=float)

    return pd.concat(series_list, axis=1).sum(axis=1, min_count=1)


# ── Main evaluation ──────────────────────────────────────────────────────────

def load_daily_evaluation(target_name: str, tso_dirs: list[str],
                          forecasts_base: Path) -> pd.DataFrame | None:
    """Load evaluation_daily.csv for each TSO, sum fitted + actual to national level.

    Returns DataFrame with columns: national_actual, national_fitted, hours_ahead, cutoff.
    Returns None if no evaluation_daily.csv files are found.
    """
    from forecasting_modules.utils import convert_ensemble_string

    fitted_dfs = []
    actual_dfs = []
    hours_ahead_series = None

    for tso_dir_name in tso_dirs:
        target_dir = forecasts_base / tso_dir_name
        if not target_dir.exists():
            continue

        # Find best model — prefer best_model.json (honest rolling CV)
        for json_name in ["best_model.json", "best_model_forecast.json"]:
            json_path = target_dir / json_name
            if json_path.exists():
                with open(json_path) as f:
                    best_models = json.load(f)
                break
        else:
            continue

        target_key = list(best_models.keys())[0]
        model_label = best_models[target_key]["model_label"]
        if "ensemble" in model_label:
            model_dir = convert_ensemble_string(model_label)
        else:
            model_dir = model_label

        eval_csv = target_dir / model_dir / "forecast" / "evaluation_daily.csv"
        if not eval_csv.exists():
            return None  # If any TSO lacks daily eval, fall back entirely

        df = pd.read_csv(eval_csv, index_col=0, parse_dates=True)
        fitted_col = f"{tso_dir_name}_fitted"
        actual_col = f"{tso_dir_name}_actual"

        if fitted_col not in df.columns or actual_col not in df.columns:
            logger.warning(f"Missing columns in {eval_csv}")
            return None

        fitted_dfs.append(df[[fitted_col, "cutoff", "hours_ahead"]].rename(
            columns={fitted_col: tso_dir_name}
        ))
        actual_dfs.append(df[[actual_col]].rename(
            columns={actual_col: tso_dir_name}
        ))

    if not fitted_dfs:
        return None

    # Sum across TSOs, keeping cutoff + hours_ahead from the first TSO
    df_fitted = pd.concat([d.drop(columns=["cutoff", "hours_ahead"]) for d in fitted_dfs], axis=1)
    df_actual = pd.concat(actual_dfs, axis=1)
    meta = fitted_dfs[0][["cutoff", "hours_ahead"]]

    result = pd.DataFrame({
        "national_fitted": df_fitted.sum(axis=1, min_count=1),
        "national_actual": df_actual.sum(axis=1, min_count=1),
        "cutoff": meta["cutoff"],
        "hours_ahead": meta["hours_ahead"],
    })
    return result


def compute_metrics(y_true, y_pred):
    """Compute metrics for a pair of arrays, returning a dict."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 12:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "smape": np.nan, "hours": 0}
    yt, yp = y_true[valid], y_pred[valid]
    return {
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "r2": r2(yt, yp),
        "smape": smape(yt, yp),
        "hours": int(valid.sum()),
    }


def compute_actual_gen_load_diff(smard_v2: pd.DataFrame) -> pd.Series:
    """Compute actual gen_load_diff from SMARD v2 per-TSO data.

    gen_load_diff = total_generation - total_load (national DE/LU).
    """
    gen_cols = [c for c in smard_v2.columns if not c.startswith("load_")
                and not c.startswith("generation_")]
    # Sum all non-load generation columns per TSO
    load_cols = [c for c in smard_v2.columns if c.startswith("load_")]
    # For generation, sum everything except load columns
    all_gen_cols = [c for c in smard_v2.columns if c not in load_cols]
    total_gen = smard_v2[all_gen_cols].sum(axis=1, min_count=1)
    total_load = smard_v2[load_cols].sum(axis=1, min_count=1)
    return total_gen - total_load


def _evaluate_target_daily(target_name: str, tso_dirs: list[str],
                            forecasts_base: Path, actuals: pd.Series,
                            eval_start, eval_end) -> list[dict]:
    """Evaluate a target using daily-rolling evaluation_daily.csv files.

    Returns list of per-horizon-day metric dicts, or empty list if no daily eval data.
    """
    df_eval = load_daily_evaluation(target_name, tso_dirs, forecasts_base)
    if df_eval is None:
        return []

    # Filter to eval window
    df_eval = df_eval[(df_eval.index >= eval_start) & (df_eval.index <= eval_end)]
    # Align with SMARD actuals (overwrite the model's own actuals with ground truth)
    df_eval["smard_actual"] = actuals.reindex(df_eval.index)
    valid = df_eval["smard_actual"].notna() & df_eval["national_fitted"].notna()
    df_eval = df_eval[valid]

    if len(df_eval) < 24:
        return []

    rows = []
    hours_ahead = df_eval["hours_ahead"].values
    y_actual = df_eval["smard_actual"].values
    y_fitted = df_eval["national_fitted"].values

    # Per-horizon day metrics
    for horizon_day in range(1, 8):
        lo = (horizon_day - 1) * 24
        hi = horizon_day * 24
        mask = (hours_ahead >= lo) & (hours_ahead < hi)
        if mask.sum() < 12:
            continue
        m = compute_metrics(y_actual[mask], y_fitted[mask])
        rows.append({"target": target_name, "day": horizon_day, **m})

    # Overall
    m_all = compute_metrics(y_actual, y_fitted)
    rows.append({"target": target_name, "day": "all", **m_all})
    return rows


def _evaluate_target_legacy(target_name: str, tso_dirs: list[str],
                             forecasts_base: Path, actuals: pd.Series,
                             eval_start, eval_end) -> list[dict]:
    """Fallback evaluation using result.csv (no per-horizon stratification).

    Returns overall-only metrics when evaluation_daily.csv is not available.
    """
    our_forecast = load_our_forecast(target_name, tso_dirs, forecasts_base)
    if our_forecast.empty:
        return []

    common_idx = actuals.index.intersection(our_forecast.index)
    common_idx = common_idx[(common_idx >= eval_start) & (common_idx <= eval_end)]
    if len(common_idx) < 24:
        return []

    y_actual = actuals.loc[common_idx].values
    y_ours = our_forecast.loc[common_idx].values
    m_all = compute_metrics(y_actual, y_ours)
    return [{"target": target_name, "day": "all", **m_all}]


def evaluate(weeks: int = 5):
    """Run the comparison and print results."""
    forecasts_base = Path("output/DE/forecasts")

    # Load SMARD legacy data (has official forecasts)
    smard_legacy = pd.read_parquet("database/DE/smard/history_hourly.parquet")
    # Load SMARD v2 data (has per-TSO actuals)
    smard_v2 = pd.read_parquet("database/DE/smard_v2/history_hourly.parquet")

    # Determine evaluation window
    eval_end = smard_legacy.index[-1]
    eval_start = eval_end - pd.Timedelta(weeks=weeks)
    logger.info(f"Evaluation window: {eval_start} to {eval_end} ({weeks} weeks)")

    our_day_rows = []     # per-day breakdown of our forecast
    smard_rows = []       # single SMARD row per target
    targets_evaluated = []
    has_daily_eval = {}   # track which targets have daily-rolling data

    for target_name in ["wind_onshore", "wind_offshore", "solar", "load"]:
        logger.info(f"Evaluating {target_name}...")

        # 1. Compute national actuals from SMARD v2 per-TSO data
        actual_cols = SMARD_ACTUAL_SUM[target_name]
        available = [c for c in actual_cols if c in smard_v2.columns]
        actuals = smard_v2[available].sum(axis=1, min_count=1)

        # 2. SMARD official forecast — evaluate on full window (always day-ahead)
        smard_fc_col = SMARD_FORECAST_MAP[target_name]
        if smard_fc_col in smard_legacy.columns:
            smard_window = smard_legacy.loc[eval_start:eval_end, smard_fc_col]
            actuals_for_smard = actuals.reindex(smard_window.index)
            both_valid = smard_window.notna() & actuals_for_smard.notna()
            if both_valid.sum() > 24:
                sm = compute_metrics(
                    actuals_for_smard[both_valid].values,
                    smard_window[both_valid].values,
                )
                smard_rows.append({"target": target_name, **sm})

        # 3. Try daily-rolling evaluation first (proper per-horizon metrics)
        daily_rows = _evaluate_target_daily(
            target_name, OUR_COMPONENTS[target_name], forecasts_base, actuals,
            eval_start, eval_end
        )
        if daily_rows:
            has_daily_eval[target_name] = True
            our_day_rows.extend(daily_rows)
            targets_evaluated.append(target_name)
            logger.info(f"  {target_name}: using daily-rolling evaluation ({len(daily_rows)-1} horizon days)")
        else:
            # Fall back to result.csv with overall-only metrics
            has_daily_eval[target_name] = False
            legacy_rows = _evaluate_target_legacy(
                target_name, OUR_COMPONENTS[target_name], forecasts_base, actuals,
                eval_start, eval_end
            )
            if legacy_rows:
                our_day_rows.extend(legacy_rows)
                targets_evaluated.append(target_name)
                logger.warning(
                    f"  {target_name}: no evaluation_daily.csv found, using result.csv (no per-day breakdown). "
                    f"Run 'python update_forecasts.py DE {target_name.replace('_', '_')} all evaluate hourly' to generate."
                )

    # ── gen_load_diff evaluation ─────────────────────────────────────────
    logger.info("Evaluating gen_load_diff_delu...")

    # Actual gen_load_diff from SMARD v2
    load_cols = [c for c in smard_v2.columns if c.startswith("load_")]
    non_load_cols = [c for c in smard_v2.columns if not c.startswith("load_")]
    actual_total_gen = smard_v2[non_load_cols].sum(axis=1, min_count=1)
    actual_total_load = smard_v2[load_cols].sum(axis=1, min_count=1)
    actual_gld = actual_total_gen - actual_total_load

    # SMARD's implied gen_load_diff = total_gen_forecasted - total_grid_load_forecasted
    if "total_gen_forecasted" in smard_legacy.columns and "total_grid_load_forecasted" in smard_legacy.columns:
        smard_gld = smard_legacy["total_gen_forecasted"] - smard_legacy["total_grid_load_forecasted"]
        smard_gld_window = smard_gld.loc[eval_start:eval_end]
        actual_gld_for_smard = actual_gld.reindex(smard_gld_window.index)
        both_valid = smard_gld_window.notna() & actual_gld_for_smard.notna()
        if both_valid.sum() > 24:
            sm = compute_metrics(
                actual_gld_for_smard[both_valid].values,
                smard_gld_window[both_valid].values,
            )
            smard_rows.append({"target": "gen_load_diff", **sm})

    # Our gen_load_diff_delu — try daily evaluation first
    gld_dir = forecasts_base / "gen_load_diff_delu"
    if gld_dir.exists():
        daily_rows = _evaluate_target_daily(
            "gen_load_diff", ["gen_load_diff_delu"], forecasts_base, actual_gld,
            eval_start, eval_end
        )
        if daily_rows:
            has_daily_eval["gen_load_diff"] = True
            our_day_rows.extend(daily_rows)
            targets_evaluated.append("gen_load_diff")
        else:
            has_daily_eval["gen_load_diff"] = False
            legacy_rows = _evaluate_target_legacy(
                "gen_load_diff", ["gen_load_diff_delu"], forecasts_base, actual_gld,
                eval_start, eval_end
            )
            if legacy_rows:
                our_day_rows.extend(legacy_rows)
                targets_evaluated.append("gen_load_diff")

    # ── Print results ────────────────────────────────────────────────────
    df_ours = pd.DataFrame(our_day_rows)
    df_smard = pd.DataFrame(smard_rows)

    any_daily = any(has_daily_eval.values())
    eval_mode = "daily-rolling (step=24h)" if any_daily else "legacy (fold-position)"

    print("\n" + "=" * 100)
    print("  FORECAST EVALUATION: Our Models vs SMARD Official Forecasts")
    print(f"  Window: {eval_start.strftime('%Y-%m-%d')} to {eval_end.strftime('%Y-%m-%d')} ({weeks} weeks)")
    print(f"  Horizon mode: {eval_mode}")
    print("=" * 100)

    # Print SMARD baseline (one row per target — always day-ahead)
    print("\n  SMARD OFFICIAL FORECASTS (day-ahead, published ~18:00 CET)")
    print(f"  {'Target':<16} {'Hours':>6}  {'RMSE':>10} {'MAE':>9} {'R²':>8}")
    print("  " + "-" * 52)
    for _, row in df_smard.iterrows():
        print(f"  {row['target']:<16} {row['hours']:>6}  "
              f"{row['rmse']:>10.0f} {row['mae']:>9.0f} {row['r2']:>8.3f}")

    # Print our forecast by horizon day
    print("\n  OUR FORECASTS (weather-based, by forecast horizon day)")
    for target_name in targets_evaluated:
        target_rows = df_ours[df_ours["target"] == target_name]
        if target_rows.empty:
            continue

        # Get SMARD baseline for comparison
        smard_row = df_smard[df_smard["target"] == target_name]
        smard_label = ""
        if not smard_row.empty:
            sr = smard_row.iloc[0]
            smard_label = f"  (SMARD: RMSE={sr['rmse']:.0f}, MAE={sr['mae']:.0f}, R²={sr['r2']:.3f})"

        daily_marker = " [daily-rolling]" if has_daily_eval.get(target_name) else " [overall only]"
        print(f"\n  {target_name.upper()}{smard_label}{daily_marker}")
        print(f"  {'Day':<6} {'Hours':>6}  {'RMSE':>10} {'MAE':>9} {'R²':>8}")
        print("  " + "-" * 45)

        for _, row in target_rows.iterrows():
            day_label = f"  d{row['day']}" if row['day'] != 'all' else "  ALL"
            print(f"  {day_label:<6} {row['hours']:>6}  "
                  f"{row['rmse']:>10.0f} {row['mae']:>9.0f} {row['r2']:>8.3f}")

    print("\n" + "-" * 100)
    print("Notes:")
    if any_daily:
        print("  - d1 = hours 0-23 ahead (day-ahead, comparable to SMARD)")
        print("  - d7 = hours 144-167 ahead (7 days out)")
        print("  - Daily-rolling: each calendar day appears at all 7 horizon positions (~30 samples/day)")
    else:
        print("  - No evaluation_daily.csv found — showing overall metrics only")
        print("  - Run 'python update_forecasts.py DE all all evaluate hourly' to generate per-horizon data")
    print("  - SMARD metrics computed once on full window (always day-ahead)")
    print("  - gen_load_diff: SMARD derived as total_gen_forecasted - total_grid_load_forecasted")
    print("  - Actuals: SMARD v2 per-TSO data, summed to national DE/LU")
    print("  - Lower RMSE/MAE and higher R² = better\n")

    # Save to CSV
    outpath = Path("output/DE/evaluation_vs_smard.csv")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df_ours_out = df_ours.copy()
    df_ours_out.columns = [f"our_{c}" if c not in ("target", "day") else c for c in df_ours_out.columns]
    df_smard_out = df_smard.copy()
    df_smard_out.columns = [f"smard_{c}" if c != "target" else c for c in df_smard_out.columns]
    df_ours_out.to_csv(outpath, index=False)
    df_smard_out.to_csv(outpath.with_name("evaluation_smard_baseline.csv"), index=False)
    logger.info(f"Results saved to {outpath}")


if __name__ == "__main__":
    weeks = 5
    if len(sys.argv) > 1:
        if sys.argv[1] == "--weeks" and len(sys.argv) > 2:
            weeks = int(sys.argv[2])
        else:
            try:
                weeks = int(sys.argv[1])
            except ValueError:
                pass
    evaluate(weeks)
