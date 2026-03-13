# Predictive Analytics for the German Energy Market

> **This is a fork of [vsevolodnedora/energy_market_analysis](https://github.com/vsevolodnedora/energy_market_analysis).**
> All original domain knowledge -- curated weather station locations, physics-based feature engineering, spatial aggregation methods, and the overall forecasting architecture -- comes from the upstream project by Vsevolod Nedora. See the [upstream README](https://github.com/vsevolodnedora/energy_market_analysis#readme) and the author's [Medium articles](https://medium.com/@vsevolod.nedora) for the full project history and motivation. This fork adapts the infrastructure for a specific use case (see below) and is not affiliated with the original author.

## What This Fork Does

This fork produces **weather-based forecasts of electricity generation and load** for the German market, broken down by TSO region (50Hertz, TenneT, Amprion, TransnetBW). It is a companion to a separate [energy_prices](https://github.com/smnfrs) project that forecasts day-ahead electricity prices.

The problem: SMARD publishes TSO generation/load forecasts at ~18:00 CET, six hours *after* the 12:00 CET day-ahead auction. The price models need these forecasts as input features but cannot wait for the official numbers. This project produces equivalent forecasts from Open-Meteo weather data, which is available days in advance.

### Forecasting targets

| Target | Models | Horizon | TSO regions |
|--------|--------|---------|-------------|
| Wind offshore | LightGBM, XGBoost, ElasticNet, ensemble | 168h (7 days) | 50Hz, TenneT |
| Wind onshore | LightGBM, XGBoost, ElasticNet, ensemble | 168h | All 4 |
| Solar | LightGBM, XGBoost, ElasticNet, ensemble | 168h | All 4 |
| Load | LightGBM, XGBoost, ElasticNet, ensemble | 168h | All 4 |
| Energy mix | MultiTargetLGBM, CatBoost | 168h | All 4 |

## Changes From Upstream

### Completed

1. **Replaced ENTSO-E with SMARD v2** -- The upstream repo requires an ENTSO-E API key (multi-month wait). This fork uses the SMARD v2 API for per-TSO generation and load actuals, eliminating the ENTSO-E dependency entirely. (`8eeacca`, `b754a9a`)

2. **Added SMARD v2 per-TSO collector** -- New `collect_data_smard_v2.py` module that fetches generation and load data for each TSO from the SMARD v2 API, with proper handling of missing data series (e.g., no offshore wind for Amprion/TransnetBW, no lignite for TenneT/TransnetBW). (`1e62d7e`)

3. **Redefined energy_mix targets** -- Adapted from ENTSO-E's fine-grained generation splits to SMARD's categories: `hydro`, `other_conv`, `other_renew` replace the ENTSO-E-specific subdivisions. (`8eeacca`)

4. **Fixed XGBoost + MAPIE precision crash** -- XGBoost 3.x predicts float32 while MAPIE's `AbsoluteConformityScore` defaulted to `eps=1e-6` (below float32 precision). Fixed by setting `eps=1e-4`. (`420aded`)

5. **Enabled CatBoost in energy_mix pipeline** -- CatBoost was commented out in all task stages in the upstream repo. Uncommented and wired into the pipeline. (`420aded`)

6. **Added batch retraining script** -- `run_full_retraining.sh` runs finetune + train for all targets and models sequentially. (`809a1e2`)

### Planned

7. **`gen_load_diff` national target** -- Forecasting the generation-load differential for the DE/LU bidding zone. This allows deriving `sonstige` (conventional generation = total_gen - wind - solar) which is a key price model feature. Implementation involves adding Luxembourg (Creos TSO) data collection and weather locations, then training a national-level model. See `scratch/total-generation-forecast-plan.md` for background analysis.

## Setup

### Prerequisites

- Python 3.11 (tested with 3.11.5)
- conda (recommended) or pip

### Installation

```bash
# Clone this fork
git clone <this-repo-url>
cd energy_market_analysis

# Create conda environment
conda create -n energy_market python=3.11
conda activate energy_market

# Install dependencies
pip install -r requirements.txt
```

No API keys are required. All data sources (SMARD, Open-Meteo) are freely accessible.

### Key dependencies

- lightgbm 4.6, xgboost 3.2, catboost
- mapie 0.9.2 (not 1.x -- API break with `MapieRegressor`)
- optuna (hyperparameter tuning)
- open-meteo SDK, pandas, scikit-learn

## Usage

The pipeline has three stages, run in order.

### 1. Update database

Collect weather and energy data from APIs:

```bash
# SMARD generation/load actuals (DE only)
python update_database.py DE update_smard hourly

# Open-Meteo weather data (multiple location types)
python update_database.py all update_openmeteo_windfarms_offshore hourly
python update_database.py all update_openmeteo_windfarms_onshore hourly
python update_database.py all update_openmeteo_solarfarms hourly
python update_database.py all update_openmeteo_cities hourly
```

### 2. Update forecasts

Train models and generate forecasts:

```bash
# Finetune hyperparameters (Optuna, computationally expensive -- run on-premises)
python update_forecasts.py DE wind_offshore LightGBM finetune hourly

# Train model on full dataset with best parameters
python update_forecasts.py DE wind_offshore LightGBM train hourly

# Generate forecasts using trained models (lightweight -- runs in GitHub Actions)
python update_forecasts.py DE all all forecast hourly

# Run everything for all targets
python update_forecasts.py DE all all all hourly
```

### 3. Publish data

Prepare forecasts for the static webpage:

```bash
python publish_data.py DE all
```

### Batch retraining

To retrain all models from scratch:

```bash
bash run_full_retraining.sh
```

### Argument reference

| Script | Argument | Values |
|--------|----------|--------|
| `update_database.py` | country | `DE`, `FR`, `all` |
| | task | `update_smard`, `update_openmeteo_windfarms_offshore`, `update_openmeteo_windfarms_onshore`, `update_openmeteo_solarfarms`, `update_openmeteo_cities`, `all` |
| | freq | `hourly`, `minutely_15` |
| `update_forecasts.py` | target | `wind_offshore`, `wind_onshore`, `solar`, `load`, `energy_mix`, `all` |
| | model | `LightGBM`, `XGBoost`, `ElasticNet`, `ensemble[...]`, `MultiTargetLGBM`, `MultiTargetElasticNet`, `all` |
| | mode | `finetune`, `train`, `forecast`, `plot`, `summarize`, `all` |

## Architecture

For detailed architecture documentation, see [CLAUDE.md](CLAUDE.md).

### Data flow

```
Open-Meteo API ──> database/{country}/openmeteo/   ──┐
SMARD v2 API   ──> database/{country}/smard/        ──┤──> update_forecasts.py ──> output/{country}/forecasts/
                                                      │                                    │
                                                      └──────────────────────── publish_data.py ──> deploy/data/
```

### Key modules

- **`data_collection_modules/`** -- API collectors and location metadata. `eu_locations.py` contains 30+ curated weather station coordinates near actual generation assets, TSO mappings, and installed capacity data.
- **`data_modules/`** -- Physics-informed feature engineering (`feature_eng.py`): wind power density, air density correction, wind shear profiles, solar angle calculations, heating/cooling degree days.
- **`forecasting_modules/`** -- ML pipeline with Optuna hyperparameter tuning, MAPIE prediction intervals, and ensemble stacking.

### Pipeline flow

1. `update_forecasts.py` creates task configs per target/TSO region
2. `interface.py` loads and cleans data, creates forecasting tasks
3. Tasks execute: **finetune** (Optuna) -> **train** (full dataset) -> **forecast** (inference) -> **summarize** (metrics)
4. Finetuning/training run on-premises; forecasting runs in GitHub Actions

## Model Performance

Best single-model R² (5-week rolling evaluation, SMARD v2 targets):

| Target | Best model | R² range across TSOs |
|--------|-----------|---------------------|
| Solar | LightGBM | 0.32 -- 0.90 |
| Load | Ensemble | 0.77 -- 0.91 |
| Wind onshore | LightGBM | 0.45 -- 0.80 |
| Wind offshore | Ensemble | 0.51 -- 0.81 |

Energy mix (conventional generation by fuel type) has poor performance (negative R² on gas/coal/lignite) due to missing economic features (fuel prices, CO2 prices). This is a known limitation also present in the upstream repo.

## License

The upstream project codebase is licensed under the MIT License.

Datasets collected and used in this project may be subject to additional licensing. See:
- [SMARD](https://www.smard.de/home)
- [Open-Meteo](https://open-meteo.com/)
- [EPEX SPOT](https://www.epexspot.com/en)

## Credits

This project is a fork of [vsevolodnedora/energy_market_analysis](https://github.com/vsevolodnedora/energy_market_analysis) by [Vsevolod Nedora](https://github.com/vsevolodnedora). The upstream project provides:

- Curated locations of wind farms, solar parks, and cities across Germany with TSO mappings
- Physics-informed feature engineering for wind power, solar power, and load forecasting
- Multi-step recursive forecasting architecture with ensemble stacking
- Spatial aggregation using installed capacity weights
- Automated pipeline with GitHub Actions

See the upstream repo for the full project history, live demo, and detailed technical write-up.
