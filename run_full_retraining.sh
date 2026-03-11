#!/bin/bash
# Full retraining pipeline: finetune + train all targets, all models on SMARD v2 data
# Base models must be finetuned+trained before ensembles
set -e

CONDA="conda run -n energy_market --no-capture-output"
LOG="full_retraining.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"
}

log "=== FULL RETRAINING START ==="

# --- Single targets: base models first, then ensembles ---
for target in wind_offshore wind_onshore solar load; do
    log "--- $target: finetuning base models ---"
    for model in LightGBM XGBoost ElasticNet; do
        log "FINETUNE $target / $model"
        $CONDA python update_forecasts.py DE "$target" "$model" finetune hourly 2>&1 | tee -a "$LOG"
    done

    log "--- $target: training base models ---"
    for model in LightGBM XGBoost ElasticNet; do
        log "TRAIN $target / $model"
        $CONDA python update_forecasts.py DE "$target" "$model" train hourly 2>&1 | tee -a "$LOG"
    done

    log "--- $target: finetuning ensembles ---"
    for model in "ensemble[XGBoost](XGBoost,ElasticNet)" "ensemble[LightGBM](LightGBM,ElasticNet)"; do
        log "FINETUNE $target / $model"
        $CONDA python update_forecasts.py DE "$target" "$model" finetune hourly 2>&1 | tee -a "$LOG"
    done

    log "--- $target: training ensembles ---"
    for model in "ensemble[XGBoost](XGBoost,ElasticNet)" "ensemble[LightGBM](LightGBM,ElasticNet)"; do
        log "TRAIN $target / $model"
        $CONDA python update_forecasts.py DE "$target" "$model" train hourly 2>&1 | tee -a "$LOG"
    done

    log "=== $target COMPLETE ==="
done

# --- Energy mix: multi-target base models first, then ensemble ---
log "--- energy_mix: finetuning base models ---"
for model in MultiTargetLGBM MultiTargetCatBoost MultiTargetElasticNet; do
    log "FINETUNE energy_mix / $model"
    $CONDA python update_forecasts.py DE energy_mix "$model" finetune hourly 2>&1 | tee -a "$LOG"
done

log "--- energy_mix: training base models ---"
for model in MultiTargetLGBM MultiTargetCatBoost MultiTargetElasticNet; do
    log "TRAIN energy_mix / $model"
    $CONDA python update_forecasts.py DE energy_mix "$model" train hourly 2>&1 | tee -a "$LOG"
done

log "--- energy_mix: finetuning ensemble ---"
log "FINETUNE energy_mix / ensemble[MultiTargetLGBM]"
$CONDA python update_forecasts.py DE energy_mix "ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)" finetune hourly 2>&1 | tee -a "$LOG"

log "--- energy_mix: training ensemble ---"
log "TRAIN energy_mix / ensemble[MultiTargetLGBM]"
$CONDA python update_forecasts.py DE energy_mix "ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)" train hourly 2>&1 | tee -a "$LOG"

log "=== energy_mix COMPLETE ==="
log "=== FULL RETRAINING DONE ==="
