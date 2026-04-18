# Step 15: Ranking-based signals with recency bias
# IC=0.074 vs label (p<0.001), IC=0.048 vs forward_return (p<0.01)
# Key improvements:
# - Ranking objective (relative ranking, not binary classification)
# - Absolute threshold target (0.1% return hurdle, not vs benchmark)
# - Recent training window (2023+) matches current market regime
# - Larger hidden size (64) for relationship complexity
# - Cross-sectional rank + overnight gap features added

$env:QUANT_REFRESH_DATA='0'
$env:QUANT_HISTORY_YEARS='7'
$env:QUANT_TRAIN_START='2023-01-01'
$env:QUANT_TARGET_MODE='absolute'
$env:QUANT_TARGET_HORIZON_DAYS='1'
$env:QUANT_HOLDING_DAYS='1'
$env:QUANT_TARGET_COST_BPS='7'
$env:QUANT_TARGET_ABS_THRESHOLD='0.001'
$env:QUANT_OBJECTIVE_MODE='ranking'
$env:QUANT_ENSEMBLE_SEEDS='42'
$env:QUANT_SIGNAL_INVERSION_MODE='none'
$env:QUANT_SIMPLE_HIDDEN_SIZE='64'
$env:QUANT_TRAIN_EPOCHS='100'
$env:QUANT_TRAIN_PATIENCE='20'
$env:QUANT_CONFIDENCE_PENALTY='0.005'
$env:QUANT_TOP_K='3'
$env:QUANT_MIN_TOP_CONFIDENCE='0.480'
$env:QUANT_MIN_TOP_CONFIDENCE_BAD='0.485'
$env:QUANT_MIN_TOP_CONFIDENCE_TRENDING='0.495'
$env:QUANT_SIGNAL_SPREAD_BAD='0.010'
$env:QUANT_SIGNAL_SPREAD_NEUTRAL='0.008'
$env:QUANT_SIGNAL_SPREAD_TRENDING='0.006'
$env:QUANT_RANK_THRESHOLD_BAD='0.65'
$env:QUANT_RANK_THRESHOLD_TRENDING='0.68'
$env:QUANT_TURNOVER_COOLDOWN_DAYS='2'
$env:QUANT_MAX_NEW_POSITIONS_PER_DAY='3'
$env:QUANT_REGIME_HIDDEN_SIZE='64'
$env:QUANT_FEATURE_SIGNAL_ENABLED='0'
$env:QUANT_TRADE_ACCEPTANCE_ENABLED='0'
$env:QUANT_REGIME_EXPERTS_ENABLED='0'
$env:QUANT_FLIP_RISK_ENABLED='0'
$env:QUANT_DISABLE_HIGH_VOL_TRADES='1'
$env:QUANT_MIN_DAILY_UNIVERSE_FRACTION='0.80'
$env:QUANT_ROBUSTNESS_MIN_SHARPE_DELTA='-0.55'

# Run backtest
venv/Scripts/python.exe train.py --backtest-only --start 2026-04-08 --end 2026-04-10
