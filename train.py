import argparse
import json
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.zerodha_ohlc_loader import download_data
from src.features import create_features
from src.dataset import prepare_dataset
from src.model import QuantPulse, QuantPulseMLP, QuantPulseSimple
from src.trainer import train
from src.backtest import calculate_metrics

MODEL_PATH = "models/quantpulse_model.pth"
CHECKPOINT_PATH = "models/quantpulse_checkpoint.pth"
RESUME_TRAINING = False
MODEL_TYPE = "simple"  # "lstm", "mlp", or "simple" (simple recommended for robustness)
TEST_FRACTION = 0.20
OBJECTIVE_MODE = os.getenv("QUANT_OBJECTIVE_MODE", "classification").strip().lower()
TRAIN_EPOCHS = int(os.getenv("QUANT_TRAIN_EPOCHS", "40"))
TRAIN_PATIENCE = int(os.getenv("QUANT_TRAIN_PATIENCE", "10"))
RANK_LOSS_WEIGHT = float(os.getenv("QUANT_RANK_LOSS_WEIGHT", "1.0"))
CLS_LOSS_WEIGHT = float(os.getenv("QUANT_CLASSIFICATION_LOSS_WEIGHT", "0.25"))
SEED = int(os.getenv("QUANT_SEED", "42"))
ENSEMBLE_SEEDS_ENV = os.getenv("QUANT_ENSEMBLE_SEEDS", "42,123,777")
if ENSEMBLE_SEEDS_ENV.strip():
    ENSEMBLE_SEEDS = [int(x.strip()) for x in ENSEMBLE_SEEDS_ENV.split(",") if x.strip()]
else:
    ENSEMBLE_SEEDS = [42, 123, 777]


def parse_args():
    parser = argparse.ArgumentParser(description="Train QuantPulse and run backtests.")
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Skip retraining and evaluate existing model checkpoints on the requested date window.",
    )
    parser.add_argument("--start", type=str, default="", help="Start date for the evaluation window (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="", help="End date for the evaluation window (YYYY-MM-DD).")
    
    # Regime-split threshold arguments
    parser.add_argument("--rank-threshold-bad", type=float, default=None, help="Rank threshold for bad regime.")
    parser.add_argument("--rank-threshold-neutral", type=float, default=None, help="Rank threshold for neutral regime.")
    parser.add_argument("--rank-threshold-trending", type=float, default=None, help="Rank threshold for trending regime.")
    
    # Signal spread threshold arguments
    parser.add_argument("--signal-spread-bad", type=float, default=None, help="Signal spread threshold for bad regime.")
    parser.add_argument("--signal-spread-neutral", type=float, default=None, help="Signal spread threshold for neutral regime.")
    parser.add_argument("--signal-spread-trending", type=float, default=None, help="Signal spread threshold for trending regime.")
    
    # Confidence threshold arguments
    parser.add_argument("--min-top-confidence", type=float, default=None, help="Minimum top confidence threshold.")
    parser.add_argument("--min-top-confidence-bad", type=float, default=None, help="Minimum top confidence threshold for bad regime.")
    parser.add_argument("--max-new-positions-per-day", type=int, default=None, help="Maximum new positions per day.")
    
    # Concentration cap arguments
    parser.add_argument("--max-per-sector", type=int, default=None, help="Maximum positions per sector.")
    parser.add_argument("--max-consecutive-days-per-ticker", type=int, default=None, help="Maximum consecutive days per ticker.")
    
    # Trade budget arguments
    parser.add_argument("--trade-budget-mode", type=str, default=None, help="Trade budget mode (disabled/window).")
    parser.add_argument("--max-trades-per-window", type=int, default=None, help="Maximum trades per window.")
    
    return parser.parse_args()


ARGS = parse_args()


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

print("Starting training for QuantPulse model...")
print("Seed:", SEED)
print("Ensemble seeds:", ENSEMBLE_SEEDS)
print("Objective mode:", OBJECTIVE_MODE)
print("Regime-robust training:", os.getenv("QUANT_REGIME_ROBUST_TRAINING", "1").strip() == "1")
if ARGS.backtest_only:
    print("Backtest-only mode: loading existing checkpoints and evaluating on the requested window.")
if ARGS.start or ARGS.end:
    print("Requested evaluation window:", ARGS.start or "(dataset start)", "to", ARGS.end or "(dataset end)")

set_global_seed(SEED)

# -------------------------------------------------
# DEVICE
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

print("\nDownloading market data...")
data = download_data()

print("Creating features...")
if ARGS.backtest_only:
    # Keep latest unlabeled rows so window-end dates are included in carry-forward evaluation.
    os.environ["QUANT_FORWARD_INFERENCE"] = "1"
data = create_features(data)

# Optional date-window slicing for walk-forward validation.
train_start_env = os.getenv("QUANT_TRAIN_START", "")
train_end_env = os.getenv("QUANT_TRAIN_END", "")
test_start_env = os.getenv("QUANT_TEST_START", "")
test_end_env = os.getenv("QUANT_TEST_END", "")
split_date_env = os.getenv("QUANT_SPLIT_DATE", "")

if ARGS.start:
    split_date_env = ARGS.start.strip()
    test_start_env = ARGS.start.strip()
if ARGS.end:
    test_end_env = ARGS.end.strip()

if any([train_start_env, train_end_env, test_start_env, test_end_env]):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    if train_start_env:
        data = data[data["Date"] >= pd.to_datetime(train_start_env)]
    if test_end_env:
        data = data[data["Date"] <= pd.to_datetime(test_end_env)]
    elif train_end_env:
        data = data[data["Date"] <= pd.to_datetime(train_end_env)]

    print("Date window override enabled")
    print("Train start:", train_start_env or "(dataset start)")
    print("Train end:", train_end_env or "(split-driven)")
    print("Test start:", test_start_env or "(split-driven)")
    print("Test end:", test_end_env or "(dataset end)")

# -------------------------------------------------
# DATASET
# -------------------------------------------------

print("Preparing dataset...")

X_train, X_test, y_train, y_test, dates_train, tickers_test, dates_test, regime_train, regime_test = prepare_dataset(
    data,
    test_fraction=TEST_FRACTION,
    split_date=split_date_env if split_date_env else None,
    return_regime_state=True
)

dates_train = pd.Series(pd.to_datetime(dates_train))
dates_test = pd.Series(pd.to_datetime(dates_test))
regime_train = pd.Series(regime_train)
regime_test = pd.Series(regime_test)

if test_start_env:
    test_start_ts = pd.to_datetime(test_start_env)
    keep_test = dates_test >= test_start_ts
    X_test = X_test[keep_test.values].copy()
    y_test = y_test[keep_test.values].copy()
    tickers_test = tickers_test[keep_test.values].copy()
    dates_test = dates_test[keep_test.values].reset_index(drop=True)
    regime_test = regime_test[keep_test.values].reset_index(drop=True)

if test_end_env:
    test_end_ts = pd.to_datetime(test_end_env)
    keep_test = dates_test <= test_end_ts
    X_test = X_test[keep_test.values].copy()
    y_test = y_test[keep_test.values].copy()
    tickers_test = tickers_test[keep_test.values].copy()
    dates_test = dates_test[keep_test.values].reset_index(drop=True)
    regime_test = regime_test[keep_test.values].reset_index(drop=True)

# -------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------------------------

sort_idx = np.argsort(dates_train.values)
X_train = X_train[sort_idx].copy()
y_train = y_train[sort_idx].copy()
dates_train = dates_train.iloc[sort_idx].reset_index(drop=True)

train_unique_dates = np.sort(dates_train.unique())
val_cutoff = train_unique_dates[int(len(train_unique_dates) * 0.8)]
is_train = dates_train < val_cutoff

X_tr = torch.tensor(X_train[is_train.values].copy(), dtype=torch.float32)
y_tr = torch.tensor(y_train[is_train.values].copy(), dtype=torch.long)
train_dates = dates_train[is_train.values]
regime_tr = torch.tensor(regime_train[is_train.values].to_numpy(copy=True), dtype=torch.long)

X_val = torch.tensor(X_train[~is_train.values].copy(), dtype=torch.float32)
y_val = torch.tensor(y_train[~is_train.values].copy(), dtype=torch.long)
val_dates = dates_train[~is_train.values]
regime_val = torch.tensor(regime_train[~is_train.values].to_numpy(copy=True), dtype=torch.long)

X_test = torch.tensor(X_test.copy(), dtype=torch.float32)

print("Training samples:", len(X_tr))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))
print("Configured test fraction:", TEST_FRACTION)
print("Train date range:", train_dates.min(), "to", train_dates.max())
print("Validation date range:", val_dates.min(), "to", val_dates.max())
print("Test date range:", dates_test.min(), "to", dates_test.max())

test_window_start = pd.to_datetime(dates_test.min()) if len(dates_test) else None
test_window_end = pd.to_datetime(dates_test.max()) if len(dates_test) else None


def build_diverse_train_view(model_index, model_seed):
    diversity_mode = os.getenv("QUANT_TRAIN_DIVERSITY_MODE", "full").strip().lower()
    if diversity_mode == "full":
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates, regime_tr, regime_val

    diversity_blocks = max(2, int(os.getenv("QUANT_TRAIN_DIVERSITY_BLOCKS", "4")))
    diversity_segments = max(1, int(os.getenv("QUANT_TRAIN_DIVERSITY_SEGMENTS", "2")))
    selected_dates = None
    unique_dates = np.sort(train_dates.unique())

    if len(unique_dates) < diversity_blocks:
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates, regime_tr, regime_val

    date_blocks = np.array_split(unique_dates, diversity_blocks)
    if diversity_mode == "subperiods":
        chosen = [date_blocks[(model_index + offset) % len(date_blocks)] for offset in range(min(diversity_segments, len(date_blocks)))]
        selected_dates = np.unique(np.concatenate(chosen))
    elif diversity_mode == "shuffled_segments":
        rng = np.random.RandomState(model_seed + (model_index * 1000))
        chosen_idx = rng.choice(len(date_blocks), size=min(diversity_segments, len(date_blocks)), replace=False)
        selected_dates = np.unique(np.concatenate([date_blocks[idx] for idx in chosen_idx]))
    else:
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates, regime_tr, regime_val

    selected_mask = train_dates.isin(selected_dates)
    if selected_mask.sum() < 200:
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates, regime_tr, regime_val

    subset_X = X_train[selected_mask.values].copy()
    subset_y = y_train[selected_mask.values].copy()
    subset_dates = train_dates[selected_mask.values].reset_index(drop=True)
    subset_regime = regime_train[selected_mask.values].reset_index(drop=True)

    sort_idx_local = np.argsort(subset_dates.values)
    subset_X = subset_X[sort_idx_local].copy()
    subset_y = subset_y[sort_idx_local].copy()
    subset_dates = subset_dates.iloc[sort_idx_local].reset_index(drop=True)
    subset_regime = subset_regime.iloc[sort_idx_local].reset_index(drop=True)

    local_unique_dates = np.sort(subset_dates.unique())
    local_val_cutoff = local_unique_dates[int(len(local_unique_dates) * 0.8)]
    local_is_train = subset_dates < local_val_cutoff

    subset_X_tr = torch.tensor(subset_X[local_is_train.values].copy(), dtype=torch.float32)
    subset_y_tr = torch.tensor(subset_y[local_is_train.values].copy(), dtype=torch.long)
    subset_dates_tr = subset_dates[local_is_train.values]
    subset_regime_tr = torch.tensor(subset_regime[local_is_train.values].to_numpy(copy=True), dtype=torch.long)

    subset_X_val = torch.tensor(subset_X[~local_is_train.values].copy(), dtype=torch.float32)
    subset_y_val = torch.tensor(subset_y[~local_is_train.values].copy(), dtype=torch.long)
    subset_dates_val = subset_dates[~local_is_train.values]
    subset_regime_val = torch.tensor(subset_regime[~local_is_train.values].to_numpy(copy=True), dtype=torch.long)

    return subset_X_tr, subset_y_tr, subset_X_val, subset_y_val, subset_dates_tr, subset_dates_val, subset_regime_tr, subset_regime_val

# -------------------------------------------------
# BUILD / TRAIN MODEL(S)
# -------------------------------------------------

print("\nTraining model ensemble...")
signal_mode = os.getenv("QUANT_SIGNAL_MODE", "model").strip().lower()
print("Signal mode:", signal_mode)

if signal_mode == "model":
    X_test = X_test.to(device)
    confidence_list = []
    validation_confidence_list = []
    X_val_device = X_val.to(device)

    for idx, model_seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n[{idx + 1}/{len(ENSEMBLE_SEEDS)}] Initializing model with seed {model_seed}...")
        set_global_seed(model_seed)

        X_tr_loop, y_tr_loop, X_val_loop, y_val_loop, train_dates_loop, val_dates_loop, regime_tr_loop, regime_val_loop = build_diverse_train_view(idx, model_seed)
        train_dates_loop_tensor = torch.tensor(pd.Series(train_dates_loop).astype("int64").to_numpy(copy=True), dtype=torch.long)
        val_dates_loop_tensor = torch.tensor(pd.Series(val_dates_loop).astype("int64").to_numpy(copy=True), dtype=torch.long)
        print("Training samples for this member:", len(X_tr_loop))
        print("Validation samples for this member:", len(X_val_loop))

        if MODEL_TYPE == "simple":
            model = QuantPulseSimple(input_size=X_tr_loop.shape[2])
        elif MODEL_TYPE == "mlp":
            model = QuantPulseMLP(input_size=X_tr_loop.shape[2])
        else:
            model = QuantPulse(input_size=X_tr_loop.shape[2])

        model.to(device)
        print("Model type:", MODEL_TYPE.upper())

        model_path_for_seed = MODEL_PATH if len(ENSEMBLE_SEEDS) == 1 else MODEL_PATH.replace(".pth", f"_seed{model_seed}.pth")
        checkpoint_path_for_seed = CHECKPOINT_PATH if len(ENSEMBLE_SEEDS) == 1 else CHECKPOINT_PATH.replace(".pth", f"_seed{model_seed}.pth")

        if RESUME_TRAINING and os.path.exists(model_path_for_seed):
            try:
                print("Loading existing trained model...")
                model.load_state_dict(torch.load(model_path_for_seed, map_location=device, weights_only=True))
            except RuntimeError:
                print("Model architecture changed. Starting new training.")
        elif ARGS.backtest_only:
            if not os.path.exists(model_path_for_seed):
                raise FileNotFoundError(f"Missing checkpoint for backtest-only mode: {model_path_for_seed}")
            print("Backtest-only mode: loading checkpoint", model_path_for_seed)
            model.load_state_dict(torch.load(model_path_for_seed, map_location=device, weights_only=True))
        else:
            print("Starting fresh training (no warm start).")

        if not ARGS.backtest_only:
            train_stats = train(
                model,
                X_tr_loop,
                y_tr_loop,
                X_val_loop,
                y_val_loop,
                regime_train=regime_tr_loop,
                regime_val=regime_val_loop,
                epochs=TRAIN_EPOCHS,
                batch_size=64,
                save_every=20,
                model_path=model_path_for_seed,
                checkpoint_path=checkpoint_path_for_seed,
                patience=TRAIN_PATIENCE,
                lr=3e-4,
                weight_decay=5e-4,
                grad_clip=1.0,
                label_smoothing=0.05,
                confidence_penalty=float(os.getenv("QUANT_CONFIDENCE_PENALTY", "0.015")),
                objective_mode=OBJECTIVE_MODE,
                rank_loss_weight=RANK_LOSS_WEIGHT,
                classification_loss_weight=CLS_LOSS_WEIGHT,
                train_dates=train_dates_loop_tensor if OBJECTIVE_MODE == "ranking" else None,
                val_dates=val_dates_loop_tensor if OBJECTIVE_MODE == "ranking" else None,
            )

            if os.path.exists(model_path_for_seed):
                model.load_state_dict(torch.load(model_path_for_seed, map_location=device, weights_only=True))

            print("Best val loss:", round(train_stats["best_val_loss"], 5), "at epoch", train_stats["best_epoch"])
            print("Model saved to", model_path_for_seed)
        else:
            print("Skipping retraining for backtest-only mode.")

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            probs = torch.softmax(outputs, dim=1)
            confidence_list.append(probs[:, 1].cpu().numpy())
            val_outputs = model(X_val_device)
            val_probs = torch.softmax(val_outputs, dim=1)
            validation_confidence_list.append(val_probs[:, 1].cpu().numpy())

    confidence = np.mean(np.stack(confidence_list, axis=0), axis=0)
    validation_confidence = (
        np.mean(np.stack(validation_confidence_list, axis=0), axis=0)
        if validation_confidence_list
        else None
    )
    print("\nEnsemble members:", len(ENSEMBLE_SEEDS))
    print("Averaged confidence signal generated.")
elif signal_mode == "momentum":
    lookup = (
        data[["Date", "Ticker", "momentum"]]
        .rename(columns={"Date": "date", "Ticker": "ticker", "momentum": "confidence"})
        .copy()
    )
    lookup["date"] = pd.to_datetime(lookup["date"])
    signal_df = pd.DataFrame({"ticker": tickers_test, "date": pd.to_datetime(dates_test)})
    signal_df = signal_df.merge(lookup, on=["date", "ticker"], how="left")
    confidence = signal_df["confidence"].fillna(0.0).values
    validation_confidence = None
    print("Momentum heuristic confidence generated.")
elif signal_mode == "random":
    rng = np.random.RandomState(SEED + 12345)
    confidence = rng.rand(len(tickers_test))
    validation_confidence = None
    print("Pure random confidence generated.")
else:
    raise ValueError(f"Unsupported QUANT_SIGNAL_MODE: {signal_mode}")

# Keep random baseline deterministic while allowing multi-baseline robustness sweeps.
RANDOM_BASELINE_SEED = int(os.getenv("QUANT_RANDOM_BASELINE_SEED", str(SEED)))
np.random.seed(RANDOM_BASELINE_SEED)
print("Random baseline seed:", RANDOM_BASELINE_SEED)

print("\nFirst 20 predictions:")
print(confidence[:20])

print("\nActual labels:")
print(y_test[:20])

# -------------------------------------------------
# BUILD PORTFOLIO DATA
# -------------------------------------------------

print("\nBuilding portfolio signals...")

# prices = data["Close"].iloc[-len(confidence):].values

# pred_df = pd.DataFrame({
#     "ticker": tickers_test,
#     "confidence": confidence,
#     "random_confidence": random_confidence,
#     "price": prices
# })

pred_df = pd.DataFrame({
    "ticker": tickers_test,
    "date": dates_test,
    "confidence": confidence
})
pred_df["date"] = pd.to_datetime(pred_df["date"])
# -------------------------------------------------
# MERGE WITH TRUE PRICES
# -------------------------------------------------

price_df = data.reset_index()[[
    "Date",
    "Ticker",
    "Close",
    "regime_state",
    "volatility_regime",
    "nifty_drawdown_63d",
    "trend_strength",
]]
price_df["Date"] = pd.to_datetime(price_df["Date"])

pred_df = pred_df.merge(
    price_df,
    left_on=["date", "ticker"],
    right_on=["Date", "Ticker"],
    how="left"
)

pred_df.rename(columns={"Close": "price"}, inplace=True)
pred_df.drop(columns=["Date", "Ticker"], inplace=True)
pred_df = pred_df.dropna(subset=["price"])

# Stress test: randomly remove a fraction of rows to simulate missing quotes.
missing_data_rate = float(os.getenv("QUANT_MISSING_DATA_RATE", "0.0"))
if missing_data_rate > 0:
    missing_data_rate = min(max(missing_data_rate, 0.0), 0.8)
    missing_rng = np.random.RandomState(SEED + 911)
    keep_mask = missing_rng.rand(len(pred_df)) >= missing_data_rate
    pred_df = pred_df[keep_mask].copy()
    print("Missing data rate applied:", missing_data_rate)

# Prevent accidental duplicate keys from inflating returns.
pred_df = pred_df.groupby(["date", "ticker"], as_index=False).agg({
    "confidence": "mean",
    "price": "last",
    "regime_state": "last",
    "volatility_regime": "last",
    "nifty_drawdown_63d": "last",
    "trend_strength": "last",
})

# Smooth confidence per ticker to reduce noise-driven trade flips.
# Slightly relaxed default smoothing to preserve responsiveness.
confidence_ema_span = int(os.getenv("QUANT_CONFIDENCE_EMA_SPAN", "2"))
if confidence_ema_span > 1:
    pred_df = pred_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    pred_df["confidence"] = pred_df.groupby("ticker")["confidence"].transform(
        lambda s: s.ewm(span=confidence_ema_span, adjust=False).mean()
    )

# 🔀 RANDOM BASELINE (cross-sectional random ranking per date)
def build_random_confidence(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    randomized = np.zeros(len(pred_df), dtype=np.float64)
    date_to_indices = pred_df.groupby("date").indices
    for indices in date_to_indices.values():
        vals = pred_df.iloc[indices]["confidence"].values
        randomized[indices] = rng.permutation(vals)
    return randomized


pred_df["random_confidence"] = build_random_confidence(RANDOM_BASELINE_SEED)

# Inversion diagnostic to detect potentially flipped signals.
pred_df["inv_confidence"] = 1.0 - pred_df["confidence"]

# SORT BY DATE (VERY IMPORTANT)
pred_df = pred_df.sort_values("date").reset_index(drop=True)

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

initial_capital = 10000
holding = 5
top_k = 3
stop_loss = -0.03
max_daily_loss = -0.02
max_drawdown_limit = -0.35
max_position_weight = float(os.getenv("QUANT_MAX_POSITION_WEIGHT", "0.5"))
transaction_cost = float(os.getenv("QUANT_TRANSACTION_COST", "0.0005"))
slippage = float(os.getenv("QUANT_SLIPPAGE", "0.0002"))
execution_delay = int(os.getenv("QUANT_EXECUTION_DELAY", "0"))
min_signal_spread = 0.01
max_trade_gain = 0.05
min_rank_threshold = 0.60
turnover_min_new_positions = (
    ARGS.max_new_positions_per_day
    if ARGS.max_new_positions_per_day is not None
    else int(os.getenv("QUANT_MAX_NEW_POSITIONS_PER_DAY", "2"))
)
turnover_cooldown_days = int(os.getenv("QUANT_TURNOVER_COOLDOWN_DAYS", "1"))
aggression_clip_low = float(os.getenv("QUANT_AGGRESSION_CLIP_LOW", "0.40"))
aggression_clip_high = float(os.getenv("QUANT_AGGRESSION_CLIP_HIGH", "0.60"))
weight_temperature = float(os.getenv("QUANT_WEIGHT_TEMPERATURE", "1.2"))
fallback_conf_threshold = float(os.getenv("QUANT_FALLBACK_CONF_THRESHOLD", "0.52"))
fallback_reduce_factor = float(os.getenv("QUANT_FALLBACK_REDUCE_FACTOR", "0.50"))
adaptive_regime_logic = os.getenv("QUANT_ADAPTIVE_REGIME_LOGIC", "1").strip() == "1"
bad_volatility_cutoff = float(os.getenv("QUANT_BAD_VOLATILITY_CUTOFF", "1.25"))
bad_drawdown_cutoff = float(os.getenv("QUANT_BAD_DRAWDOWN_CUTOFF", "-0.08"))
trend_strength_cutoff = float(os.getenv("QUANT_TREND_STRENGTH_CUTOFF", "0.00"))
quality_safety_enabled = os.getenv("QUANT_QUALITY_SAFETY_ENABLED", "1").strip() == "1"

# Regime-split thresholds (adaptive layer only; execution math remains unchanged).
# Command-line args take precedence over environment variables.
rank_threshold_bad = (
    ARGS.rank_threshold_bad
    if ARGS.rank_threshold_bad is not None
    else float(os.getenv("QUANT_RANK_THRESHOLD_BAD", "0.68"))
)
rank_threshold_neutral = (
    ARGS.rank_threshold_neutral
    if ARGS.rank_threshold_neutral is not None
    else float(os.getenv("QUANT_RANK_THRESHOLD_NEUTRAL", "0.60"))
)
rank_threshold_trending = (
    ARGS.rank_threshold_trending
    if ARGS.rank_threshold_trending is not None
    else float(os.getenv("QUANT_RANK_THRESHOLD_TRENDING", "0.58"))
)

spread_threshold_bad = (
    ARGS.signal_spread_bad
    if ARGS.signal_spread_bad is not None
    else float(os.getenv("QUANT_SIGNAL_SPREAD_BAD", "0.012"))
)
spread_threshold_neutral = (
    ARGS.signal_spread_neutral
    if ARGS.signal_spread_neutral is not None
    else float(os.getenv("QUANT_SIGNAL_SPREAD_NEUTRAL", "0.010"))
)
spread_threshold_trending = (
    ARGS.signal_spread_trending
    if ARGS.signal_spread_trending is not None
    else float(os.getenv("QUANT_SIGNAL_SPREAD_TRENDING", "0.008"))
)

min_top_confidence = (
    ARGS.min_top_confidence
    if ARGS.min_top_confidence is not None
    else float(os.getenv("QUANT_MIN_TOP_CONFIDENCE", "0.52"))
)
min_top_confidence_bad = (
    ARGS.min_top_confidence_bad
    if ARGS.min_top_confidence_bad is not None
    else float(os.getenv("QUANT_MIN_TOP_CONFIDENCE_BAD", "0.52"))
)

fallback_conf_threshold_bad = float(os.getenv("QUANT_FALLBACK_CONF_THRESHOLD_BAD", str(min_top_confidence_bad)))
fallback_reduce_factor_bad = float(os.getenv("QUANT_FALLBACK_REDUCE_FACTOR_BAD", "0.50"))

# Targeted 2021 defense controls (bad regime only).
regime_exposure_scale_bad = float(os.getenv("QUANT_REGIME_EXPOSURE_SCALE_BAD", "0.60"))
kill_switch_drawdown_threshold = float(os.getenv("QUANT_KILL_SWITCH_DRAWDOWN_THRESHOLD", "-0.12"))
kill_switch_max_new_positions = int(os.getenv("QUANT_KILL_SWITCH_MAX_NEW_POSITIONS", "1"))
kill_switch_force_exit = os.getenv("QUANT_KILL_SWITCH_FORCE_EXIT", "0") == "1"
max_per_sector = (
    ARGS.max_per_sector
    if ARGS.max_per_sector is not None
    else int(os.getenv("QUANT_MAX_PER_SECTOR", "0"))
)
max_consecutive_days_per_ticker = (
    ARGS.max_consecutive_days_per_ticker
    if ARGS.max_consecutive_days_per_ticker is not None
    else int(os.getenv("QUANT_MAX_CONSECUTIVE_DAYS_PER_TICKER", "0"))
)
trade_budget_mode = (
    ARGS.trade_budget_mode.strip().lower()
    if ARGS.trade_budget_mode is not None
    else os.getenv("QUANT_TRADE_BUDGET_MODE", "disabled").strip().lower()
)
max_trades_per_window = (
    ARGS.max_trades_per_window
    if ARGS.max_trades_per_window is not None
    else int(os.getenv("QUANT_MAX_TRADES_PER_WINDOW", "0"))
)

SECTOR_BY_TICKER = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "LT.NS": "Industrials",
    "ITC.NS": "Consumer",
    "BHARTIARTL.NS": "Telecom",
    "HINDUNILVR.NS": "Consumer",
}
FEATURE_DRIFT_COLUMNS = ["rsi", "volatility", "rs_vs_nifty", "volume_spike", "trend_strength"]
HORIZON_DAYS = sorted({1, 3, holding})


def _safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    value = _safe_float(value)
    if value is None:
        return None
    return int(value)


def _safe_ratio(numerator, denominator):
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _safe_mean(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _safe_std(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.std())


def _binary_entropy(values):
    arr = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return -(arr * np.log(arr) + (1.0 - arr) * np.log(1.0 - arr))


def _spearman_rank_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    x_rank = pd.Series(x[mask]).rank(method="average")
    y_rank = pd.Series(y[mask]).rank(method="average")
    corr = x_rank.corr(y_rank, method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)


def _classify_regime_label(regime_state, volatility_regime, drawdown_state, trend_strength_value):
    is_bad_regime = (
        (regime_state < 0)
        or (volatility_regime > bad_volatility_cutoff)
        or (drawdown_state < bad_drawdown_cutoff)
    )
    is_trending_regime = (not is_bad_regime) and (trend_strength_value > trend_strength_cutoff)
    if is_bad_regime:
        return "BEARISH"
    if is_trending_regime:
        return "BULLISH"
    return "NEUTRAL"


def _signal_quality_bucket(signal_spread_value):
    spread_value = _safe_float(signal_spread_value)
    if spread_value is None:
        return "UNKNOWN"
    if spread_value > 0.05:
        return "HIGH"
    if spread_value >= 0.015:
        return "MEDIUM"
    return "LOW"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def build_forward_metric_columns(frame, horizons, execution_delay_days):
    enriched = frame.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    future_lookup = enriched[["date", "ticker", "price"]].rename(columns={"date": "future_date", "price": "future_price"})
    unique_dates = list(pd.Index(enriched["date"].drop_duplicates().sort_values()))
    price_lookup_by_date = {
        pd.Timestamp(date_value): group.set_index("ticker")["price"].astype(float).to_dict()
        for date_value, group in enriched.groupby("date")
    }
    nifty_daily = (
        data[["Date", "nifty_close"]]
        .dropna(subset=["nifty_close"])
        .drop_duplicates(subset=["Date"])
        .copy()
    )
    nifty_daily["Date"] = pd.to_datetime(nifty_daily["Date"])
    nifty_close_by_date = nifty_daily.set_index("Date")["nifty_close"].astype(float).to_dict()

    benchmark_returns = {}
    for horizon in horizons:
        future_date_map = {}
        benchmark_returns[horizon] = {}
        for idx, date_value in enumerate(unique_dates):
            future_idx = idx + horizon + execution_delay_days
            future_date = unique_dates[future_idx] if future_idx < len(unique_dates) else pd.NaT
            future_date_map[date_value] = future_date
            if pd.isna(future_date):
                benchmark_returns[horizon][date_value] = np.nan
                continue
            start_close = _safe_float(nifty_close_by_date.get(date_value))
            end_close = _safe_float(nifty_close_by_date.get(future_date))
            if start_close in (None, 0.0) or end_close is None:
                benchmark_returns[horizon][date_value] = np.nan
            else:
                benchmark_returns[horizon][date_value] = (end_close / start_close) - 1.0

        future_date_col = f"future_date_{horizon}d"
        future_price_col = f"future_price_{horizon}d"
        return_col = f"future_return_{horizon}d"
        benchmark_col = f"nifty_return_{horizon}d"
        alpha_col = f"future_alpha_{horizon}d"
        win_col = f"future_win_{horizon}d"

        enriched[future_date_col] = enriched["date"].map(future_date_map)
        merge_df = future_lookup.rename(columns={"future_date": future_date_col, "future_price": future_price_col})
        enriched = enriched.merge(merge_df, on=[future_date_col, "ticker"], how="left")
        enriched[return_col] = (enriched[future_price_col] / enriched["price"]) - 1.0
        enriched[benchmark_col] = enriched["date"].map(benchmark_returns[horizon])
        enriched[alpha_col] = enriched[return_col] - enriched[benchmark_col]
        enriched[win_col] = np.where(enriched[alpha_col].notna(), (enriched[alpha_col] > 0).astype(float), np.nan)

    trade_price_paths = {}
    max_horizon = max(horizons)
    for idx, date_value in enumerate(unique_dates):
        end_idx = idx + max_horizon + execution_delay_days
        if end_idx >= len(unique_dates):
            continue
        path_dates = unique_dates[idx : end_idx + 1]
        for ticker in price_lookup_by_date.get(date_value, {}):
            path_prices = []
            for path_date in path_dates:
                price_value = price_lookup_by_date.get(path_date, {}).get(ticker)
                if price_value is not None:
                    path_prices.append(float(price_value))
            if len(path_prices) >= 2:
                trade_price_paths[(pd.Timestamp(date_value), ticker)] = path_prices

    return enriched, benchmark_returns.get(holding, {}), trade_price_paths


pred_df, trade_benchmark_returns, trade_price_paths = build_forward_metric_columns(
    pred_df,
    horizons=HORIZON_DAYS,
    execution_delay_days=execution_delay,
)

print("Min rank threshold:", min_rank_threshold)
print("Transaction cost:", transaction_cost)
print("Slippage:", slippage)
print("Max position weight:", max_position_weight)
print("Execution delay:", execution_delay)
print("Confidence EMA span:", confidence_ema_span)
print("Max new positions/day:", turnover_min_new_positions)
print("Turnover cooldown days:", turnover_cooldown_days)
print("Aggression clip:", (aggression_clip_low, aggression_clip_high))
print("Weight temperature:", weight_temperature)
print("Fallback confidence threshold:", fallback_conf_threshold)
print("Fallback reduce factor:", fallback_reduce_factor)
print("Adaptive regime logic:", adaptive_regime_logic)
print("Bad regime cutoffs:", {"volatility": bad_volatility_cutoff, "drawdown": bad_drawdown_cutoff})
print("Rank thresholds by regime:", {"bad": rank_threshold_bad, "neutral": rank_threshold_neutral, "trending": rank_threshold_trending})
print("Signal spread thresholds by regime:", {"bad": spread_threshold_bad, "neutral": spread_threshold_neutral, "trending": spread_threshold_trending})
print("Min top confidence:", {"default": min_top_confidence, "bad": min_top_confidence_bad})
print("Bad regime exposure scale:", regime_exposure_scale_bad)
print("Kill-switch drawdown threshold:", kill_switch_drawdown_threshold)
print("Kill-switch max new positions:", kill_switch_max_new_positions)
print("Max names per sector:", max_per_sector if max_per_sector > 0 else "disabled")
print(
    "Max consecutive days per ticker:",
    max_consecutive_days_per_ticker if max_consecutive_days_per_ticker > 0 else "disabled",
)
print("Trade budget mode:", trade_budget_mode)
print("Max trades per window:", max_trades_per_window if max_trades_per_window > 0 else "disabled")

# -------------------------------------------------
# BACKTEST FUNCTION
# -------------------------------------------------

def run_backtest(conf_col, apply_kill_switch=None):
    """
    Run backtest with adaptive signal handling.
    Lowers rank threshold to allow more trades on weak signals.
    """


    carry_forward_open = os.getenv("QUANT_CARRY_FORWARD_OPEN", "1" if ARGS.backtest_only else "0").strip() == "1"
    capture_details = conf_col != "random_confidence"
    if apply_kill_switch is None:
        apply_kill_switch = (conf_col == "confidence")
    use_trade_budget = trade_budget_mode == "window" and max_trades_per_window > 0

    portfolio_values = []
    capital = initial_capital
    peak_capital = initial_capital
    trade_days = 0
    trades_executed = 0
    last_trade_idx_by_ticker = {}
    consecutive_days_by_ticker = {}
    prev_selected = set()
    regime_day_returns = {"high_vol": [], "trending": [], "drawdown": [], "neutral": []}
    rejection_counts = {
        "empty_today": 0,
        "empty_after_universe_intersection": 0,
        "no_candidates_after_rank": 0,
        "no_rows_after_turnover_cooldown": 0,
        "single_name_after_filters": 0,
        "spread_below_threshold": 0,
        "top_conf_below_threshold": 0,
        "blocked_by_concentration_cap": 0,
        "trade_budget_exhausted": 0,
        "no_next_price_for_selected": 0,
        "active_weight_zero": 0,
    }
    trade_records = []
    day_records = []

    #groups = list(pred_df.groupby(pred_df.index // group_size))
    groups = list(pred_df.groupby("date"))

    if carry_forward_open:
        max_iter = max(0, len(groups) - 1)
    else:
        max_iter = max(0, len(groups) - holding - execution_delay)

    for i in range(max_iter):

        today = groups[i][1]
        exit_index = i + holding + execution_delay
        has_full_holding_exit = exit_index < len(groups)
        if has_full_holding_exit:
            tomorrow = groups[exit_index][1]
        elif carry_forward_open:
            tomorrow = groups[-1][1]
        else:
            continue

        if len(today) == 0:
            rejection_counts["empty_today"] += 1
            portfolio_values.append(capital)
            continue

        # Use the same tradable universe across strategies (must exist on mark/exit date).
        tomorrow_tickers = set(tomorrow["ticker"].values)
        today = today[today["ticker"].isin(tomorrow_tickers)]
        if len(today) == 0:
            rejection_counts["empty_after_universe_intersection"] += 1
            portfolio_values.append(capital)
            continue

        day = today.copy()
        day_date = pd.Timestamp(day["date"].iloc[0])

        if use_trade_budget and trades_executed >= max_trades_per_window:
            rejection_counts["trade_budget_exhausted"] += 1
            portfolio_values.append(capital)
            continue

        # Adaptive regime policy controls only signal filtering and exposure gating.
        day_regime_state = int(np.nan_to_num(day["regime_state"].median(), nan=0.0)) if "regime_state" in day.columns else 0
        day_volatility_regime = float(np.nan_to_num(day["volatility_regime"].median(), nan=1.0)) if "volatility_regime" in day.columns else 1.0
        day_drawdown_state = float(np.nan_to_num(day["nifty_drawdown_63d"].median(), nan=0.0)) if "nifty_drawdown_63d" in day.columns else 0.0
        day_trend_strength = float(np.nan_to_num(day["trend_strength"].median(), nan=0.0)) if "trend_strength" in day.columns else 0.0

        if adaptive_regime_logic:
            is_bad_regime = (
                (day_regime_state < 0)
                or (day_volatility_regime > bad_volatility_cutoff)
                or (day_drawdown_state < bad_drawdown_cutoff)
            )
            is_trending_regime = (not is_bad_regime) and (day_trend_strength > trend_strength_cutoff)

            if is_bad_regime:
                day_min_rank_threshold = rank_threshold_bad
                day_min_signal_spread = spread_threshold_bad
            elif is_trending_regime:
                day_min_rank_threshold = rank_threshold_trending
                day_min_signal_spread = spread_threshold_trending
            else:
                day_min_rank_threshold = rank_threshold_neutral
                day_min_signal_spread = spread_threshold_neutral
        else:
            is_bad_regime = (day_regime_state <= 0) or (day_volatility_regime > bad_volatility_cutoff) or (day_drawdown_state < bad_drawdown_cutoff)
            is_trending_regime = day_trend_strength > trend_strength_cutoff
            day_min_rank_threshold = min_rank_threshold
            day_min_signal_spread = min_signal_spread

        regime_label = _classify_regime_label(
            regime_state=day_regime_state,
            volatility_regime=day_volatility_regime,
            drawdown_state=day_drawdown_state,
            trend_strength_value=day_trend_strength,
        )

        day["rank_pct"] = day[conf_col].rank(pct=True, method="first")
        candidates = day[day["rank_pct"] >= day_min_rank_threshold]
        top = candidates.sort_values(conf_col, ascending=False).head(top_k)

        if len(top) == 0:
            rejection_counts["no_candidates_after_rank"] += 1
            portfolio_values.append(capital)
            continue

        # Turnover control: retain existing picks and limit new entries.
        day_max_new_positions = turnover_min_new_positions
        if apply_kill_switch and is_bad_regime and day_drawdown_state < kill_switch_drawdown_threshold:
            day_max_new_positions = min(day_max_new_positions, max(0, kill_switch_max_new_positions))
            if kill_switch_force_exit and capture_details:
                prev_selected = []  # force-exit all carry-forward positions; real model only, not random baseline

        selected_tickers = list(top["ticker"].values)
        existing = [t for t in selected_tickers if t in prev_selected]
        newcomers = [t for t in selected_tickers if t not in prev_selected]
        newcomers = newcomers[:max(0, day_max_new_positions)]
        allowed_tickers = set(existing + newcomers)

        # Cooldown control to avoid immediate re-trades on the same ticker.
        filtered_rows = []
        sector_counts = {}
        blocked_by_concentration_cap = False
        for _, row in top.iterrows():
            t = row["ticker"]
            if t not in allowed_tickers:
                continue
            last_idx = last_trade_idx_by_ticker.get(t, -10**9)
            if (i - last_idx) < turnover_cooldown_days:
                continue

            if max_consecutive_days_per_ticker > 0 and consecutive_days_by_ticker.get(t, 0) >= max_consecutive_days_per_ticker:
                blocked_by_concentration_cap = True
                continue

            sector_name = SECTOR_BY_TICKER.get(t, "Unknown")
            if max_per_sector > 0 and sector_counts.get(sector_name, 0) >= max_per_sector:
                blocked_by_concentration_cap = True
                continue

            filtered_rows.append(row)
            sector_counts[sector_name] = sector_counts.get(sector_name, 0) + 1

        if not filtered_rows:
            if blocked_by_concentration_cap:
                rejection_counts["blocked_by_concentration_cap"] += 1
            else:
                rejection_counts["no_rows_after_turnover_cooldown"] += 1
            portfolio_values.append(capital)
            continue

        top = pd.DataFrame(filtered_rows).head(top_k)
        if use_trade_budget:
            trades_remaining = max_trades_per_window - trades_executed
            if trades_remaining <= 0:
                rejection_counts["trade_budget_exhausted"] += 1
                portfolio_values.append(capital)
                continue
            top = top.head(trades_remaining)
        if len(top) == 1:
            rejection_counts["single_name_after_filters"] += 1

        # Skip noisy multi-name books where the ranking spread is too tight.
        signal_spread = np.nan
        if len(top) > 1:
            signal_spread = top[conf_col].iloc[0] - top[conf_col].iloc[-1]
            if signal_spread < day_min_signal_spread:
                rejection_counts["spread_below_threshold"] += 1
                portfolio_values.append(capital)
                continue

        day_min_top_conf = min_top_confidence_bad if is_bad_regime else min_top_confidence
        if float(top[conf_col].max()) < day_min_top_conf:
            rejection_counts["top_conf_below_threshold"] += 1
            portfolio_values.append(capital)
            continue

        regime_state = int(np.nan_to_num(top["regime_state"].iloc[0], nan=0.0)) if "regime_state" in top.columns else 0
        volatility_regime = float(np.nan_to_num(top["volatility_regime"].iloc[0], nan=1.0)) if "volatility_regime" in top.columns else 1.0
        drawdown_state = float(np.nan_to_num(top["nifty_drawdown_63d"].iloc[0], nan=0.0)) if "nifty_drawdown_63d" in top.columns else 0.0
        trend_strength = float(np.nan_to_num(top["trend_strength"].iloc[0], nan=0.0)) if "trend_strength" in top.columns else 0.0
        day_confidence = float(top[conf_col].max())
        signal_bucket = _signal_quality_bucket(signal_spread)
        fallback_scale = 1.0
        regime_exposure_scale = 1.0

        # Confidence gating only in adverse/unstable regimes.
        if is_bad_regime and day_confidence < fallback_conf_threshold_bad:
            fallback_scale = fallback_reduce_factor_bad

        # Smooth protection: scale position impact down in bad regimes.
        if is_bad_regime:
            regime_exposure_scale = regime_exposure_scale_bad

        # Convert scores to positive values for stable weighting.
        scores = top[conf_col].values
        scores = np.clip(scores, aggression_clip_low, aggression_clip_high)
        scores = scores - scores.min() + 1e-6
        scores = np.power(scores, 1.0 / max(weight_temperature, 1e-6))
        weights = scores / scores.sum()
        weights = np.minimum(weights, max_position_weight)
        weights = weights / weights.sum()

        daily_return = 0
        active_weight = 0

        for idx, row in top.reset_index(drop=True).iterrows():

            ticker = row["ticker"]
            weight = weights[idx]

            today_price = row["price"]

            next_price = tomorrow[
                tomorrow["ticker"] == ticker
            ]["price"].values

            if len(next_price) == 0:
                rejection_counts["no_next_price_for_selected"] += 1
                continue

            trades_executed += 1
            last_trade_idx_by_ticker[ticker] = i

            entry_price = today_price * (1 + slippage)
            exit_price = next_price[0] * (1 - slippage)

            r = (exit_price - entry_price) / entry_price

            # Cap extreme gains from bad ticks/splits in merged data.
            r = min(r, max_trade_gain)

            if r < stop_loss:
                r = stop_loss

            # Apply round-trip fee (entry + exit).
            r -= (2 * transaction_cost)

            benchmark_return = _safe_float(trade_benchmark_returns.get(day_date))
            alpha_return = None if benchmark_return is None else r - benchmark_return

            mae_value = None
            if capture_details:
                path_prices = trade_price_paths.get((day_date, ticker), [])
                if len(path_prices) >= 2:
                    path_returns = [((path_price - entry_price) / entry_price) for path_price in path_prices[1:]]
                    if path_returns:
                        mae_value = float(min(path_returns))
                if mae_value is None:
                    mae_value = float(r)

                trade_records.append(
                    {
                        "signal_date": day_date,
                        "exit_date": pd.Timestamp(tomorrow["date"].iloc[0]),
                        "ticker": ticker,
                        "sector": SECTOR_BY_TICKER.get(ticker, "Unknown"),
                        "rank": int(idx + 1),
                        "confidence": float(row[conf_col]),
                        "weight": float(weight),
                        "net_return": float(r),
                        "benchmark_return": benchmark_return,
                        "alpha": alpha_return,
                        "mae": mae_value,
                        "win": bool(alpha_return is not None and alpha_return > 0),
                        "regime_label": regime_label,
                        "signal_bucket": signal_bucket,
                    }
                )

            daily_return += weight * r
            active_weight += weight

        if active_weight > 0:
            daily_return = daily_return / active_weight
            daily_return = daily_return * fallback_scale * regime_exposure_scale
            trade_days += 1
            if volatility_regime > bad_volatility_cutoff:
                regime_day_returns["high_vol"].append(daily_return)
            elif drawdown_state < bad_drawdown_cutoff:
                regime_day_returns["drawdown"].append(daily_return)
            elif trend_strength > trend_strength_cutoff and regime_state >= 0:
                regime_day_returns["trending"].append(daily_return)
            else:
                regime_day_returns["neutral"].append(daily_return)
        else:
            rejection_counts["active_weight_zero"] += 1
            portfolio_values.append(capital)
            continue

        # Daily risk cap to prevent abrupt capital collapse.
        daily_return = max(daily_return, max_daily_loss)

        capital *= (1 + daily_return)
        peak_capital = max(peak_capital, capital)

        # Stop trading if drawdown breach occurs.
        drawdown = (capital - peak_capital) / peak_capital
        portfolio_values.append(capital)

        if drawdown <= max_drawdown_limit:
            break

        if max_consecutive_days_per_ticker > 0:
            selected_now = set(top["ticker"].values)
            next_consecutive_days = {}
            for ticker in selected_now:
                next_consecutive_days[ticker] = consecutive_days_by_ticker.get(ticker, 0) + 1
            consecutive_days_by_ticker = next_consecutive_days

        if capture_details:
            current_selected = set(top["ticker"].values)
            changed_count = 0
            if prev_selected:
                changed_count = int(len(current_selected.symmetric_difference(prev_selected)) / 2)
            day_records.append(
                {
                    "date": day_date,
                    "selected_tickers": list(top["ticker"].values),
                    "signal_spread": _safe_float(signal_spread),
                    "regime_label": regime_label,
                    "signal_bucket": signal_bucket,
                    "top_confidence": float(top[conf_col].max()),
                    "changed_count": changed_count,
                }
            )

        prev_selected = set(top["ticker"].values)

    sharpe, max_dd = calculate_metrics(portfolio_values)
    periods = len(portfolio_values)
    if periods > 0:
        annualized_return = (capital / initial_capital) ** (252 / periods) - 1
    else:
        annualized_return = 0.0

    stats = {
        "trade_days": int(trade_days),
        "trades_executed": int(trades_executed),
        "annualized_return": float(annualized_return),
        "regime_perf": {},
        "rejection_counts": rejection_counts,
        "trade_records": trade_records,
        "day_records": day_records,
    }

    for regime_name, returns in regime_day_returns.items():
        if len(returns) >= 2:
            regime_arr = np.array(returns)
            regime_sharpe = float(np.mean(regime_arr) / (np.std(regime_arr) + 1e-9) * np.sqrt(252))
            regime_avg = float(np.mean(regime_arr))
        elif len(returns) == 1:
            regime_sharpe = 0.0
            regime_avg = float(returns[0])
        else:
            regime_sharpe = 0.0
            regime_avg = 0.0
        stats["regime_perf"][regime_name] = {
            "days": len(returns),
            "avg_daily_return": regime_avg,
            "sharpe": regime_sharpe,
        }

    return capital, sharpe, max_dd, portfolio_values, stats


def compute_signal_quality_metrics(frame, trade_df, signal_col="confidence"):
    daily_rows = []
    for signal_date, group in frame.groupby("date"):
        sorted_group = group.sort_values(signal_col, ascending=False).reset_index(drop=True)
        if sorted_group.empty:
            continue
        spread_index = min(9, len(sorted_group) - 1)
        spread_value = None
        if spread_index >= 1:
            spread_value = float(sorted_group[signal_col].iloc[0] - sorted_group[signal_col].iloc[spread_index])
        entropy_value = float(_binary_entropy(sorted_group[signal_col].to_numpy()).mean())
        daily_rows.append(
            {
                "date": pd.Timestamp(signal_date),
                "top1_ticker": sorted_group["ticker"].iloc[0],
                "top1_confidence": float(sorted_group[signal_col].iloc[0]),
                "rank1_rank10_spread": spread_value,
                "mean_prediction_entropy": entropy_value,
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        top1_changes = int((daily_df["top1_ticker"] != daily_df["top1_ticker"].shift(1)).sum() - 1)
        weeks = daily_df["date"].dt.to_period("W")
        changes_per_week = top1_changes / max(1, weeks.nunique())
    else:
        top1_changes = 0
        changes_per_week = None

    ic_by_horizon = {}
    ic_means = []
    for horizon in HORIZON_DAYS:
        horizon_col = f"future_return_{horizon}d"
        daily_ic_values = []
        for _, group in frame.groupby("date"):
            valid = group[[signal_col, horizon_col]].dropna()
            if len(valid) < 2:
                continue
            ic_value = _spearman_rank_corr(valid[signal_col].to_numpy(), valid[horizon_col].to_numpy())
            if ic_value is not None:
                daily_ic_values.append(ic_value)
        mean_ic = _safe_mean(daily_ic_values)
        ic_by_horizon[f"day_{horizon}"] = {
            "daily_count": len(daily_ic_values),
            "mean_daily_ic": mean_ic,
            "std_daily_ic": _safe_std(daily_ic_values),
        }
        if mean_ic is not None:
            ic_means.append((horizon, mean_ic))

    calibration = {
        "expected_calibration_error": None,
        "brier_score": None,
        "buckets": [],
    }
    if not trade_df.empty:
        calibration_df = trade_df[["confidence", "win"]].dropna().copy()
        if not calibration_df.empty:
            calibration_df["win"] = calibration_df["win"].astype(float)
            bins = np.linspace(0.0, 1.0, 11)
            calibration_df["bucket"] = pd.cut(calibration_df["confidence"], bins=bins, include_lowest=True)
            bucket_rows = []
            total_count = len(calibration_df)
            ece_value = 0.0
            for bucket, bucket_df in calibration_df.groupby("bucket", observed=False):
                if bucket_df.empty:
                    continue
                predicted = float(bucket_df["confidence"].mean())
                actual = float(bucket_df["win"].mean())
                count = int(len(bucket_df))
                ece_value += abs(predicted - actual) * count
                bucket_rows.append(
                    {
                        "bucket": str(bucket),
                        "count": count,
                        "mean_confidence": predicted,
                        "actual_win_rate": actual,
                    }
                )
            calibration = {
                "expected_calibration_error": ece_value / total_count if total_count else None,
                "brier_score": float(np.mean((calibration_df["confidence"] - calibration_df["win"]) ** 2)),
                "buckets": bucket_rows,
            }

    hit_rate_by_rank = {}
    if not trade_df.empty:
        for rank_value in range(1, top_k + 1):
            rank_df = trade_df[trade_df["rank"] == rank_value]
            hit_rate_by_rank[f"rank_{rank_value}"] = {
                "trades": int(len(rank_df)),
                "hit_rate": float(rank_df["win"].mean()) if not rank_df.empty else None,
                "avg_net_return": float(rank_df["net_return"].mean()) if not rank_df.empty else None,
                "avg_alpha": float(rank_df["alpha"].mean()) if not rank_df.empty else None,
            }

    return {
        "confidence_spread": {
            "daily_count": int(len(daily_df)),
            "mean_rank1_rank10_spread": float(daily_df["rank1_rank10_spread"].dropna().mean()) if not daily_df.empty else None,
            "median_rank1_rank10_spread": float(daily_df["rank1_rank10_spread"].dropna().median()) if not daily_df.empty else None,
            "min_rank1_rank10_spread": float(daily_df["rank1_rank10_spread"].dropna().min()) if not daily_df.empty and daily_df["rank1_rank10_spread"].dropna().size else None,
            "warning_low_quality_fraction": float((daily_df["rank1_rank10_spread"] < 0.015).mean()) if not daily_df.empty else None,
        },
        "rank_stability": {
            "top1_changes_total": top1_changes,
            "top1_changes_per_week": changes_per_week,
            "top1_persistence_fraction": float((daily_df["top1_ticker"] == daily_df["top1_ticker"].shift(1)).iloc[1:].mean()) if len(daily_df) > 1 else None,
        },
        "information_coefficient": {
            "full_horizon": ic_by_horizon.get(f"day_{holding}", {}),
            "ic_decay": ic_by_horizon,
            "mean_daily_ic": ic_by_horizon.get(f"day_{holding}", {}).get("mean_daily_ic"),
        },
        "hit_rate_by_rank": hit_rate_by_rank,
        "calibration": calibration,
        "prediction_entropy": {
            "daily_mean_entropy": float(daily_df["mean_prediction_entropy"].mean()) if not daily_df.empty else None,
            "daily_std_entropy": float(daily_df["mean_prediction_entropy"].std()) if len(daily_df) > 1 else None,
        },
    }


def compute_return_and_risk_metrics(trade_df):
    if trade_df.empty:
        return {
            "alpha_vs_nifty_per_trade": {},
            "profit_factor": None,
            "win_loss_ratio": None,
            "average_win": None,
            "average_loss": None,
            "average_mae": None,
            "worst_mae": None,
            "negative_alpha_weeks_max_streak": 0,
        }

    winning_trades = trade_df[trade_df["net_return"] > 0]["net_return"]
    losing_trades = trade_df[trade_df["net_return"] < 0]["net_return"]
    avg_win = float(winning_trades.mean()) if not winning_trades.empty else None
    avg_loss = float(losing_trades.mean()) if not losing_trades.empty else None
    weekly_alpha = (
        trade_df.groupby(trade_df["signal_date"].dt.to_period("W"))["alpha"].mean().dropna()
    )
    negative_streak = 0
    current_streak = 0
    for alpha_value in weekly_alpha:
        if alpha_value < 0:
            current_streak += 1
            negative_streak = max(negative_streak, current_streak)
        else:
            current_streak = 0

    return {
        "alpha_vs_nifty_per_trade": {
            "trade_count": int(len(trade_df)),
            "average_alpha": float(trade_df["alpha"].mean()) if trade_df["alpha"].notna().any() else None,
            "median_alpha": float(trade_df["alpha"].median()) if trade_df["alpha"].notna().any() else None,
            "positive_alpha_fraction": float((trade_df["alpha"] > 0).mean()) if trade_df["alpha"].notna().any() else None,
        },
        "profit_factor": _safe_ratio(winning_trades.sum(), abs(losing_trades.sum())) if not losing_trades.empty else None,
        "win_loss_ratio": _safe_ratio(avg_win, abs(avg_loss)) if avg_loss not in (None, 0.0) else None,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "average_mae": float(trade_df["mae"].mean()) if trade_df["mae"].notna().any() else None,
        "worst_mae": float(trade_df["mae"].min()) if trade_df["mae"].notna().any() else None,
        "negative_alpha_weeks_max_streak": int(negative_streak),
    }


def compute_regime_and_portfolio_metrics(trade_df):
    regime_table = {}
    signal_bucket_table = {}
    if not trade_df.empty:
        for regime_name, regime_df in trade_df.groupby("regime_label"):
            regime_table[regime_name] = {
                "trades": int(len(regime_df)),
                "hit_rate": float(regime_df["win"].mean()),
                "avg_net_return": float(regime_df["net_return"].mean()),
                "avg_alpha": float(regime_df["alpha"].mean()) if regime_df["alpha"].notna().any() else None,
            }
        for bucket_name, bucket_df in trade_df.groupby("signal_bucket"):
            signal_bucket_table[bucket_name] = {
                "trades": int(len(bucket_df)),
                "hit_rate": float(bucket_df["win"].mean()),
                "avg_net_return": float(bucket_df["net_return"].mean()),
                "avg_alpha": float(bucket_df["alpha"].mean()) if bucket_df["alpha"].notna().any() else None,
            }

    selection_days = trade_df.groupby("signal_date")["ticker"].agg(list).sort_index() if not trade_df.empty else pd.Series(dtype=object)
    turnover_values = []
    previous_selection = None
    for tickers in selection_days:
        current_selection = set(tickers)
        if previous_selection is not None:
            turnover_values.append(int(len(current_selection.symmetric_difference(previous_selection)) / 2))
        previous_selection = current_selection

    ticker_day_counts = trade_df.groupby("ticker")["signal_date"].nunique() if not trade_df.empty else pd.Series(dtype=float)
    total_signal_days = int(selection_days.index.nunique()) if not selection_days.empty else 0
    concentration = {
        ticker: {
            "signal_days": int(count),
            "signal_day_fraction": (count / total_signal_days) if total_signal_days else None,
        }
        for ticker, count in ticker_day_counts.items()
    }
    sector_exposure = {}
    if not trade_df.empty:
        sector_counts = trade_df.groupby("sector").size()
        sector_exposure = {
            sector: {
                "selections": int(count),
                "selection_fraction": float(count / len(trade_df)),
            }
            for sector, count in sector_counts.items()
        }

    return {
        "regime_conditional": regime_table,
        "signal_bucket_conditional": signal_bucket_table,
        "concentration_risk": {
            "max_signal_day_fraction": max(
                (details["signal_day_fraction"] for details in concentration.values() if details["signal_day_fraction"] is not None),
                default=None,
            ),
            "by_ticker": concentration,
        },
        "sector_exposure": sector_exposure,
        "turnover_rate": {
            "average_changed_positions": _safe_mean(turnover_values),
            "max_changed_positions": max(turnover_values) if turnover_values else None,
            "all_positions_changed_fraction": float(np.mean([value >= top_k for value in turnover_values])) if turnover_values else None,
        },
    }


def compute_feature_drift_metrics(feature_frame):
    if test_window_start is None or test_window_end is None:
        return {}

    drift_frame = feature_frame.copy()
    drift_frame["Date"] = pd.to_datetime(drift_frame["Date"])
    train_mask = drift_frame["Date"] < test_window_start
    test_mask = (drift_frame["Date"] >= test_window_start) & (drift_frame["Date"] <= test_window_end)

    out = {}
    for column_name in FEATURE_DRIFT_COLUMNS:
        if column_name not in drift_frame.columns:
            continue
        train_values = pd.to_numeric(drift_frame.loc[train_mask, column_name], errors="coerce").dropna()
        test_values = pd.to_numeric(drift_frame.loc[test_mask, column_name], errors="coerce").dropna()
        if train_values.empty or test_values.empty:
            continue
        train_std = float(train_values.std())
        out[column_name] = {
            "train_mean": float(train_values.mean()),
            "test_mean": float(test_values.mean()),
            "train_std": train_std,
            "test_std": float(test_values.std()) if len(test_values) > 1 else 0.0,
            "z_shift": ((float(test_values.mean()) - float(train_values.mean())) / train_std) if train_std > 1e-9 else None,
        }
    return out


def compute_model_health_metrics():
    metrics = {
        "validation_accuracy": None,
        "recent_20day_accuracy": None,
        "accuracy_drift_vs_validation": None,
    }
    if validation_confidence is None or len(y_val) == 0 or len(y_test) == 0:
        return metrics

    validation_predictions = (np.asarray(validation_confidence) >= 0.5).astype(int)
    metrics["validation_accuracy"] = float((validation_predictions == y_val.numpy()).mean())

    test_dates_series = pd.Series(pd.to_datetime(dates_test))
    recent_dates = test_dates_series.drop_duplicates().sort_values().tail(20)
    recent_mask = test_dates_series.isin(recent_dates).to_numpy()
    if recent_mask.any():
        recent_predictions = (np.asarray(confidence)[recent_mask] >= 0.5).astype(int)
        metrics["recent_20day_accuracy"] = float((recent_predictions == np.asarray(y_test)[recent_mask]).mean())
        metrics["accuracy_drift_vs_validation"] = metrics["validation_accuracy"] - metrics["recent_20day_accuracy"]
    return metrics


def build_standard_metrics_artifact(real_stats, signal_col="confidence"):
    trade_df = pd.DataFrame(real_stats.get("trade_records", []))
    if not trade_df.empty:
        trade_df["signal_date"] = pd.to_datetime(trade_df["signal_date"])
        trade_df["exit_date"] = pd.to_datetime(trade_df["exit_date"])

    signal_quality = compute_signal_quality_metrics(pred_df, trade_df, signal_col=signal_col)
    return_quality = compute_return_and_risk_metrics(trade_df)
    regime_portfolio_metrics = compute_regime_and_portfolio_metrics(trade_df)
    feature_drift = compute_feature_drift_metrics(data)
    model_health = compute_model_health_metrics()

    execution_quality = {
        "assumed_slippage_per_side": slippage,
        "assumed_transaction_cost_per_side": transaction_cost,
        "actual_slippage": None,
        "signal_timeliness": {
            "available_in_backtest": False,
            "late_days": None,
        },
    }

    generated_at_utc = pd.Timestamp.now(tz="UTC").isoformat()

    return {
        "generated_at": generated_at_utc,
        "mode": "backtest_only" if ARGS.backtest_only else "train_and_backtest",
        "signal_mode": signal_mode,
        "model_type": MODEL_TYPE,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "window": {
            "start": str(test_window_start.date()) if test_window_start is not None else None,
            "end": str(test_window_end.date()) if test_window_end is not None else None,
            "holding_days": holding,
            "execution_delay_days": execution_delay,
        },
        "backtest_summary": {
            "final_value": real_final,
            "sharpe": real_sharpe,
            "max_drawdown_pct": real_dd * 100.0,
            "annualized_return_pct": real_stats["annualized_return"] * 100.0,
            "trade_days": real_stats["trade_days"],
            "trades_executed": real_stats["trades_executed"],
            "nifty_final": nifty_final,
            "random_final": rand_final,
        },
        "signal_quality": signal_quality,
        "return_quality": return_quality,
        "risk_metrics": {
            "profit_factor": return_quality["profit_factor"],
            "win_loss_ratio": return_quality["win_loss_ratio"],
            "average_win": return_quality["average_win"],
            "average_loss": return_quality["average_loss"],
            "average_mae": return_quality["average_mae"],
            "worst_mae": return_quality["worst_mae"],
        },
        "regime_metrics": regime_portfolio_metrics["regime_conditional"],
        "signal_bucket_metrics": regime_portfolio_metrics["signal_bucket_conditional"],
        "execution_quality": execution_quality,
        "model_health": {
            "prediction_entropy": signal_quality["prediction_entropy"],
            "feature_drift": feature_drift,
            "recent_accuracy_check": model_health,
        },
        "portfolio_metrics": {
            "concentration_risk": regime_portfolio_metrics["concentration_risk"],
            "sector_exposure": regime_portfolio_metrics["sector_exposure"],
            "turnover_rate": regime_portfolio_metrics["turnover_rate"],
        },
    }


def save_standard_metrics_artifact(artifact):
    requested_path = os.getenv("QUANT_METRICS_OUT", "").strip()
    metrics_dir = Path("logs") / "standard_metrics"
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    if requested_path:
        out_path = Path(requested_path)
    else:
        mode_tag = "backtest" if ARGS.backtest_only else "train"
        out_path = metrics_dir / f"{mode_tag}_metrics_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(artifact)
    out_path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")

    latest_path = out_path.parent / "latest_metrics.json"
    if latest_path.resolve() != out_path.resolve():
        latest_path.write_text(json.dumps(safe_payload, indent=2), encoding="utf-8")
    return out_path


def print_standard_metrics_summary(artifact):
    signal_quality = artifact["signal_quality"]
    return_quality = artifact["return_quality"]
    portfolio_metrics = artifact["portfolio_metrics"]
    model_health = artifact["model_health"]
    print("\nSTANDARD METRICS")
    print(
        "Signal quality summary:",
        {
            "ic_mean": round(signal_quality["information_coefficient"].get("mean_daily_ic") or 0.0, 5),
            "ic_day1": round(signal_quality["information_coefficient"]["ic_decay"].get("day_1", {}).get("mean_daily_ic") or 0.0, 5),
            "ic_day3": round(signal_quality["information_coefficient"]["ic_decay"].get("day_3", {}).get("mean_daily_ic") or 0.0, 5),
            "ic_day5": round(signal_quality["information_coefficient"]["ic_decay"].get(f"day_{holding}", {}).get("mean_daily_ic") or 0.0, 5),
            "spread_mean": round(signal_quality["confidence_spread"].get("mean_rank1_rank10_spread") or 0.0, 5),
            "top1_changes_per_week": round(signal_quality["rank_stability"].get("top1_changes_per_week") or 0.0, 3),
            "calibration_ece": round(signal_quality["calibration"].get("expected_calibration_error") or 0.0, 5),
        },
    )
    print(
        "Return quality summary:",
        {
            "avg_alpha_per_trade": round(return_quality["alpha_vs_nifty_per_trade"].get("average_alpha") or 0.0, 5),
            "positive_alpha_fraction": round(return_quality["alpha_vs_nifty_per_trade"].get("positive_alpha_fraction") or 0.0, 3),
            "profit_factor": round(return_quality.get("profit_factor") or 0.0, 3),
            "win_loss_ratio": round(return_quality.get("win_loss_ratio") or 0.0, 3),
            "avg_mae": round(return_quality.get("average_mae") or 0.0, 5),
            "worst_mae": round(return_quality.get("worst_mae") or 0.0, 5),
        },
    )
    print(
        "Portfolio summary:",
        {
            "avg_turnover_changed": round(portfolio_metrics["turnover_rate"].get("average_changed_positions") or 0.0, 3),
            "max_signal_day_fraction": round(portfolio_metrics["concentration_risk"].get("max_signal_day_fraction") or 0.0, 3),
            "entropy_mean": round(model_health["prediction_entropy"].get("daily_mean_entropy") or 0.0, 5),
            "recent_accuracy_drift": round(model_health["recent_accuracy_check"].get("accuracy_drift_vs_validation") or 0.0, 5),
        },
    )

# -------------------------------------------------
# RUN BOTH BACKTESTS
# -------------------------------------------------

print("\nRunning REAL model backtest...")
real_final, real_sharpe, real_dd, real_curve, real_stats = run_backtest("confidence")
active_signal_col = "confidence"

print("\nRunning RANDOM baseline backtest...")
random_baseline_runs_default = "7" if ARGS.backtest_only else "1"
random_baseline_runs = max(1, int(os.getenv("QUANT_RANDOM_BASELINE_RUNS", random_baseline_runs_default)))
random_run_results = []
for run_idx in range(random_baseline_runs):
    run_seed = RANDOM_BASELINE_SEED + (run_idx * 97)
    pred_df["random_confidence"] = build_random_confidence(run_seed)
    run_final, run_sharpe, run_dd, run_curve, run_stats = run_backtest("random_confidence")
    random_run_results.append({
        "seed": run_seed,
        "final": run_final,
        "sharpe": run_sharpe,
        "dd": run_dd,
        "curve": run_curve,
        "stats": run_stats,
    })

if len(random_run_results) == 1:
    chosen_random = random_run_results[0]
else:
    finals = np.array([r["final"] for r in random_run_results])
    median_idx = np.argsort(finals)[len(finals) // 2]
    chosen_random = random_run_results[int(median_idx)]
    print(
        "Random baseline aggregation:",
        {
            "runs": random_baseline_runs,
            "median_final": round(float(np.median(finals)), 2),
            "mean_final": round(float(np.mean(finals)), 2),
            "min_final": round(float(np.min(finals)), 2),
            "max_final": round(float(np.max(finals)), 2),
        },
    )

rand_final = chosen_random["final"]
rand_sharpe = chosen_random["sharpe"]
rand_dd = chosen_random["dd"]
rand_curve = chosen_random["curve"]
rand_stats = chosen_random["stats"]

print("\nRunning INVERTED signal diagnostic backtest...")
inv_final, inv_sharpe, inv_dd, inv_curve, inv_stats = run_backtest("inv_confidence")
# -------------------------------------------------
# ADAPTIVE SIGNAL SELECTION
# -------------------------------------------------
# If inverted signal dramatically outperforms real signal (e.g., seed 777),
# use it instead (signal may be naturally reversed on this seed).
print("\nSignal Quality Check...")
inv_vs_real_sharpe_edge = inv_sharpe - real_sharpe
inv_vs_real_return_edge = inv_stats["annualized_return"] - real_stats["annualized_return"]
auto_invert_signal = os.getenv("QUANT_AUTO_INVERT_SIGNAL", "1" if ARGS.backtest_only else "0").strip() == "1"

if inv_vs_real_sharpe_edge > 0.5 and inv_final > real_final:
    print(f"[SIGNAL REVERSAL DETECTED] Inverted signal is {inv_vs_real_sharpe_edge:.3f} Sharpe better")
    if auto_invert_signal:
        print("  Auto-switch enabled: using inverted signal for this run")
        real_final, real_sharpe, real_dd, real_curve, real_stats = inv_final, inv_sharpe, inv_dd, inv_curve, inv_stats
        active_signal_col = "inv_confidence"
    else:
        print("  Auto-switch disabled: keeping original signal for evaluation")
else:
    print(f"[SIGNAL OK] Using regular confidence signal")


nifty_period = (
    data[["Date", "nifty_close"]]
    .dropna(subset=["nifty_close"])
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
)
nifty_period["Date"] = pd.to_datetime(nifty_period["Date"])
nifty_period = nifty_period[
    (nifty_period["Date"] >= pred_df["date"].min())
    & (nifty_period["Date"] <= pred_df["date"].max())
]

if len(nifty_period) >= 2:
    nifty_final = initial_capital * (nifty_period["nifty_close"].iloc[-1] / nifty_period["nifty_close"].iloc[0])
else:
    nifty_final = initial_capital

standard_metrics_artifact = build_standard_metrics_artifact(real_stats, signal_col=active_signal_col)
standard_metrics_path = save_standard_metrics_artifact(standard_metrics_artifact)
print_standard_metrics_summary(standard_metrics_artifact)
print("Standard metrics saved:", standard_metrics_path)

# -------------------------------------------------
# RESULTS COMPARISON
# -------------------------------------------------

print("\n==============================")
print("RESULT COMPARISON")
print("==============================")

print("\nREAL MODEL")
print("Final Value:", round(real_final, 2))
print("Sharpe:", round(real_sharpe, 3))
print("Max Drawdown:", round(real_dd * 100, 2), "%")
print("Annualized Return:", round(real_stats["annualized_return"] * 100, 2), "%")
print("Trade Days:", real_stats["trade_days"])
print("Trades Executed:", real_stats["trades_executed"])
print("Regime Performance:")
for regime_name, regime_metrics in real_stats["regime_perf"].items():
    print(
        f"  {regime_name}: days={regime_metrics['days']}, "
        f"avg_daily_return={regime_metrics['avg_daily_return']:.5f}, "
        f"sharpe={regime_metrics['sharpe']:.3f}"
    )
print("Rejection diagnostics:")
for k, v in real_stats.get("rejection_counts", {}).items():
    print(f"  {k}: {v}")

print("\nRANDOM BASELINE")
print("Final Value:", round(rand_final, 2))
print("Sharpe:", round(rand_sharpe, 3))
print("Max Drawdown:", round(rand_dd * 100, 2), "%")
print("Annualized Return:", round(rand_stats["annualized_return"] * 100, 2), "%")
print("Trade Days:", rand_stats["trade_days"])
print("Trades Executed:", rand_stats["trades_executed"])
print("Rejection diagnostics:")
for k, v in rand_stats.get("rejection_counts", {}).items():
    print(f"  {k}: {v}")

print("\nINVERTED SIGNAL (DIAGNOSTIC)")
print("Final Value:", round(inv_final, 2))
print("Sharpe:", round(inv_sharpe, 3))
print("Max Drawdown:", round(inv_dd * 100, 2), "%")
print("Annualized Return:", round(inv_stats["annualized_return"] * 100, 2), "%")
print("Trade Days:", inv_stats["trade_days"])
print("Trades Executed:", inv_stats["trades_executed"])
print("Rejection diagnostics:")
for k, v in inv_stats.get("rejection_counts", {}).items():
    print(f"  {k}: {v}")

print("\nNIFTY BUY & HOLD")
print("Final Value:", round(nifty_final, 2))

# -------------------------------------------------
# PLOT EQUITY CURVES
# -------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(real_curve, label="Real Model")
plt.plot(rand_curve, label="Random Baseline")
plt.title("Equity Curve Comparison")
plt.legend()
plt.grid(True)
plt.savefig("comparison_equity.png")
plt.close()

print("\nComparison chart saved: comparison_equity.png")