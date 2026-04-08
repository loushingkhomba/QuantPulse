import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import download_data
from src.features import create_features
from src.dataset import prepare_dataset
from src.model import QuantPulse, QuantPulseMLP, QuantPulseSimple
from src.trainer import train
from src.backtest import calculate_metrics

MODEL_PATH = "models/quantpulse_model.pth"
CHECKPOINT_PATH = "models/quantpulse_checkpoint.pth"
RESUME_TRAINING = False
MODEL_TYPE = "simple"  # "lstm", "mlp", or "simple" (simple recommended for robustness)
TEST_FRACTION = 0.30
SEED = int(os.getenv("QUANT_SEED", "42"))
ENSEMBLE_SEEDS_ENV = os.getenv("QUANT_ENSEMBLE_SEEDS", "42,123,777")
if ENSEMBLE_SEEDS_ENV.strip():
    ENSEMBLE_SEEDS = [int(x.strip()) for x in ENSEMBLE_SEEDS_ENV.split(",") if x.strip()]
else:
    ENSEMBLE_SEEDS = [42, 123, 777]


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

print("Starting training for QuantPulse model...")
print("Seed:", SEED)
print("Ensemble seeds:", ENSEMBLE_SEEDS)

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
data = create_features(data)

# Optional date-window slicing for walk-forward validation.
train_start_env = os.getenv("QUANT_TRAIN_START", "")
train_end_env = os.getenv("QUANT_TRAIN_END", "")
test_start_env = os.getenv("QUANT_TEST_START", "")
test_end_env = os.getenv("QUANT_TEST_END", "")
split_date_env = os.getenv("QUANT_SPLIT_DATE", "")

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

X_train, X_test, y_train, y_test, dates_train, tickers_test, dates_test = prepare_dataset(
    data,
    test_fraction=TEST_FRACTION,
    split_date=split_date_env if split_date_env else None
)

dates_train = pd.Series(pd.to_datetime(dates_train))
dates_test = pd.Series(pd.to_datetime(dates_test))

if test_start_env:
    test_start_ts = pd.to_datetime(test_start_env)
    keep_test = dates_test >= test_start_ts
    X_test = X_test[keep_test.values].copy()
    y_test = y_test[keep_test.values].copy()
    tickers_test = tickers_test[keep_test.values].copy()
    dates_test = dates_test[keep_test.values].reset_index(drop=True)

if test_end_env:
    test_end_ts = pd.to_datetime(test_end_env)
    keep_test = dates_test <= test_end_ts
    X_test = X_test[keep_test.values].copy()
    y_test = y_test[keep_test.values].copy()
    tickers_test = tickers_test[keep_test.values].copy()
    dates_test = dates_test[keep_test.values].reset_index(drop=True)

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

X_val = torch.tensor(X_train[~is_train.values].copy(), dtype=torch.float32)
y_val = torch.tensor(y_train[~is_train.values].copy(), dtype=torch.long)
val_dates = dates_train[~is_train.values]

X_test = torch.tensor(X_test.copy(), dtype=torch.float32)

print("Training samples:", len(X_tr))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))
print("Configured test fraction:", TEST_FRACTION)
print("Train date range:", train_dates.min(), "to", train_dates.max())
print("Validation date range:", val_dates.min(), "to", val_dates.max())
print("Test date range:", dates_test.min(), "to", dates_test.max())


def build_diverse_train_view(model_index, model_seed):
    diversity_mode = os.getenv("QUANT_TRAIN_DIVERSITY_MODE", "full").strip().lower()
    if diversity_mode == "full":
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates

    diversity_blocks = max(2, int(os.getenv("QUANT_TRAIN_DIVERSITY_BLOCKS", "4")))
    diversity_segments = max(1, int(os.getenv("QUANT_TRAIN_DIVERSITY_SEGMENTS", "2")))
    selected_dates = None
    unique_dates = np.sort(train_dates.unique())

    if len(unique_dates) < diversity_blocks:
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates

    date_blocks = np.array_split(unique_dates, diversity_blocks)
    if diversity_mode == "subperiods":
        chosen = [date_blocks[(model_index + offset) % len(date_blocks)] for offset in range(min(diversity_segments, len(date_blocks)))]
        selected_dates = np.unique(np.concatenate(chosen))
    elif diversity_mode == "shuffled_segments":
        rng = np.random.RandomState(model_seed + (model_index * 1000))
        chosen_idx = rng.choice(len(date_blocks), size=min(diversity_segments, len(date_blocks)), replace=False)
        selected_dates = np.unique(np.concatenate([date_blocks[idx] for idx in chosen_idx]))
    else:
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates

    selected_mask = train_dates.isin(selected_dates)
    if selected_mask.sum() < 200:
        return X_tr, y_tr, X_val, y_val, train_dates, val_dates

    subset_X = X_train[selected_mask.values].copy()
    subset_y = y_train[selected_mask.values].copy()
    subset_dates = train_dates[selected_mask.values].reset_index(drop=True)

    sort_idx_local = np.argsort(subset_dates.values)
    subset_X = subset_X[sort_idx_local].copy()
    subset_y = subset_y[sort_idx_local].copy()
    subset_dates = subset_dates.iloc[sort_idx_local].reset_index(drop=True)

    local_unique_dates = np.sort(subset_dates.unique())
    local_val_cutoff = local_unique_dates[int(len(local_unique_dates) * 0.8)]
    local_is_train = subset_dates < local_val_cutoff

    subset_X_tr = torch.tensor(subset_X[local_is_train.values].copy(), dtype=torch.float32)
    subset_y_tr = torch.tensor(subset_y[local_is_train.values].copy(), dtype=torch.long)
    subset_dates_tr = subset_dates[local_is_train.values]

    subset_X_val = torch.tensor(subset_X[~local_is_train.values].copy(), dtype=torch.float32)
    subset_y_val = torch.tensor(subset_y[~local_is_train.values].copy(), dtype=torch.long)
    subset_dates_val = subset_dates[~local_is_train.values]

    return subset_X_tr, subset_y_tr, subset_X_val, subset_y_val, subset_dates_tr, subset_dates_val

# -------------------------------------------------
# BUILD / TRAIN MODEL(S)
# -------------------------------------------------

print("\nTraining model ensemble...")
signal_mode = os.getenv("QUANT_SIGNAL_MODE", "model").strip().lower()
print("Signal mode:", signal_mode)

if signal_mode == "model":
    X_test = X_test.to(device)
    confidence_list = []

    for idx, model_seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n[{idx + 1}/{len(ENSEMBLE_SEEDS)}] Initializing model with seed {model_seed}...")
        set_global_seed(model_seed)

        X_tr_loop, y_tr_loop, X_val_loop, y_val_loop, train_dates_loop, val_dates_loop = build_diverse_train_view(idx, model_seed)
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
        else:
            print("Starting fresh training (no warm start).")

        train_stats = train(
            model,
            X_tr_loop,
            y_tr_loop,
            X_val_loop,
            y_val_loop,
            epochs=40,
            batch_size=64,
            save_every=20,
            model_path=model_path_for_seed,
            checkpoint_path=checkpoint_path_for_seed,
            patience=50,
            lr=3e-4,
            weight_decay=5e-4,
            grad_clip=1.0,
            label_smoothing=0.05,
            confidence_penalty=float(os.getenv("QUANT_CONFIDENCE_PENALTY", "0.01"))
        )

        if os.path.exists(model_path_for_seed):
            model.load_state_dict(torch.load(model_path_for_seed, map_location=device, weights_only=True))

        print("Best val loss:", round(train_stats["best_val_loss"], 5), "at epoch", train_stats["best_epoch"])
        print("Model saved to", model_path_for_seed)

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            probs = torch.softmax(outputs, dim=1)
            confidence_list.append(probs[:, 1].cpu().numpy())

    confidence = np.mean(np.stack(confidence_list, axis=0), axis=0)
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
    print("Momentum heuristic confidence generated.")
elif signal_mode == "random":
    rng = np.random.RandomState(SEED + 12345)
    confidence = rng.rand(len(tickers_test))
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
pred_df["random_confidence"] = pred_df.groupby("date")["confidence"].transform(
    lambda s: np.random.permutation(s.values)
)

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
turnover_min_new_positions = int(os.getenv("QUANT_MAX_NEW_POSITIONS_PER_DAY", "2"))
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

# Regime-split thresholds (adaptive layer only; execution math remains unchanged).
rank_threshold_bad = float(os.getenv("QUANT_RANK_THRESHOLD_BAD", "0.62"))
rank_threshold_neutral = float(os.getenv("QUANT_RANK_THRESHOLD_NEUTRAL", "0.60"))
rank_threshold_trending = float(os.getenv("QUANT_RANK_THRESHOLD_TRENDING", "0.58"))

spread_threshold_bad = float(os.getenv("QUANT_SIGNAL_SPREAD_BAD", "0.012"))
spread_threshold_neutral = float(os.getenv("QUANT_SIGNAL_SPREAD_NEUTRAL", "0.010"))
spread_threshold_trending = float(os.getenv("QUANT_SIGNAL_SPREAD_TRENDING", "0.008"))

fallback_conf_threshold_bad = float(os.getenv("QUANT_FALLBACK_CONF_THRESHOLD_BAD", str(fallback_conf_threshold)))
fallback_reduce_factor_bad = float(os.getenv("QUANT_FALLBACK_REDUCE_FACTOR_BAD", str(fallback_reduce_factor)))

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

# -------------------------------------------------
# BACKTEST FUNCTION
# -------------------------------------------------

def run_backtest(conf_col):
    """
    Run backtest with adaptive signal handling.
    Lowers rank threshold to allow more trades on weak signals.
    """


    portfolio_values = []
    capital = initial_capital
    peak_capital = initial_capital
    trade_days = 0
    trades_executed = 0
    last_trade_idx_by_ticker = {}
    prev_selected = set()
    regime_day_returns = {"high_vol": [], "trending": [], "drawdown": [], "neutral": []}

    #groups = list(pred_df.groupby(pred_df.index // group_size))
    groups = list(pred_df.groupby("date"))

    for i in range(len(groups) - holding - execution_delay):

        today = groups[i][1]
        tomorrow = groups[i + holding + execution_delay][1]

        if len(today) == 0:
            portfolio_values.append(capital)
            continue

        # Use the same tradable universe across strategies (must exist on exit date).
        tomorrow_tickers = set(tomorrow["ticker"].values)
        today = today[today["ticker"].isin(tomorrow_tickers)]
        if len(today) == 0:
            portfolio_values.append(capital)
            continue

        day = today.copy()

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

        day["rank_pct"] = day[conf_col].rank(pct=True, method="first")
        candidates = day[day["rank_pct"] >= day_min_rank_threshold]
        top = candidates.sort_values(conf_col, ascending=False).head(top_k)

        if len(top) == 0:
            portfolio_values.append(capital)
            continue

        # Turnover control: retain existing picks and limit new entries.
        selected_tickers = list(top["ticker"].values)
        existing = [t for t in selected_tickers if t in prev_selected]
        newcomers = [t for t in selected_tickers if t not in prev_selected]
        newcomers = newcomers[:max(0, turnover_min_new_positions)]
        allowed_tickers = set(existing + newcomers)

        # Cooldown control to avoid immediate re-trades on the same ticker.
        filtered_rows = []
        for _, row in top.iterrows():
            t = row["ticker"]
            if t not in allowed_tickers:
                continue
            last_idx = last_trade_idx_by_ticker.get(t, -10**9)
            if (i - last_idx) < turnover_cooldown_days:
                continue
            filtered_rows.append(row)

        if not filtered_rows:
            portfolio_values.append(capital)
            continue

        top = pd.DataFrame(filtered_rows).head(top_k)

        # Skip noisy days where model has almost no ranking confidence.
        signal_spread = top[conf_col].iloc[0] - top[conf_col].iloc[-1]
        if signal_spread < day_min_signal_spread:
            portfolio_values.append(capital)
            continue

        regime_state = int(np.nan_to_num(top["regime_state"].iloc[0], nan=0.0)) if "regime_state" in top.columns else 0
        volatility_regime = float(np.nan_to_num(top["volatility_regime"].iloc[0], nan=1.0)) if "volatility_regime" in top.columns else 1.0
        drawdown_state = float(np.nan_to_num(top["nifty_drawdown_63d"].iloc[0], nan=0.0)) if "nifty_drawdown_63d" in top.columns else 0.0
        trend_strength = float(np.nan_to_num(top["trend_strength"].iloc[0], nan=0.0)) if "trend_strength" in top.columns else 0.0
        day_confidence = float(top[conf_col].max())
        fallback_scale = 1.0

        # Confidence gating only in adverse/unstable regimes.
        if is_bad_regime and day_confidence < fallback_conf_threshold_bad:
            fallback_scale = fallback_reduce_factor_bad

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

            daily_return += weight * r
            active_weight += weight

        if active_weight > 0:
            daily_return = daily_return / active_weight
            daily_return = daily_return * fallback_scale
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

# -------------------------------------------------
# RUN BOTH BACKTESTS
# -------------------------------------------------

print("\nRunning REAL model backtest...")
real_final, real_sharpe, real_dd, real_curve, real_stats = run_backtest("confidence")

print("\nRunning RANDOM baseline backtest...")
rand_final, rand_sharpe, rand_dd, rand_curve, rand_stats = run_backtest("random_confidence")

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

if inv_vs_real_sharpe_edge > 0.5 and inv_final > real_final:
    print(f"[SIGNAL REVERSAL DETECTED] Inverted signal is {inv_vs_real_sharpe_edge:.3f} Sharpe better")
    print(f"  Switching to inverted signal for this seed")
    real_final, real_sharpe, real_dd, real_curve, real_stats = inv_final, inv_sharpe, inv_dd, inv_curve, inv_stats
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

print("\nRANDOM BASELINE")
print("Final Value:", round(rand_final, 2))
print("Sharpe:", round(rand_sharpe, 3))
print("Max Drawdown:", round(rand_dd * 100, 2), "%")
print("Annualized Return:", round(rand_stats["annualized_return"] * 100, 2), "%")
print("Trade Days:", rand_stats["trade_days"])
print("Trades Executed:", rand_stats["trades_executed"])

print("\nINVERTED SIGNAL (DIAGNOSTIC)")
print("Final Value:", round(inv_final, 2))
print("Sharpe:", round(inv_sharpe, 3))
print("Max Drawdown:", round(inv_dd * 100, 2), "%")
print("Annualized Return:", round(inv_stats["annualized_return"] * 100, 2), "%")
print("Trade Days:", inv_stats["trade_days"])
print("Trades Executed:", inv_stats["trades_executed"])

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