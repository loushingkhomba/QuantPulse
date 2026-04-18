"""
predict_next_day.py
===================
Top-10 prediction versus realized 5-day alpha comparison.

Workflow
--------
1. Load 2-year cached dataset
2. Create features in forward-inference mode (keeps rows where target is NaN)
3. For each ticker, build the last 20-day sequence whose signal date falls
    on or before the requested cutoff and run the trained ensemble
4. Apply temperature calibration + feature-signal blending (same params as
    the last train run)
5. Rank all 10 tickers by blended score
6. Compute realized 5-trading-day stock return, Nifty return, and alpha
7. Print the side-by-side comparison table

Usage
-----
     python predict_next_day.py
     python predict_next_day.py --signal 2026-04-11 --horizon 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--signal",  default="2026-04-11",
                    help="Last trading date used as signal (default: 2026-04-11)")
parser.add_argument("--horizon", type=int, default=5,
                    help="Forward trading-day horizon for realized return/alpha (default: 5)")
parser.add_argument("--top-n", type=int, default=10,
                    help="Number of top picks to highlight (default: 10)")
args = parser.parse_args()

SIGNAL_CUTOFF = pd.Timestamp(args.signal)
HORIZON_DAYS  = int(args.horizon)
TOP_N         = args.top_n

# ---------------------------------------------------------------------------
# Constants (must match last training run)
# ---------------------------------------------------------------------------
ENSEMBLE_SEEDS        = [42, 123, 777]
MODEL_DIR             = "models"
TEMPERATURE           = 3.0            # from calibration in last train run
FEATURE_SIGNAL_SCALE  = 0.85
FEATURE_SIGNAL_W      = {
    "trend": 0.14, "volatility": 0.10, "drawdown": 0.10,
    "rs": 0.08,    "momentum":   0.06, "rsi":      0.03,
}
FEATURE_COLS = [
    "returns",         "ma10_ma50_ratio",  "ma20_ma50_ratio",   "ma50_ma200_ratio",
    "volatility",      "rsi",              "momentum",          "volume_ema",
    "nifty_return",    "distance_from_52w_high", "rs_vs_nifty", "rs_vs_nifty_20d",
    "volume_spike",    "nifty_trend",      "market_volatility", "market_volatility_60d",
    "volatility_regime","nifty_drawdown_63d","nifty_slope_20",  "adx_14",
    "trend_strength",  "regime_state",
]
SEQUENCE_LEN = 20
INPUT_SIZE   = len(FEATURE_COLS)   # 22
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
os.environ["QUANT_FORWARD_INFERENCE"] = "1"
os.environ["QUANT_HISTORY_YEARS"]     = "2"
os.environ["QUANT_USE_DATA_CACHE"]    = "1"

from src.zerodha_ohlc_loader import download_data
from src.features import create_features
from src.model import QuantPulseSimple

print("=" * 70)
print("QuantPulse - 5-Day Alpha Prediction vs Actual")
print(f"Signal cutoff : {SIGNAL_CUTOFF.date()}")
print(f"Horizon days  : {HORIZON_DAYS}")
print("=" * 70)

print("[1/5] Loading 2-year cached data...")
raw = download_data()
raw["Date"] = pd.to_datetime(raw["Date"])
print(f"      {len(raw):,} rows  |  {raw['Date'].min().date()} -> {raw['Date'].max().date()}")

nifty_by_date = (
    raw[["Date", "nifty_close"]]
    .dropna(subset=["nifty_close"])
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
    .set_index("Date")["nifty_close"]
    .astype(float)
)

trading_dates = pd.Index(sorted(raw["Date"].drop_duplicates()))
latest_usable_signal = None
for date_value in trading_dates:
    date_loc = trading_dates.get_loc(date_value)
    if isinstance(date_loc, slice):
        date_loc = date_loc.start
    future_loc = int(date_loc) + HORIZON_DAYS
    if future_loc < len(trading_dates):
        latest_usable_signal = pd.Timestamp(date_value)

if latest_usable_signal is None:
    print("No scorable signal date found for the requested horizon.")
    sys.exit(1)

effective_signal_cutoff = min(SIGNAL_CUTOFF, latest_usable_signal)
if effective_signal_cutoff < SIGNAL_CUTOFF:
    print(
        "Requested signal cutoff cannot be fully scored at the requested horizon; "
        f"using latest scorable signal date: {effective_signal_cutoff.date()}"
    )

print("[2/5] Creating features (forward-inference)...")
feat = create_features(raw)
feat["Date"] = pd.to_datetime(feat["Date"])
print(f"      {len(feat):,} feature rows")

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def apply_temperature(prob, T=3.0):
    return float(_sigmoid(_safe_logit(prob) / T))

def _zscore(series):
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0)
    m, s = arr.mean(), arr.std()
    return ((arr - m) / (s + 1e-6))

# ---------------------------------------------------------------------------
# Load ensemble models
# ---------------------------------------------------------------------------
print("[3/5] Loading ensemble models...")
models = []
for seed in ENSEMBLE_SEEDS:
    path = os.path.join(MODEL_DIR, f"quantpulse_model_seed{seed}.pth")
    if not os.path.exists(path):
        print(f"  !! Model not found: {path}  — run train.py first")
        sys.exit(1)
    m = QuantPulseSimple(input_size=INPUT_SIZE, hidden_size=32, dropout=0.0)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    m.to(DEVICE)
    models.append(m)
print(f"      {len(models)} models loaded")

# ---------------------------------------------------------------------------
# Build inference sequences + run models
# ---------------------------------------------------------------------------
print("[4/5] Running inference for each ticker...\n")

raw_prices = raw[["Date", "Ticker", "Close"]].copy()
raw_prices["Date"] = pd.to_datetime(raw_prices["Date"])

rows = []
tickers_all = feat["Ticker"].unique()

for ticker in sorted(tickers_all):
    tdf = feat[feat["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

    # Find last date <= SIGNAL_CUTOFF
    valid_mask = tdf["Date"] <= effective_signal_cutoff
    if valid_mask.sum() < SEQUENCE_LEN + 1:
        print(f"  [{ticker}] skip - not enough history before {effective_signal_cutoff.date()}")
        continue

    signal_idx  = tdf[valid_mask].index[-1]           # absolute index in tdf
    signal_date = tdf.loc[signal_idx, "Date"]

    # Sequence = the SEQUENCE_LEN rows BEFORE signal_idx (the window used to
    # predict outcomes starting at signal_date)
    if signal_idx < SEQUENCE_LEN:
        print(f"  [{ticker}] skip — not enough rows before signal date")
        continue

    seq_slice = tdf.iloc[signal_idx - SEQUENCE_LEN : signal_idx]

    # Fit scaler on data strictly before signal_date (no look-ahead)
    train_mask  = tdf["Date"] < signal_date
    if train_mask.sum() < 10:
        print(f"  [{ticker}] skip — insufficient training rows for scaler")
        continue

    scaler = StandardScaler()
    scaler.fit(tdf.loc[train_mask, FEATURE_COLS].values)
    seq_scaled = scaler.transform(seq_slice[FEATURE_COLS].values)

    # Ensemble inference
    x_t = torch.tensor(seq_scaled[np.newaxis], dtype=torch.float32).to(DEVICE)
    member_probs = []
    with torch.no_grad():
        for m in models:
            logits = m(x_t)
            p      = F.softmax(logits, dim=-1)[0, 1].item()
            member_probs.append(p)

    raw_prob    = float(np.mean(member_probs))
    calibrated  = apply_temperature(raw_prob, TEMPERATURE)

    # Snapshot raw features at signal_date for blending
    sig_feat = tdf.loc[signal_idx]

    # Price on signal_date from raw data
    price_sig_rows = raw_prices[
        (raw_prices["Ticker"] == ticker) & (raw_prices["Date"] == signal_date)
    ]
    price_sig = float(price_sig_rows["Close"].iloc[0]) if not price_sig_rows.empty else None

    ticker_prices = raw_prices[raw_prices["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    ticker_dates = pd.Index(ticker_prices["Date"])
    signal_loc = ticker_dates.get_loc(signal_date)
    if isinstance(signal_loc, slice):
        signal_loc = signal_loc.start
    future_loc = int(signal_loc) + HORIZON_DAYS

    if future_loc < len(ticker_prices):
        exit_row = ticker_prices.iloc[future_loc]
        exit_date = pd.Timestamp(exit_row["Date"])
        price_exit = float(exit_row["Close"])
        actual_ret_h = (price_exit - price_sig) / price_sig * 100 if price_sig else None

        nifty_start = float(nifty_by_date.loc[signal_date]) if signal_date in nifty_by_date.index else None
        nifty_end = float(nifty_by_date.loc[exit_date]) if exit_date in nifty_by_date.index else None
        nifty_ret_h = ((nifty_end - nifty_start) / nifty_start * 100) if nifty_start not in (None, 0.0) and nifty_end is not None else None
        alpha_h = actual_ret_h - nifty_ret_h if actual_ret_h is not None and nifty_ret_h is not None else None
    else:
        exit_date = None
        price_exit = None
        actual_ret_h = None
        nifty_ret_h = None
        alpha_h = None

    print(
        f"  {ticker:<18}  signal={signal_date.date()}  "
        f"raw_prob={raw_prob*100:.1f}%  calibrated={calibrated*100:.1f}%  "
        f"exit_date={exit_date.date() if exit_date is not None else 'N/A'}  "
        f"ret_{HORIZON_DAYS}d={actual_ret_h:.2f}%  alpha={alpha_h:.2f}%" if actual_ret_h is not None and alpha_h is not None else
        f"  {ticker:<18}  signal={signal_date.date()}  "
        f"raw_prob={raw_prob*100:.1f}%  calibrated={calibrated*100:.1f}%  "
        f"exit_date=N/A  ret_{HORIZON_DAYS}d=N/A"
    )

    rows.append({
        "Ticker":         ticker,
        "signal_date":    signal_date,
        "raw_prob":       raw_prob,
        "calibrated":     calibrated,
        "trend_strength": float(sig_feat.get("trend_strength", 0.0)),
        "volatility_regime": float(sig_feat.get("volatility_regime", 1.0)),
        "nifty_drawdown_63d": float(sig_feat.get("nifty_drawdown_63d", 0.0)),
        "rs_vs_nifty":    float(sig_feat.get("rs_vs_nifty", 0.0)),
        "momentum":       float(sig_feat.get("momentum", 0.0)),
        "rsi":            float(sig_feat.get("rsi", 50.0)),
        "regime_state":   int(sig_feat.get("regime_state", 0)),
        "signal_close":   price_sig,
        "exit_date":      exit_date,
        "exit_close":     price_exit,
        "actual_horizon_pct":  actual_ret_h,
        "nifty_horizon_pct":   nifty_ret_h,
        "alpha_horizon_pct":   alpha_h,
    })

if not rows:
    print("No valid prediction rows generated - check data / model paths.")
    sys.exit(1)

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Feature-signal blending  (cross-sectional z-score, same as train.py)
# ---------------------------------------------------------------------------
base_logit = _safe_logit(df["calibrated"].values)
adj        = np.zeros(len(df))

adj += FEATURE_SIGNAL_W["trend"]      *  _zscore(df["trend_strength"]).values
adj -= FEATURE_SIGNAL_W["volatility"] *  _zscore(df["volatility_regime"]).values
adj += FEATURE_SIGNAL_W["drawdown"]   *  _zscore(df["nifty_drawdown_63d"]).values
adj += FEATURE_SIGNAL_W["rs"]         *  _zscore(df["rs_vs_nifty"]).values
adj += FEATURE_SIGNAL_W["momentum"]   *  _zscore(df["momentum"]).values
rsi_centered = df["rsi"].astype(float) - 50.0
adj += FEATURE_SIGNAL_W["rsi"]        * _zscore(rsi_centered).values

df["blended_score"] = np.clip(_sigmoid(base_logit + FEATURE_SIGNAL_SCALE * adj), 0, 1) * 100

# ---------------------------------------------------------------------------
# Sort and print comparison table
# ---------------------------------------------------------------------------
df = df.sort_values("blended_score", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", range(1, len(df) + 1))
df["Ticker_clean"] = df["Ticker"].str.replace(r"\.NS$", "", regex=True)

print("\n\n" + "=" * 100)
print(
    f"  TOP-{TOP_N} PREDICTION  vs  REALIZED {HORIZON_DAYS}-DAY ALPHA  "
    f"(signal cut-off: {effective_signal_cutoff.date()})"
)
print("=" * 100)
header = (
    f"{'Rk':<3} {'Stock':<14} {'Blended%':>8} {'Calib%':>7} "
    f"{'Regime':>6} {'Signal Close':>12} {'Exit Date':>11} "
    f"{'StockRet':>9} {'NiftyRet':>9} {'Alpha':>8}  Result"
)
print(header)
print("-" * 100)

correct = 0
valid = 0

for _, row in df.iterrows():
    rk        = int(row["Rank"])
    ticker    = row["Ticker_clean"]
    blended   = row["blended_score"]
    calib     = row["calibrated"] * 100
    regime    = {-1: "BAD", 0: "NEUT", 1: "TREND"}.get(row["regime_state"], "?")
    sig_close = f"{row['signal_close']:>10.2f}" if row["signal_close"] else "       N/A"
    exit_dt   = str(row["exit_date"].date()) if row["exit_date"] is not None else "       N/A"

    if row["alpha_horizon_pct"] is not None:
        stock_ret_str = f"{row['actual_horizon_pct']:>+8.2f}%"
        nifty_ret_str = f"{row['nifty_horizon_pct']:>+8.2f}%"
        alpha_str = f"{row['alpha_horizon_pct']:>+7.2f}%"
        predicted_outperform = blended > 50
        outperformed = row["alpha_horizon_pct"] > 0
        valid += 1
        if predicted_outperform == outperformed:
            correct += 1
            marker = "CORRECT [OK]" if outperformed else "CORRECT [OK] (avoided)"
        else:
            marker = "missed  [X]"
    else:
        stock_ret_str = "      N/A"
        nifty_ret_str = "      N/A"
        alpha_str = "     N/A"
        marker  = "-"

    flag = "  *** TOP PICK" if rk <= TOP_N else ""
    print(
        f"{rk:<3} {ticker:<14} {blended:>7.2f}% {calib:>6.2f}% "
        f"{regime:>6} {sig_close} {exit_dt:>11} "
        f"{stock_ret_str} {nifty_ret_str} {alpha_str}  {marker}{flag}"
    )

print("=" * 100)

print(f"\nTotal tickers evaluated : {len(df)}")
if valid > 0:
    acc = correct / valid * 100
    print(f"Alpha sign accuracy     : {correct}/{valid} = {acc:.1f}%  (blended > 50% -> positive alpha)")

top_df  = df[df["Rank"] <= TOP_N]
valid_t = top_df[top_df["alpha_horizon_pct"].notna()]
if len(valid_t) > 0:
    hit = (valid_t["alpha_horizon_pct"] > 0).sum()
    print(f"Top-{TOP_N} positive-alpha hit : {hit}/{len(valid_t)} = {hit/len(valid_t)*100:.1f}%")
    avg_stock_ret = valid_t["actual_horizon_pct"].mean()
    avg_nifty_ret = valid_t["nifty_horizon_pct"].mean()
    avg_alpha = valid_t["alpha_horizon_pct"].mean()
    print(f"Top-{TOP_N} avg stock return   : {avg_stock_ret:+.2f}%")
    print(f"Top-{TOP_N} avg Nifty return   : {avg_nifty_ret:+.2f}%")
    print(f"Top-{TOP_N} avg alpha          : {avg_alpha:+.2f}%")

print("\nNote: Model target = 5-day excess return vs Nifty (> 0.5% alpha over 5 days).")
print("      This report now evaluates the same target family directly.\n")
