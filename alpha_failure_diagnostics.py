import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from src.zerodha_ohlc_loader import download_data
from src.features import create_features
from src.model import QuantPulseSimple

# Keep settings aligned with the current production setup.
os.environ["QUANT_FORWARD_INFERENCE"] = "1"
os.environ["QUANT_HISTORY_YEARS"] = "2"
os.environ["QUANT_USE_DATA_CACHE"] = "1"

ENSEMBLE_SEEDS = [42, 123, 777]
MODEL_DIR = Path("models")
SEQUENCE_LEN = 20
TOP_N = 10
HORIZON = 5

FEATURE_COLS = [
    "returns", "ma10_ma50_ratio", "ma20_ma50_ratio", "ma50_ma200_ratio",
    "volatility", "rsi", "momentum", "volume_ema",
    "nifty_return", "distance_from_52w_high", "rs_vs_nifty", "rs_vs_nifty_20d",
    "volume_spike", "nifty_trend", "market_volatility", "market_volatility_60d",
    "volatility_regime", "nifty_drawdown_63d", "nifty_slope_20", "adx_14",
    "trend_strength", "regime_state",
]

FEATURE_SIGNAL_SCALE = 0.85
FEATURE_SIGNAL_W = {
    "trend": 0.14,
    "volatility": 0.10,
    "drawdown": 0.10,
    "rs": 0.08,
    "momentum": 0.06,
    "rsi": 0.03,
}
TEMPERATURE = 3.0


def _safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    m = s.mean()
    std = s.std()
    return (s - m) / (std + 1e-6)


def apply_temperature(prob, temperature):
    return float(_sigmoid(_safe_logit(prob) / max(temperature, 1e-6)))


print("Loading cached data and features...")
raw = download_data().copy()
raw["Date"] = pd.to_datetime(raw["Date"])
feat = create_features(raw)
feat["Date"] = pd.to_datetime(feat["Date"])

raw_prices = raw[["Date", "Ticker", "Close", "nifty_close"]].copy()
raw_prices["Date"] = pd.to_datetime(raw_prices["Date"])

nifty_by_date = (
    raw[["Date", "nifty_close"]]
    .dropna(subset=["nifty_close"])
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
    .set_index("Date")["nifty_close"]
    .astype(float)
)

all_dates = sorted(raw["Date"].drop_duplicates())
scorable_dates = []
for d in all_dates:
    idx = all_dates.index(d)
    if idx + HORIZON < len(all_dates):
        scorable_dates.append(pd.Timestamp(d))

print(f"Scorable signal dates: {len(scorable_dates)}")

# Load ensemble
print("Loading ensemble checkpoints...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = []
for seed in ENSEMBLE_SEEDS:
    p = MODEL_DIR / f"quantpulse_model_seed{seed}.pth"
    if not p.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {p}")
    model = QuantPulseSimple(input_size=len(FEATURE_COLS), hidden_size=32, dropout=0.0)
    model.load_state_dict(torch.load(str(p), map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    models.append(model)

rows = []
all_top_rows = []

for signal_date in scorable_dates:
    exit_date = all_dates[all_dates.index(signal_date) + HORIZON]

    day_rows = []
    for ticker in sorted(feat["Ticker"].unique()):
        tdf = feat[feat["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        mask = tdf["Date"] <= signal_date
        if mask.sum() < SEQUENCE_LEN + 1:
            continue

        signal_idx = tdf[mask].index[-1]
        if signal_idx < SEQUENCE_LEN:
            continue

        signal_date_eff = pd.Timestamp(tdf.loc[signal_idx, "Date"])
        if signal_date_eff != signal_date:
            continue

        seq = tdf.iloc[signal_idx - SEQUENCE_LEN : signal_idx]
        train_mask = tdf["Date"] < signal_date_eff
        if train_mask.sum() < 10:
            continue

        scaler = StandardScaler()
        scaler.fit(tdf.loc[train_mask, FEATURE_COLS].values)
        seq_scaled = scaler.transform(seq[FEATURE_COLS].values)

        x = torch.tensor(seq_scaled[np.newaxis], dtype=torch.float32).to(device)
        probs = []
        with torch.no_grad():
            for m in models:
                logits = m(x)
                probs.append(F.softmax(logits, dim=-1)[0, 1].item())

        raw_prob = float(np.mean(probs))
        calibrated = apply_temperature(raw_prob, TEMPERATURE)

        row_sig = raw_prices[(raw_prices["Ticker"] == ticker) & (raw_prices["Date"] == signal_date_eff)]
        row_exit = raw_prices[(raw_prices["Ticker"] == ticker) & (raw_prices["Date"] == exit_date)]
        if row_sig.empty or row_exit.empty:
            continue

        p0 = float(row_sig["Close"].iloc[0])
        p1 = float(row_exit["Close"].iloc[0])
        stock_ret = (p1 / p0) - 1.0

        n0 = float(nifty_by_date.loc[signal_date_eff])
        n1 = float(nifty_by_date.loc[exit_date])
        nifty_ret = (n1 / n0) - 1.0
        alpha = stock_ret - nifty_ret

        sig_feat = tdf.loc[signal_idx]

        day_rows.append(
            {
                "signal_date": signal_date_eff,
                "exit_date": exit_date,
                "ticker": ticker,
                "raw_prob": raw_prob,
                "calibrated": calibrated,
                "trend_strength": float(sig_feat.get("trend_strength", 0.0)),
                "volatility_regime": float(sig_feat.get("volatility_regime", 1.0)),
                "nifty_drawdown_63d": float(sig_feat.get("nifty_drawdown_63d", 0.0)),
                "rs_vs_nifty": float(sig_feat.get("rs_vs_nifty", 0.0)),
                "momentum": float(sig_feat.get("momentum", 0.0)),
                "rsi": float(sig_feat.get("rsi", 50.0)),
                "regime_state": int(sig_feat.get("regime_state", 0)),
                "stock_ret": stock_ret,
                "nifty_ret": nifty_ret,
                "alpha": alpha,
                "alpha_up": int(alpha > 0),
            }
        )

    if len(day_rows) < 8:
        continue

    df = pd.DataFrame(day_rows)
    base_logit = _safe_logit(df["calibrated"].values)
    adj = np.zeros(len(df), dtype=np.float64)
    adj += FEATURE_SIGNAL_W["trend"] * _zscore(df["trend_strength"]).to_numpy()
    adj -= FEATURE_SIGNAL_W["volatility"] * _zscore(df["volatility_regime"]).to_numpy()
    adj += FEATURE_SIGNAL_W["drawdown"] * _zscore(df["nifty_drawdown_63d"]).to_numpy()
    adj += FEATURE_SIGNAL_W["rs"] * _zscore(df["rs_vs_nifty"]).to_numpy()
    adj += FEATURE_SIGNAL_W["momentum"] * _zscore(df["momentum"]).to_numpy()
    adj += FEATURE_SIGNAL_W["rsi"] * _zscore(df["rsi"] - 50.0).to_numpy()
    df["blended"] = _sigmoid(base_logit + FEATURE_SIGNAL_SCALE * adj)

    df = df.sort_values("blended", ascending=False).reset_index(drop=True)
    top = df.head(TOP_N).copy()

    top_export = top.copy()
    top_export["signal_date"] = pd.Timestamp(signal_date)
    top_export["exit_date"] = pd.Timestamp(exit_date)
    all_top_rows.append(top_export)

    # Metrics aligned with your reporting.
    pred_up = (top["blended"] > 0.5).astype(int)
    sign_acc = float((pred_up == top["alpha_up"]).mean())
    hit_rate = float((top["alpha"] > 0).mean())
    avg_alpha = float(top["alpha"].mean())

    # Ranking quality diagnostics for the day.
    try:
        rank_ic = float(pd.Series(top["blended"]).rank().corr(pd.Series(top["alpha"]).rank(), method="pearson"))
    except Exception:
        rank_ic = np.nan

    # Separation quality: top-half alpha minus bottom-half alpha within top10.
    first_half = top.head(5)["alpha"].mean()
    second_half = top.tail(5)["alpha"].mean()
    separation = float(first_half - second_half)

    rows.append(
        {
            "signal_date": pd.Timestamp(signal_date),
            "exit_date": pd.Timestamp(exit_date),
            "n_tickers": int(len(top)),
            "regime_state_med": float(df["regime_state"].median()),
            "vol_regime_med": float(df["volatility_regime"].median()),
            "nifty_ret_5d": float(top["nifty_ret"].mean()),
            "alpha_sign_acc": sign_acc,
            "top10_pos_alpha_hit": hit_rate,
            "top10_avg_alpha": avg_alpha,
            "top10_rank_ic": rank_ic,
            "top10_half_separation": separation,
            "blended_mean": float(top["blended"].mean()),
            "blended_std": float(top["blended"].std()),
            "calibrated_mean": float(top["calibrated"].mean()),
            "calibrated_std": float(top["calibrated"].std()),
        }
    )

summary = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
if summary.empty:
    raise RuntimeError("No diagnostic rows generated.")

top_rows_df = pd.concat(all_top_rows, ignore_index=True)

out_dir = Path("logs") / "alpha_diagnostics"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rolling_5d_alpha_summary.csv"
summary.to_csv(out_path, index=False)
top_rows_path = out_dir / "rolling_5d_alpha_top_rows.csv"
top_rows_df.to_csv(top_rows_path, index=False)

print("\nSaved:", out_path)
print("Saved:", top_rows_path)

# Overall stats
print("\n=== OVERALL ===")
print("windows:", len(summary))
print("mean alpha_sign_acc:", round(float(summary["alpha_sign_acc"].mean()), 4))
print("mean top10_pos_alpha_hit:", round(float(summary["top10_pos_alpha_hit"].mean()), 4))
print("mean top10_avg_alpha:", round(float(summary["top10_avg_alpha"].mean()), 4))
print("median top10_avg_alpha:", round(float(summary["top10_avg_alpha"].median()), 4))
print("p10 top10_avg_alpha:", round(float(summary["top10_avg_alpha"].quantile(0.10)), 4))
print("p90 top10_avg_alpha:", round(float(summary["top10_avg_alpha"].quantile(0.90)), 4))

# Fail windows by 90% sign-accuracy target.
fails = summary[summary["alpha_sign_acc"] < 0.90].copy()
print("\n=== FAILURE SET (alpha_sign_acc < 0.90) ===")
print("count:", len(fails), "/", len(summary))
print("mean fail alpha_sign_acc:", round(float(fails["alpha_sign_acc"].mean()), 4) if len(fails) else None)
print("mean fail top10_avg_alpha:", round(float(fails["top10_avg_alpha"].mean()), 4) if len(fails) else None)
print("mean fail top10_rank_ic:", round(float(fails["top10_rank_ic"].mean()), 4) if len(fails) else None)
print("mean fail half-separation:", round(float(fails["top10_half_separation"].mean()), 4) if len(fails) else None)
print("mean fail blended_std:", round(float(fails["blended_std"].mean()), 6) if len(fails) else None)

# Worst windows snapshot.
print("\n=== 10 WORST WINDOWS BY SIGN ACC ===")
cols = [
    "signal_date",
    "exit_date",
    "alpha_sign_acc",
    "top10_pos_alpha_hit",
    "top10_avg_alpha",
    "top10_rank_ic",
    "top10_half_separation",
    "nifty_ret_5d",
    "vol_regime_med",
    "blended_std",
]
print(summary.sort_values("alpha_sign_acc").head(10)[cols].to_string(index=False))

# Correlations for root-cause clues.
print("\n=== DIAGNOSTIC CORRELATIONS (Pearson) ===")
for x in ["top10_rank_ic", "top10_half_separation", "blended_std", "nifty_ret_5d", "vol_regime_med", "top10_avg_alpha"]:
    corr = summary["alpha_sign_acc"].corr(summary[x])
    print(f"corr(alpha_sign_acc, {x}) = {corr:.4f}")

# Theoretical ceiling from threshold tuning on current scores.
print("\n=== POOLED THRESHOLD CEILING (CURRENT SIGNAL) ===")
y = (top_rows_df["alpha"] > 0).astype(int).to_numpy()
s = top_rows_df["blended"].to_numpy()
thresholds = np.linspace(0.35, 0.65, 61)
best_t = None
best_acc = -1.0
for t in thresholds:
    pred = (s > t).astype(int)
    acc = float((pred == y).mean())
    if acc > best_acc:
        best_acc = acc
        best_t = t

base_acc = float(((s > 0.50).astype(int) == y).mean())
print("samples:", len(top_rows_df))
print("accuracy@0.50:", round(base_acc, 4))
print("best_threshold:", round(float(best_t), 4))
print("best_accuracy:", round(float(best_acc), 4))

# Window-level best possible with per-window threshold sweep (optimistic upper bound).
best_per_window = []
for signal_date, g in top_rows_df.groupby("signal_date"):
    yy = (g["alpha"] > 0).astype(int).to_numpy()
    ss = g["blended"].to_numpy()
    local_best = 0.0
    for t in thresholds:
        local_acc = float((((ss > t).astype(int)) == yy).mean())
        if local_acc > local_best:
            local_best = local_acc
    best_per_window.append(local_best)

print("mean best_per_window_accuracy:", round(float(np.mean(best_per_window)), 4))
print("p90 best_per_window_accuracy:", round(float(np.quantile(best_per_window, 0.90)), 4))
