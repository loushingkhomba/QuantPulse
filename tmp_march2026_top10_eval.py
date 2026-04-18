import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.zerodha_ohlc_loader import download_data
from src.features import create_features
from src.dataset import prepare_dataset
from src.model import QuantPulseSimple


MODEL_PATH = Path("models/quantpulse_model.pth")
SEQUENCE_LEN = 20
HIDDEN_SIZE = int(os.getenv("QUANT_SIMPLE_HIDDEN_SIZE", "32"))
TARGET_EXIT_DATE = pd.Timestamp("2026-04-16")


def derive_signal_date_from_exit(all_dates: np.ndarray, exit_date: pd.Timestamp, hold_days: int) -> pd.Timestamp:
    dates = pd.to_datetime(np.sort(all_dates))
    if exit_date not in dates:
        last_available = pd.Timestamp(dates[-1]) if len(dates) else None
        raise RuntimeError(
            f"Requested exit date {exit_date.date()} not present in data. "
            f"Last available trading date: {last_available.date() if last_available is not None else 'N/A'}."
        )
    exit_idx = int(np.where(dates == exit_date)[0][0])
    signal_idx = exit_idx - hold_days
    if signal_idx < 0:
        raise RuntimeError("Not enough historical trading sessions before requested exit date.")
    return pd.Timestamp(dates[signal_idx])


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    raw = download_data().reset_index().copy()
    raw["Date"] = pd.to_datetime(raw["Date"])

    EXIT_DAYS = 5  # hold for 5 trading days

    all_unique_dates = np.sort(raw["Date"].unique())
    exit_date = TARGET_EXIT_DATE
    signal_date = derive_signal_date_from_exit(all_unique_dates, exit_date, EXIT_DAYS)

    window_start = signal_date - pd.DateOffset(years=2)
    window_end = exit_date

    window_raw = raw[(raw["Date"] >= window_start) & (raw["Date"] <= window_end)].copy()
    if window_raw.empty:
        raise RuntimeError("2-year window is empty after filtering.")

    # Keep forward inference rows so latest date rows are preserved.
    os.environ["QUANT_FORWARD_INFERENCE"] = "1"
    feat = create_features(window_raw)

    # ── Market regime detection (mirrors train.py thresholds exactly) ──────
    BAD_VOLATILITY_CUTOFF  = 1.25
    BAD_DRAWDOWN_CUTOFF    = -0.08
    TREND_STRENGTH_CUTOFF  = 0.00

    feat_date_col = "Date" if "Date" in feat.columns else feat.index.name
    if feat_date_col and feat_date_col != feat.index.name:
        regime_day = feat[pd.to_datetime(feat["Date"]) == signal_date]
    else:
        regime_day = feat[pd.to_datetime(feat.index) == signal_date]

    if regime_day.empty:
        # fall back to last available date <= signal_date
        if feat_date_col and feat_date_col != feat.index.name:
            regime_day = feat[pd.to_datetime(feat["Date"]) <= signal_date].tail(1)
        else:
            regime_day = feat[pd.to_datetime(feat.index) <= signal_date].tail(1)

    day_regime_state      = float(regime_day["regime_state"].median())      if "regime_state"      in regime_day.columns else 0.0
    day_volatility_regime = float(regime_day["volatility_regime"].median())  if "volatility_regime"  in regime_day.columns else 1.0
    day_drawdown_state    = float(regime_day["nifty_drawdown_63d"].median()) if "nifty_drawdown_63d" in regime_day.columns else 0.0
    day_trend_strength    = float(regime_day["trend_strength"].median())     if "trend_strength"     in regime_day.columns else 0.0

    is_bad_regime = (
        (day_regime_state < 0)
        or (day_volatility_regime > BAD_VOLATILITY_CUTOFF)
        or (day_drawdown_state    < BAD_DRAWDOWN_CUTOFF)
    )
    is_trending_regime = (not is_bad_regime) and (day_trend_strength > TREND_STRENGTH_CUTOFF)
    regime_label = "BEARISH (BAD)" if is_bad_regime else ("BULLISH (TRENDING)" if is_trending_regime else "NEUTRAL")
    # ── end regime detection ────────────────────────────────────────────────

    # Build dataset with split at the selected signal date.
    X_train, X_test, y_train, y_test, dates_train, tickers_test, dates_test = prepare_dataset(
        feat,
        sequence_length=SEQUENCE_LEN,
        split_date=str(signal_date.date()),
    )

    if len(X_test) == 0:
        raise RuntimeError("No test samples generated for selected signal date.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantPulseSimple(input_size=X_test.shape[2], hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.tensor(X_test.copy(), dtype=torch.float32).to(device)
        probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()

    pred = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_test),
            "ticker": tickers_test,
            "confidence": probs,
            "label_5d": y_test,
        }
    )

    day_pred = pred[pred["date"] == signal_date].copy()
    if day_pred.empty:
        nearest_pred_dates = np.sort(pred["date"].dropna().unique())
        if len(nearest_pred_dates) == 0:
            raise RuntimeError("No prediction rows available for requested window.")
        fallback_date = pd.Timestamp(nearest_pred_dates[-1])
        print(f"Warning: exact signal date {signal_date.date()} missing in predictions; using {fallback_date.date()} instead.")
        signal_date = fallback_date
        day_pred = pred[pred["date"] == signal_date].copy()

    top10 = day_pred.sort_values("confidence", ascending=False).head(10).copy()

    # 5-day realized return for each ticker.
    px = raw[["Date", "Ticker", "Close"]].copy()
    px = px.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    px["exit_date"] = px.groupby("Ticker")["Date"].shift(-EXIT_DAYS)
    px["exit_close"] = px.groupby("Ticker")["Close"].shift(-EXIT_DAYS)
    px["return_5d"] = (px["exit_close"] / px["Close"]) - 1.0

    signal_px = px[px["Date"] == signal_date][["Ticker", "Close", "exit_date", "exit_close", "return_5d"]]
    signal_px = signal_px.rename(columns={"Ticker": "ticker"})

    nifty_day = (
        raw[["Date", "nifty_close"]]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    nifty_row = nifty_day[nifty_day["Date"] == signal_date]
    nifty_exit_row = nifty_day[nifty_day["Date"] == exit_date]
    if nifty_row.empty or nifty_exit_row.empty:
        raise RuntimeError("Missing Nifty close for signal/exit date.")
    nifty_ret = float(nifty_exit_row["nifty_close"].iloc[0] / nifty_row["nifty_close"].iloc[0] - 1.0)

    out = top10.merge(signal_px, on="ticker", how="left")
    out["beat_nifty_5d"] = out["return_5d"] > nifty_ret
    out["return_5d_positive"] = out["return_5d"] > 0

    # Save detailed output artifact.
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "top10_apr2026_5d_eval.csv"
    out.to_csv(out_path, index=False)

    valid_returns = out["return_5d"].dropna()
    avg_ret = float(valid_returns.mean()) if len(valid_returns) else float("nan")
    hit_rate = float((out["return_5d_positive"] == True).mean()) if len(out) else float("nan")
    beat_nifty = float((out["beat_nifty_5d"] == True).mean()) if len(out) else float("nan")

    print("\nTOP-10  5-DAY EXIT EVALUATION")
    # ── Capital simulation ──────────────────────────────────────────────────
    CAPITAL = 10000.0
    TOP_K = 3                        # matches train.py top_k=3 (actual model behaviour)
    TRANSACTION_COST_PCT = 0.0005    # 0.05% each side
    SLIPPAGE_PCT = 0.0002            # 0.02% each side
    round_trip_cost = 2 * (TRANSACTION_COST_PCT + SLIPPAGE_PCT)

    nifty_exit_val = CAPITAL * (1.0 + nifty_ret)

    def _run_sim(picks: pd.DataFrame, label: str, weighted: bool) -> dict:
        picks = picks.copy().reset_index(drop=True)
        n = len(picks)
        if weighted:
            raw_w = picks["confidence"].values.astype(float)
            raw_w = np.clip(raw_w - raw_w.min() + 1e-6, 0, None)
            allocs = (raw_w / raw_w.sum()) * CAPITAL
        else:
            allocs = np.full(n, CAPITAL / n)
        rows = []
        for i, row in picks.iterrows():
            ret = row["return_5d"]
            alloc = allocs[i]
            if pd.isna(ret):
                rows.append({"ticker": row["ticker"], "confidence": row["confidence"],
                             "alloc": alloc, "gross": float("nan"),
                             "net": float("nan"), "exit_val": float("nan"), "pnl": float("nan")})
            else:
                net = float(ret) - round_trip_cost
                exit_val = alloc * (1.0 + net)
                rows.append({"ticker": row["ticker"], "confidence": row["confidence"],
                             "alloc": alloc, "gross": float(ret),
                             "net": net, "exit_val": exit_val, "pnl": exit_val - alloc})
        df = pd.DataFrame(rows)
        total_exit = df["exit_val"].sum()
        total_pnl  = df["pnl"].sum()
        edge       = total_pnl - (nifty_exit_val - CAPITAL)
        return {"label": label, "df": df, "total_exit": total_exit,
                "total_pnl": total_pnl, "edge": edge}

    top3   = out.head(TOP_K)
    sim_a  = _run_sim(out.head(10),  "Scenario A : Top-10  equal-weight  (all picks)", weighted=False)
    sim_b  = _run_sim(top3,          "Scenario B : Top-3   equal-weight  (model behaviour)", weighted=False)
    sim_c  = _run_sim(top3,          "Scenario C : Top-3   confidence-weighted            ", weighted=True)

    # Scenario D: Top-3 + regime filter (hold cash in bad regime)
    if is_bad_regime:
        sim_d = {
            "label": "Scenario D : Top-3   equal-weight  + REGIME FILTER (HOLD CASH)",
            "df": top3[["ticker", "confidence"]].assign(
                alloc=0.0, gross=float("nan"), net=float("nan"),
                exit_val=float("nan"), pnl=float("nan")
            ),
            "total_exit": CAPITAL,
            "total_pnl": 0.0,
            "edge": 0.0 - (nifty_exit_val - CAPITAL),
        }
    else:
        _tmp = _run_sim(top3, "Scenario D : Top-3   equal-weight  + REGIME FILTER (TRADE)", weighted=False)
        sim_d = _tmp

    # save combined CSV
    sim_path = out_dir / "top10_apr2026_5d_simulation.csv"
    sim_a["df"].assign(scenario="A_top10_equal").to_csv(sim_path, index=False)
    for tag, sim in [("B_top3_equal", sim_b), ("C_top3_weighted", sim_c), ("D_top3_regime_filter", sim_d)]:
        sim["df"].assign(scenario=tag).to_csv(sim_path, mode="a", header=False, index=False)
    # ── end simulation ──────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print(f"  MARKET REGIME CHECK  —  {signal_date.date()}")
    print(f"{'='*65}")
    print(f"  regime_state      : {day_regime_state:.2f}  (< 0 = bad)")
    print(f"  volatility_regime : {day_volatility_regime:.3f}  (> {BAD_VOLATILITY_CUTOFF} = bad)")
    print(f"  nifty_drawdown_63d: {day_drawdown_state:.4f}  (< {BAD_DRAWDOWN_CUTOFF} = bad)")
    print(f"  trend_strength    : {day_trend_strength:.4f}  (> {TREND_STRENGTH_CUTOFF} = trending)")
    print(f"  *** REGIME = {regime_label} ***")
    if is_bad_regime:
        print(f"  Regime filter decision: HOLD CASH — no new positions opened")
    else:
        print(f"  Regime filter decision: PROCEED — market conditions acceptable")
    print(f"{'='*65}")

    print(f"\nSignal date:  {signal_date.date()}")
    print(f"Exit date:    {exit_date.date()}  ({EXIT_DAYS} trading days later)")
    print(f"2-year window: {window_start.date()} to {window_end.date()}")
    print(f"Samples in train/test: {len(X_train)} / {len(X_test)}")
    print(f"Nifty 5-day return:      {nifty_ret:.5f}  ({nifty_ret:.2%})")
    print(f"Top10 avg 5-day return:  {avg_ret:.5f}  ({avg_ret:.2%})")
    print(f"Top10 hit-rate (5d > 0): {hit_rate:.2%}")
    print(f"Top10 beat-Nifty rate:   {beat_nifty:.2%}")
    print(f"Detailed CSV: {out_path}")
    print("\nTop 10 rows:")
    display_cols = [
        "ticker",
        "confidence",
        "return_5d",
        "return_5d_positive",
        "beat_nifty_5d",
        "exit_date",
    ]
    print(out[display_cols].to_string(index=False))

    def _print_sim(sim: dict) -> None:
        df = sim["df"]
        print(f"\n{'='*65}")
        print(f"  {sim['label']}")
        print(f"{'='*65}")
        print(f"  Entry: {signal_date.date()}   Exit: {exit_date.date()}   Capital: ₹{CAPITAL:,.0f}")
        print(f"  Round-trip cost: {round_trip_cost*100:.3f}%  (slippage + brokerage)")
        print(f"  {'-'*61}")
        print(f"  {'Ticker':<16} {'Alloc (₹)':>10} {'Gross':>8} {'Net':>8} {'Exit ₹':>10} {'P&L ₹':>10}")
        print(f"  {'-'*61}")
        for _, r in df.iterrows():
            gross = f"{r['gross']:.2%}" if not pd.isna(r['gross']) else "N/A"
            net   = f"{r['net']:.2%}"   if not pd.isna(r['net'])   else "N/A"
            ev    = f"₹{r['exit_val']:,.2f}" if not pd.isna(r['exit_val']) else "N/A"
            pnl_s = f"₹{r['pnl']:+,.2f}" if not pd.isna(r['pnl']) else "N/A"
            print(f"  {r['ticker']:<16} ₹{r['alloc']:>8,.2f} {gross:>8} {net:>8} {ev:>10} {pnl_s:>10}")
        print(f"  {'-'*61}")
        print(f"  {'TOTAL':<16} ₹{CAPITAL:>8,.2f} {'':>8} {'':>8} ₹{sim['total_exit']:>8,.2f} ₹{sim['total_pnl']:>+8,.2f}")
        print(f"\n  Nifty buy-and-hold : ₹{CAPITAL:,.0f} → ₹{nifty_exit_val:,.2f}  (₹{nifty_exit_val - CAPITAL:+,.2f})")
        print(f"  Model vs Nifty     : ₹{sim['edge']:+,.2f}")

    for sim in [sim_a, sim_b, sim_c, sim_d]:
        _print_sim(sim)

    print(f"\n{'='*65}")
    print(f"  SUMMARY COMPARISON  —  ₹{CAPITAL:,.0f} invested")
    print(f"  Market regime on {signal_date.date()}: {regime_label}")
    print(f"{'='*65}")
    print(f"  {'Scenario':<48} {'Exit ₹':>9} {'P&L ₹':>9} {'vs Nifty':>9}")
    print(f"  {'-'*61}")
    for sim in [sim_a, sim_b, sim_c, sim_d]:
        short = sim['label'].split(':')[1].strip()
        print(f"  {short:<48} ₹{sim['total_exit']:>7,.2f} ₹{sim['total_pnl']:>+7,.2f} ₹{sim['edge']:>+7,.2f}")
    print(f"  {'Nifty buy-and-hold':<48} ₹{nifty_exit_val:>7,.2f} ₹{nifty_exit_val - CAPITAL:>+7,.2f} {'—':>9}")
    print(f"{'='*65}")
    if is_bad_regime:
        print(f"  Regime filter saved ₹{sim_b['total_pnl'] - sim_d['total_pnl']:+,.2f} vs trading without filter")
        print(f"  Regime filter saved ₹{sim_d['edge']:+,.2f} vs Nifty")
    else:
        print(f"  Regime is OK — filter allows trading (same result as Scenario B)")
    print(f"\n  Simulation CSV: {sim_path}")


if __name__ == "__main__":
    main()
