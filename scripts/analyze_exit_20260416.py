import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import prepare_dataset
from src.features import create_features
from src.model import QuantPulseSimple
from src.zerodha_ohlc_loader import download_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model confidence picks for custom entry/exit dates.")
    parser.add_argument("--analysis-cutoff", type=str, default="", help="Last date allowed in training split (YYYY-MM-DD).")
    parser.add_argument("--entry-date", type=str, default="", help="Entry date for model picks (YYYY-MM-DD).")
    parser.add_argument(
        "--exit-date",
        type=str,
        default="",
        help="Exit date for realized return (YYYY-MM-DD). If empty with --entry-date, exit is derived from holding days.",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Number of stocks to select by confidence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_date = pd.Timestamp(args.exit_date) if args.exit_date else None
    holding_days = max(
        1,
        int(os.getenv("QUANT_HOLDING_DAYS", os.getenv("QUANT_TARGET_HORIZON_DAYS", "3"))),
    )
    top_k = max(1, int(args.top_k if args.top_k > 0 else int(os.getenv("QUANT_TOP_K", "3"))))
    hidden_size = int(os.getenv("QUANT_SIMPLE_HIDDEN_SIZE", "32"))
    model_path = Path("models/quantpulse_model.pth")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")

    os.environ["QUANT_FORWARD_INFERENCE"] = "1"
    raw = download_data().reset_index(drop=True).copy()
    raw["Date"] = pd.to_datetime(raw["Date"])

    trading_dates = np.sort(raw["Date"].drop_duplicates().values)
    trading_dates = pd.to_datetime(trading_dates)
    if args.entry_date:
        entry_date = pd.Timestamp(args.entry_date)
        if entry_date not in set(trading_dates):
            raise RuntimeError(f"Entry date {entry_date.date()} not available in data")
        entry_idx = int(np.where(trading_dates == entry_date)[0][0])
        if exit_date is None:
            exit_idx = entry_idx + holding_days
            if exit_idx >= len(trading_dates):
                raise RuntimeError("Not enough future history after entry date for configured holding")
            exit_date = pd.Timestamp(trading_dates[exit_idx])
    else:
        if exit_date is None:
            raise RuntimeError("Provide --exit-date when --entry-date is not provided")
        if exit_date not in set(trading_dates):
            raise RuntimeError(f"Exit date {exit_date.date()} not available in data")
        exit_idx = int(np.where(trading_dates == exit_date)[0][0])
        entry_idx = exit_idx - holding_days
        if entry_idx < 0:
            raise RuntimeError("Not enough history before exit date for configured holding")
        entry_date = pd.Timestamp(trading_dates[entry_idx])

    if exit_date not in set(trading_dates):
        raise RuntimeError(f"Exit date {exit_date.date()} not available in data")

    split_date = str(pd.Timestamp(args.analysis_cutoff).date()) if args.analysis_cutoff else str(entry_date.date())

    feat = create_features(raw)
    x_train, x_test, y_train, y_test, dates_train, tickers_test, dates_test = prepare_dataset(
        feat,
        split_date=split_date,
    )

    if len(x_test) == 0:
        raise RuntimeError("No test samples generated")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantPulseSimple(input_size=x_test.shape[2], hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor(x_test.copy(), dtype=torch.float32).to(device)),
            dim=1,
        )[:, 1].cpu().numpy()

    pred = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_test),
            "ticker": tickers_test,
            "confidence": probs,
        }
    )
    entry_pred = pred[pred["date"] == entry_date].copy()
    if entry_pred.empty:
        raise RuntimeError(f"No predictions available on entry date {entry_date.date()}")

    picks = entry_pred.sort_values("confidence", ascending=False).head(top_k).copy()

    px = raw[["Date", "Ticker", "Close"]].copy()
    px = px.rename(columns={"Date": "date", "Ticker": "ticker", "Close": "close"})

    entry_px = px[px["date"] == entry_date][["ticker", "close"]].rename(columns={"close": "entry_close"})
    exit_px = px[px["date"] == exit_date][["ticker", "close"]].rename(columns={"close": "exit_close"})

    out = picks.merge(entry_px, on="ticker", how="left").merge(exit_px, on="ticker", how="left")
    out["stock_return"] = (out["exit_close"] / out["entry_close"]) - 1.0

    nifty = (
        raw[["Date", "nifty_close"]]
        .dropna(subset=["nifty_close"])
        .drop_duplicates(subset=["Date"])
        .rename(columns={"Date": "date"})
    )
    nifty_entry = float(nifty.loc[nifty["date"] == entry_date, "nifty_close"].iloc[0])
    nifty_exit = float(nifty.loc[nifty["date"] == exit_date, "nifty_close"].iloc[0])
    nifty_return = (nifty_exit / nifty_entry) - 1.0

    out["nifty_return"] = nifty_return
    out["alpha"] = out["stock_return"] - out["nifty_return"]
    out["beat_nifty"] = out["alpha"] > 0

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_exit = str(exit_date.date())
    safe_entry = str(entry_date.date())
    csv_path = out_dir / f"entry_{safe_entry}_exit_{safe_exit}_top{top_k}.csv"
    json_path = out_dir / f"entry_{safe_entry}_exit_{safe_exit}_top{top_k}_summary.json"
    out.to_csv(csv_path, index=False)

    summary = {
        "analysis_cutoff": split_date,
        "entry_date": str(entry_date.date()),
        "exit_date": str(exit_date.date()),
        "holding_days": int(holding_days),
        "top_k": int(top_k),
        "nifty_return": float(nifty_return),
        "avg_stock_return": float(out["stock_return"].mean()),
        "avg_alpha": float(out["alpha"].mean()),
        "beat_nifty_rate": float(out["beat_nifty"].mean()),
        "picks": out[
            ["ticker", "confidence", "entry_close", "exit_close", "stock_return", "alpha", "beat_nifty"]
        ].to_dict(orient="records"),
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Model setting evaluation complete")
    print("entry_date", summary["entry_date"])
    print("exit_date", summary["exit_date"])
    print("holding_days", summary["holding_days"])
    print("top_k", summary["top_k"])
    print("nifty_return", round(summary["nifty_return"], 6))
    print("avg_stock_return", round(summary["avg_stock_return"], 6))
    print("avg_alpha", round(summary["avg_alpha"], 6))
    print("beat_nifty_rate", round(summary["beat_nifty_rate"], 6))
    print("csv", str(csv_path))
    print("json", str(json_path))


if __name__ == "__main__":
    main()
