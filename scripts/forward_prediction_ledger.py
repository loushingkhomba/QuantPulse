import argparse
import json
import os
import sys
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
    parser = argparse.ArgumentParser(description="Generate forward top-k predictions and append to ledger.")
    parser.add_argument("--analysis-date", type=str, default="", help="Analyze data up to this date (YYYY-MM-DD). Default=today.")
    parser.add_argument("--entry-date", type=str, required=True, help="Planned entry date (YYYY-MM-DD).")
    parser.add_argument("--exit-date", type=str, required=True, help="Planned exit date (YYYY-MM-DD).")
    parser.add_argument("--top-k", type=int, default=10, help="Number of stocks to keep in ledger.")
    parser.add_argument("--holding-days", type=int, default=3, help="Configured model holding days metadata.")
    parser.add_argument(
        "--ledger-path",
        type=str,
        default="logs/prediction_ledger.csv",
        help="CSV ledger path.",
    )
    return parser.parse_args()


def _choose_signal_date(available_dates: pd.DatetimeIndex, analysis_date: pd.Timestamp) -> pd.Timestamp:
    usable = available_dates[available_dates <= analysis_date]
    if len(usable) == 0:
        raise RuntimeError(f"No trading dates available on or before {analysis_date.date()}")
    return pd.Timestamp(usable[-1])


def _load_model(input_size: int, hidden_size: int) -> QuantPulseSimple:
    model_path = Path("models/quantpulse_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantPulseSimple(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _append_ledger(ledger_path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
    else:
        ledger = pd.DataFrame()

    combined = pd.concat([ledger, rows], ignore_index=True)
    combined = combined.sort_values(["created_at", "rank", "ticker"]).drop_duplicates(
        subset=["signal_date", "planned_entry_date", "planned_exit_date", "ticker"],
        keep="last",
    )
    combined.to_csv(ledger_path, index=False)
    return combined


def main() -> None:
    args = parse_args()

    top_k = max(1, int(args.top_k))
    holding_days = max(1, int(args.holding_days))
    hidden_size = int(os.getenv("QUANT_SIMPLE_HIDDEN_SIZE", "32"))
    analysis_date = pd.Timestamp(args.analysis_date).normalize() if args.analysis_date else pd.Timestamp.now().normalize()
    entry_date = pd.Timestamp(args.entry_date).normalize()
    exit_date = pd.Timestamp(args.exit_date).normalize()

    os.environ["QUANT_FORWARD_INFERENCE"] = "1"
    raw = download_data().reset_index(drop=True).copy()
    raw["Date"] = pd.to_datetime(raw["Date"]) 

    trading_dates = pd.DatetimeIndex(sorted(raw["Date"].drop_duplicates()))
    signal_date = _choose_signal_date(trading_dates, analysis_date)

    feat = create_features(raw)
    x_train, x_test, y_train, y_test, dates_train, tickers_test, dates_test = prepare_dataset(
        feat,
        split_date=str(signal_date.date()),
    )

    if len(x_test) == 0:
        raise RuntimeError("No test samples generated")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(input_size=x_test.shape[2], hidden_size=hidden_size)

    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor(x_test.copy(), dtype=torch.float32).to(device)),
            dim=1,
        )[:, 1].cpu().numpy()

    pred = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(dates_test),
            "ticker": tickers_test,
            "confidence": probs,
        }
    )

    signal_pred = pred[pred["signal_date"] == signal_date].copy()
    if signal_pred.empty:
        raise RuntimeError(f"No predictions generated for signal date {signal_date.date()}")

    picks = signal_pred.sort_values("confidence", ascending=False).head(top_k).copy().reset_index(drop=True)
    picks["rank"] = np.arange(1, len(picks) + 1)

    run_id = f"pred_{signal_date.date()}_entry_{entry_date.date()}_exit_{exit_date.date()}_top{top_k}"
    created_at = pd.Timestamp.now().isoformat()

    ledger_rows = picks[["rank", "ticker", "confidence"]].copy()
    ledger_rows["run_id"] = run_id
    ledger_rows["created_at"] = created_at
    ledger_rows["analysis_date"] = str(analysis_date.date())
    ledger_rows["signal_date"] = str(signal_date.date())
    ledger_rows["planned_entry_date"] = str(entry_date.date())
    ledger_rows["planned_exit_date"] = str(exit_date.date())
    ledger_rows["holding_days"] = int(holding_days)
    ledger_rows["status"] = "PENDING_ENTRY"
    ledger_rows["entry_close"] = np.nan
    ledger_rows["exit_close"] = np.nan
    ledger_rows["stock_return"] = np.nan
    ledger_rows["nifty_return"] = np.nan
    ledger_rows["alpha"] = np.nan

    ledger_path = Path(args.ledger_path)
    full_ledger = _append_ledger(ledger_path, ledger_rows)

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    picks_csv = out_dir / f"forward_picks_{run_id}.csv"
    summary_json = out_dir / f"forward_picks_{run_id}.json"

    ledger_rows.to_csv(picks_csv, index=False)
    summary = {
        "run_id": run_id,
        "analysis_date": str(analysis_date.date()),
        "signal_date": str(signal_date.date()),
        "planned_entry_date": str(entry_date.date()),
        "planned_exit_date": str(exit_date.date()),
        "holding_days": int(holding_days),
        "top_k": int(top_k),
        "ledger_path": str(ledger_path),
        "count_written": int(len(ledger_rows)),
        "ledger_total_rows": int(len(full_ledger)),
        "picks": ledger_rows[["rank", "ticker", "confidence"]].to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Forward prediction ledger update complete")
    print("analysis_date", summary["analysis_date"])
    print("signal_date", summary["signal_date"])
    print("planned_entry_date", summary["planned_entry_date"])
    print("planned_exit_date", summary["planned_exit_date"])
    print("holding_days", summary["holding_days"])
    print("top_k", summary["top_k"])
    print("count_written", summary["count_written"])
    print("ledger_total_rows", summary["ledger_total_rows"])
    print("ledger", summary["ledger_path"])
    print("picks_csv", str(picks_csv))
    print("summary_json", str(summary_json))


if __name__ == "__main__":
    main()
