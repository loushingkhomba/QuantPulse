import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zerodha_ohlc_loader import download_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build slippage and alpha tracking sheet from ledger + fills.")
    parser.add_argument("--ledger", type=str, default="logs/prediction_ledger.csv", help="Prediction ledger CSV path.")
    parser.add_argument("--fills", type=str, default="logs/live_fills.csv", help="Live fills CSV path.")
    parser.add_argument(
        "--out",
        type=str,
        default="logs/slippage_alpha_tracking.csv",
        help="Output tracking sheet path.",
    )
    return parser.parse_args()


def _safe_read_csv(path: Path, required_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=required_cols)
    df = pd.read_csv(path)
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _prep_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    px = raw[["Date", "Ticker", "Close"]].copy()
    px["Date"] = pd.to_datetime(px["Date"]).dt.normalize()
    px = px.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"], keep="last")

    nifty = (
        raw[["Date", "nifty_close"]]
        .dropna(subset=["nifty_close"])
        .copy()
    )
    nifty["Date"] = pd.to_datetime(nifty["Date"]).dt.normalize()
    nifty = nifty.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    nifty_series = nifty.set_index("Date")["nifty_close"].astype(float)
    return px, nifty_series


def _prep_fills(fills: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "signal_date",
        "ticker",
        "expected_entry_price",
        "actual_entry_price",
        "expected_exit_price",
        "actual_exit_price",
        "status",
    ]
    fills = fills[[c for c in key_cols if c in fills.columns]].copy()
    if fills.empty:
        return pd.DataFrame(columns=key_cols)

    fills["signal_date"] = pd.to_datetime(fills["signal_date"], errors="coerce").dt.normalize()
    fills = fills.sort_values("signal_date").drop_duplicates(subset=["signal_date", "ticker"], keep="last")
    return fills


def _status_from_row(row: pd.Series) -> str:
    if pd.notna(row.get("actual_exit_price")):
        return "CLOSED"
    if pd.notna(row.get("actual_entry_price")):
        return "OPEN"
    entry_date = row.get("planned_entry_date")
    today = pd.Timestamp.now().normalize()
    if pd.notna(entry_date) and entry_date <= today:
        return "ENTRY_DUE"
    return "PENDING_ENTRY"


def main() -> None:
    args = parse_args()

    ledger_path = Path(args.ledger)
    fills_path = Path(args.fills)
    out_path = Path(args.out)

    ledger_required = [
        "run_id",
        "rank",
        "ticker",
        "confidence",
        "signal_date",
        "planned_entry_date",
        "planned_exit_date",
        "holding_days",
    ]
    ledger = _safe_read_csv(ledger_path, ledger_required)
    if ledger.empty:
        raise RuntimeError(f"Ledger empty or missing: {ledger_path}")

    for col in ["signal_date", "planned_entry_date", "planned_exit_date"]:
        ledger[col] = pd.to_datetime(ledger[col], errors="coerce").dt.normalize()

    fills_required = [
        "signal_date",
        "ticker",
        "expected_entry_price",
        "actual_entry_price",
        "expected_exit_price",
        "actual_exit_price",
        "status",
    ]
    fills = _safe_read_csv(fills_path, fills_required)
    fills = _prep_fills(fills)

    raw = download_data().reset_index(drop=True).copy()
    raw["Date"] = pd.to_datetime(raw["Date"])  # ensure datetime
    px, nifty_series = _prep_prices(raw)

    # Expected prices from actual market close on planned dates when available.
    entry_lookup = (
        px.rename(columns={"Date": "planned_entry_date", "Close": "expected_entry_price", "Ticker": "ticker"})
        [["planned_entry_date", "ticker", "expected_entry_price"]]
    )
    exit_lookup = (
        px.rename(columns={"Date": "planned_exit_date", "Close": "expected_exit_price", "Ticker": "ticker"})
        [["planned_exit_date", "ticker", "expected_exit_price"]]
    )

    track = ledger.merge(entry_lookup, on=["planned_entry_date", "ticker"], how="left")
    track = track.merge(exit_lookup, on=["planned_exit_date", "ticker"], how="left")

    if not fills.empty:
        track = track.merge(
            fills[[
                "signal_date",
                "ticker",
                "expected_entry_price",
                "actual_entry_price",
                "expected_exit_price",
                "actual_exit_price",
                "status",
            ]],
            on=["signal_date", "ticker"],
            how="left",
            suffixes=("", "_fill"),
        )

        # Prefer explicit values from fills file when provided.
        for c in ["expected_entry_price", "expected_exit_price", "actual_entry_price", "actual_exit_price"]:
            fill_col = f"{c}_fill"
            if fill_col in track.columns:
                track[c] = np.where(track[fill_col].notna(), track[fill_col], track[c])
                track.drop(columns=[fill_col], inplace=True)

        if "status_fill" in track.columns:
            track["status_from_fills"] = track["status_fill"]
            track.drop(columns=["status_fill"], inplace=True)

    track["expected_entry_price"] = pd.to_numeric(track["expected_entry_price"], errors="coerce")
    track["expected_exit_price"] = pd.to_numeric(track["expected_exit_price"], errors="coerce")
    track["actual_entry_price"] = pd.to_numeric(track.get("actual_entry_price", np.nan), errors="coerce")
    track["actual_exit_price"] = pd.to_numeric(track.get("actual_exit_price", np.nan), errors="coerce")

    track["entry_slippage_bps"] = (
        (track["actual_entry_price"] - track["expected_entry_price"]) / track["expected_entry_price"]
    ) * 10000.0
    track["exit_slippage_bps"] = (
        (track["actual_exit_price"] - track["expected_exit_price"]) / track["expected_exit_price"]
    ) * 10000.0

    track["expected_stock_return_pct"] = (
        (track["expected_exit_price"] / track["expected_entry_price"]) - 1.0
    ) * 100.0
    track["realized_stock_return_pct"] = (
        (track["actual_exit_price"] / track["actual_entry_price"]) - 1.0
    ) * 100.0

    # Nifty benchmark using planned window.
    track["nifty_entry_close"] = track["planned_entry_date"].map(nifty_series)
    track["nifty_exit_close"] = track["planned_exit_date"].map(nifty_series)
    track["nifty_return_pct"] = ((track["nifty_exit_close"] / track["nifty_entry_close"]) - 1.0) * 100.0

    track["expected_alpha_pct"] = track["expected_stock_return_pct"] - track["nifty_return_pct"]
    track["realized_alpha_pct"] = track["realized_stock_return_pct"] - track["nifty_return_pct"]

    track["status"] = track.apply(_status_from_row, axis=1)
    if "status_from_fills" in track.columns:
        track["status"] = np.where(track["status_from_fills"].notna(), track["status_from_fills"], track["status"])

    track["updated_at"] = pd.Timestamp.now().isoformat()

    out_cols = [
        "run_id",
        "rank",
        "ticker",
        "confidence",
        "signal_date",
        "planned_entry_date",
        "planned_exit_date",
        "holding_days",
        "expected_entry_price",
        "actual_entry_price",
        "entry_slippage_bps",
        "expected_exit_price",
        "actual_exit_price",
        "exit_slippage_bps",
        "expected_stock_return_pct",
        "realized_stock_return_pct",
        "nifty_return_pct",
        "expected_alpha_pct",
        "realized_alpha_pct",
        "status",
        "updated_at",
    ]
    for col in out_cols:
        if col not in track.columns:
            track[col] = np.nan

    track = track[out_cols].sort_values(["planned_entry_date", "rank", "ticker"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(out_path, index=False)

    print("Slippage and alpha tracking sheet updated")
    print("rows", len(track))
    print("out", str(out_path))
    print("pending", int((track["status"] == "PENDING_ENTRY").sum()))
    print("entry_due", int((track["status"] == "ENTRY_DUE").sum()))
    print("open", int((track["status"] == "OPEN").sum()))
    print("closed", int((track["status"] == "CLOSED").sum()))


if __name__ == "__main__":
    main()
