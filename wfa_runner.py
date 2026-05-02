import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train.py"
PYTHON_EXE = ROOT / "venv" / "Scripts" / "python.exe"
OUT_DIR = ROOT / "logs" / "wfa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_ENV = os.environ.copy()
BASE_ENV.update(
    {
        "QUANT_PROFILE": "v3robustnifty50",
        "QUANT_HISTORY_YEARS": "7",
        "QUANT_OBJECTIVE_FREEZE_PRESET": "v3robustnifty50",
        "QUANT_TRADE_ACCEPTANCE_ENABLED": "1",
        "QUANT_SIGNAL_INVERSION_MODE": "auto",
        "QUANT_AUTO_INVERT_SIGNAL": "1",
        "QUANT_FP_COST_MULTIPLIER": "0.0",
        "QUANT_FP_COST_RANDOMIZATION_ENABLED": "0",
        "QUANT_CASH_MODE_ENABLED": "1",
        "QUANT_CASH_MODE_MIN_IC": "0.0",
        "QUANT_CASH_MODE_MIN_SHARPE": "-0.5",
        "QUANT_CIRCUIT_BREAKER_ENABLED": "1",
        "QUANT_CIRCUIT_SHARPE_TIER1": "-1.0",
        "QUANT_CIRCUIT_SHARPE_TIER2": "-2.5",
        "QUANT_CIRCUIT_RISK_TIER1": "0.35",
        "QUANT_CIRCUIT_RISK_TIER2": "0.0",
        "QUANT_TRAIN_EPOCHS": "15",
        "QUANT_TRAIN_PATIENCE": "6",
    }
)

def _extract_float(name: str, text: str):
    m = re.search(rf"{re.escape(name)}:\s*([-+]?\d+(?:\.\d+)?)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _extract_int(name: str, text: str):
    m = re.search(rf"{re.escape(name)}:\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def _slice_section(text: str, section_start: str, section_end: str = "") -> str:
    start = text.find(section_start)
    if start < 0:
        return ""
    chunk = text[start:]
    if section_end:
        end = chunk.find(section_end)
        if end >= 0:
            return chunk[:end]
    return chunk


def _parse_run_output(stdout_text: str):
    real_block = _slice_section(stdout_text, "REAL MODEL", "RANDOM BASELINE")
    if not real_block:
        raise RuntimeError("Could not parse REAL MODEL block from train.py output")
    bh_block = _slice_section(stdout_text, "NIFTY BUY & HOLD")

    signal_reversal = "[SIGNAL REVERSAL DETECTED]" in stdout_text

    return {
        "real_final": _extract_float("Final Value", real_block),
        "real_sharpe": _extract_float("Sharpe", real_block),
        "real_max_dd_pct": _extract_float("Max Drawdown", real_block),
        "real_annualized_return_pct": _extract_float("Annualized Return", real_block),
        "real_trade_days": _extract_int("Trade Days", real_block),
        "real_trades_executed": _extract_int("Trades Executed", real_block),
        "nifty_final": _extract_float("Final Value", bh_block) if bh_block else None,
        "days_in_cash_mode": _extract_int("days_in_cash_mode", real_block),
        "days_circuit_breaker_tier1": _extract_int("days_circuit_breaker_tier1", real_block),
        "days_circuit_breaker_tier2": _extract_int("days_circuit_breaker_tier2", real_block),
        "blocked_by_cash_mode": _extract_int("blocked_by_cash_mode", real_block),
        "blocked_by_trade_acceptance": _extract_int("blocked_by_trade_acceptance", real_block),
        "signal_reversal_detected": bool(signal_reversal),
    }


def generate_windows(num_windows=10, train_years=2, test_months=6):
    min_split_date = pd.Timestamp(os.getenv("QUANT_WFA_MIN_SPLIT_DATE", "2020-07-01"))
    windows = []
    current_test_end = pd.Timestamp("2024-12-31")

    for _ in range(num_windows):
        test_start = current_test_end - pd.DateOffset(months=test_months) + pd.Timedelta(days=1)
        split_date = test_start
        train_start = split_date - pd.DateOffset(years=train_years)

        window = (
            {
                "TRAIN_START": train_start.strftime("%Y-%m-%d"),
                "SPLIT_DATE": split_date.strftime("%Y-%m-%d"),
                "TEST_START": test_start.strftime("%Y-%m-%d"),
                "TEST_END": current_test_end.strftime("%Y-%m-%d"),
            }
        )
        if split_date >= min_split_date:
            windows.append(window)

        current_test_end = test_start - pd.Timedelta(days=1)

    return windows[::-1]


def run_single_window(window_index, total_windows, window_cfg):
    run_env = BASE_ENV.copy()
    run_env["QUANT_TRAIN_START"] = window_cfg["TRAIN_START"]
    run_env["QUANT_SPLIT_DATE"] = window_cfg["SPLIT_DATE"]
    run_env["QUANT_TEST_START"] = window_cfg["TEST_START"]
    run_env["QUANT_TEST_END"] = window_cfg["TEST_END"]

    print(f"--- RUNNING WINDOW {window_index}/{total_windows} ---", flush=True)
    print(f"Train: {window_cfg['TRAIN_START']} -> {window_cfg['SPLIT_DATE']}", flush=True)
    print(f"Test:  {window_cfg['TEST_START']} -> {window_cfg['TEST_END']}", flush=True)

    cmd = [str(PYTHON_EXE), "-u", str(TRAIN_PY)]
    proc = subprocess.Popen(
        cmd,
        env=run_env,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)

    code = proc.wait()
    stdout_text = "".join(lines)

    if code != 0:
        run_log = OUT_DIR / f"wfa_window_{window_index:02d}_failed.log"
        run_log.write_text(stdout_text, encoding="utf-8")
        raise RuntimeError(f"Window {window_index} failed with exit code {code}. Log: {run_log}")

    metrics = _parse_run_output(stdout_text)
    metrics.update(
        {
            "window_index": window_index,
            "train_start": window_cfg["TRAIN_START"],
            "split_date": window_cfg["SPLIT_DATE"],
            "test_start": window_cfg["TEST_START"],
            "test_end": window_cfg["TEST_END"],
        }
    )

    run_log = OUT_DIR / f"wfa_window_{window_index:02d}.log"
    run_log.write_text(stdout_text, encoding="utf-8")

    print(
        "[SUMMARY] "
        f"Final={metrics['real_final']:.2f}, Sharpe={metrics['real_sharpe']:.3f}, "
        f"DD={metrics['real_max_dd_pct']:.2f}%, Trades={metrics['real_trades_executed']}, "
        f"CashDays={metrics['days_in_cash_mode']}, Tier1={metrics['days_circuit_breaker_tier1']}, "
        f"Tier2={metrics['days_circuit_breaker_tier2']}",
        flush=True,
    )
    print(f"[OK] Window {window_index} complete.\\n", flush=True)

    return metrics


def write_summary_artifacts(rows):
    json_path = OUT_DIR / "wfa_summary.json"
    csv_path = OUT_DIR / "wfa_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path


def print_aggregate(rows):
    if not rows:
        print("No completed windows.")
        return

    sharpe_values = [r["real_sharpe"] for r in rows]
    dd_values = [r["real_max_dd_pct"] for r in rows]
    final_values = [r["real_final"] for r in rows]
    cash_days = [r["days_in_cash_mode"] or 0 for r in rows]
    tier2_days = [r["days_circuit_breaker_tier2"] or 0 for r in rows]

    print("\\n========================================")
    print("WFA AGGREGATE SCORECARD")
    print("========================================")
    print(f"Windows completed: {len(rows)}")
    print(f"Sharpe mean/min/max: {pd.Series(sharpe_values).mean():.3f} / {min(sharpe_values):.3f} / {max(sharpe_values):.3f}")
    print(f"MaxDD mean/worst: {pd.Series(dd_values).mean():.2f}% / {min(dd_values):.2f}%")
    print(f"Final value mean/min/max: {pd.Series(final_values).mean():.2f} / {min(final_values):.2f} / {max(final_values):.2f}")
    print(f"Cash-mode days mean/max: {pd.Series(cash_days).mean():.1f} / {max(cash_days)}")
    print(f"Tier2 days mean/max: {pd.Series(tier2_days).mean():.1f} / {max(tier2_days)}")
    print(f"Signal reversal windows: {sum(1 for r in rows if r.get('signal_reversal_detected'))}")


def run_wfa():
    windows = generate_windows(num_windows=10, train_years=2, test_months=6)

    print("========================================")
    print("INITIATING 10-WINDOW WALK-FORWARD STRESS TEST")
    print("========================================")
    print("")
    print(f"Feasible windows after min-split filter: {len(windows)}", flush=True)

    results = []
    total = len(windows)
    for idx, window in enumerate(windows, start=1):
        metrics = run_single_window(idx, total, window)
        results.append(metrics)

    json_path, csv_path = write_summary_artifacts(results)
    print_aggregate(results)
    print(f"\\nSummary JSON: {json_path}")
    print(f"Summary CSV:  {csv_path}")


if __name__ == "__main__":
    run_wfa()
