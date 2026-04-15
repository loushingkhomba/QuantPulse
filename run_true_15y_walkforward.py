import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs" / "true_walkforward_15y"
CACHE_PATH = ROOT / "data" / "cache" / "zerodha_dataset_15y.pkl"
INITIAL_CAPITAL = 10000.0
MIN_TRAIN_YEARS = 2


def extract_metric(log_text: str, section: str, key: str):
    pattern = rf"{re.escape(section)}.*?{re.escape(key)}\s*([-0-9.]+)"
    match = re.search(pattern, log_text, flags=re.S)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_run_metrics(log_text: str):
    model_final = extract_metric(log_text, "REAL MODEL", "Final Value:")
    random_final = extract_metric(log_text, "RANDOM BASELINE", "Final Value:")
    nifty_final = extract_metric(log_text, "NIFTY BUY & HOLD", "Final Value:")

    return {
        "model_final": model_final,
        "model_sharpe": extract_metric(log_text, "REAL MODEL", "Sharpe:"),
        "model_drawdown_pct": extract_metric(log_text, "REAL MODEL", "Max Drawdown:"),
        "model_annualized_return_pct": extract_metric(log_text, "REAL MODEL", "Annualized Return:"),
        "random_final": random_final,
        "random_sharpe": extract_metric(log_text, "RANDOM BASELINE", "Sharpe:"),
        "nifty_final": nifty_final,
        "inverted_final": extract_metric(log_text, "INVERTED SIGNAL (DIAGNOSTIC)", "Final Value:"),
        "inverted_sharpe": extract_metric(log_text, "INVERTED SIGNAL (DIAGNOSTIC)", "Sharpe:"),
    }


def build_windows(min_date: pd.Timestamp, max_date: pd.Timestamp):
    start_year = int(min_date.year)
    end_year = int(max_date.year)
    windows = []

    # Expanding window: train from first year, test one calendar year at a time.
    for test_year in range(start_year + MIN_TRAIN_YEARS, end_year + 1):
        test_start = pd.Timestamp(test_year, 1, 1)
        test_end = pd.Timestamp(test_year, 12, 31)
        if test_start > max_date:
            continue
        if test_end > max_date:
            test_end = max_date

        windows.append((
            test_year,
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

    return windows


def run_window(test_year: int, start: str, end: str):
    tag = f"{test_year}"
    log_path = LOG_DIR / f"window_{tag}.log"
    metrics_path = LOG_DIR / f"window_{tag}_metrics.json"

    env = os.environ.copy()
    env["QUANT_HISTORY_YEARS"] = "15"
    env["QUANT_USE_DATA_CACHE"] = "1"
    env["QUANT_REFRESH_DATA"] = "0"
    env["QUANT_SPLIT_DATE"] = start
    env["QUANT_TEST_START"] = start
    env["QUANT_TEST_END"] = end
    env["QUANT_METRICS_OUT"] = str(metrics_path)

    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--start",
        start,
        "--end",
        end,
    ]

    print(f"\n=== Walk-forward window {tag}: {start} to {end} ===")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"Window {tag} failed. See {log_path}")

    metrics = parse_run_metrics(combined)
    standard_metrics = {}
    if metrics_path.exists():
        standard_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    signal_quality = standard_metrics.get("signal_quality", {})
    return_quality = standard_metrics.get("return_quality", {})
    metrics.update({
        "year": test_year,
        "start": start,
        "end": end,
        "log_file": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "metrics_file": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
        "ic_mean": signal_quality.get("information_coefficient", {}).get("mean_daily_ic"),
        "spread_mean": signal_quality.get("confidence_spread", {}).get("mean_rank1_rank10_spread"),
        "avg_alpha_per_trade": return_quality.get("alpha_vs_nifty_per_trade", {}).get("average_alpha"),
        "profit_factor": return_quality.get("profit_factor"),
    })
    return metrics


def safe_return(final_value):
    if final_value is None:
        return None
    return (final_value / INITIAL_CAPITAL) - 1.0


def main():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Expected cache file not found: {CACHE_PATH}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(CACHE_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    min_date = df["Date"].min().normalize()
    max_date = df["Date"].max().normalize()

    windows = build_windows(min_date, max_date)
    if not windows:
        raise RuntimeError("No valid walk-forward windows generated.")

    print("Using cached data:", CACHE_PATH)
    print("Data range:", str(min_date.date()), "to", str(max_date.date()))
    print("Windows:", len(windows), "(expanding train, yearly OOS)")

    results = []
    for year, start, end in windows:
        results.append(run_window(year, start, end))

    model_compound = 1.0
    random_compound = 1.0
    nifty_compound = 1.0

    valid_model_windows = 0
    for row in results:
        model_r = safe_return(row.get("model_final"))
        random_r = safe_return(row.get("random_final"))
        nifty_r = safe_return(row.get("nifty_final"))

        row["model_window_return_pct"] = None if model_r is None else model_r * 100.0
        row["random_window_return_pct"] = None if random_r is None else random_r * 100.0
        row["nifty_window_return_pct"] = None if nifty_r is None else nifty_r * 100.0

        if model_r is not None:
            model_compound *= (1.0 + model_r)
            valid_model_windows += 1
        if random_r is not None:
            random_compound *= (1.0 + random_r)
        if nifty_r is not None:
            nifty_compound *= (1.0 + nifty_r)

    overall = {
        "windows": len(results),
        "valid_model_windows": valid_model_windows,
        "model_final_compounded": INITIAL_CAPITAL * model_compound,
        "random_final_compounded": INITIAL_CAPITAL * random_compound,
        "nifty_final_compounded": INITIAL_CAPITAL * nifty_compound,
        "model_total_return_pct": (model_compound - 1.0) * 100.0,
        "random_total_return_pct": (random_compound - 1.0) * 100.0,
        "nifty_total_return_pct": (nifty_compound - 1.0) * 100.0,
        "model_avg_sharpe": float(pd.Series([r.get("model_sharpe") for r in results], dtype="float64").dropna().mean()),
        "random_avg_sharpe": float(pd.Series([r.get("random_sharpe") for r in results], dtype="float64").dropna().mean()),
        "model_avg_ic": float(pd.Series([r.get("ic_mean") for r in results], dtype="float64").dropna().mean()),
        "model_avg_alpha_per_trade": float(pd.Series([r.get("avg_alpha_per_trade") for r in results], dtype="float64").dropna().mean()),
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_path": str(CACHE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "data_range": {
            "start": min_date.strftime("%Y-%m-%d"),
            "end": max_date.strftime("%Y-%m-%d"),
        },
        "walkforward_config": {
            "mode": "expanding_train_yearly_test",
            "min_train_years": MIN_TRAIN_YEARS,
            "test_horizon": "1_year",
        },
        "overall": overall,
        "windows": results,
    }

    out_json = LOG_DIR / "true_15y_walkforward_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("# True 15-Year Walk-Forward Report")
    lines.append("")
    lines.append(f"- Data range: {report['data_range']['start']} to {report['data_range']['end']}")
    lines.append(f"- Windows: {overall['windows']} (min train years: {MIN_TRAIN_YEARS})")
    lines.append(f"- Model compounded final: {overall['model_final_compounded']:.2f}")
    lines.append(f"- Random compounded final: {overall['random_final_compounded']:.2f}")
    lines.append(f"- Nifty compounded final: {overall['nifty_final_compounded']:.2f}")
    lines.append(f"- Model total return: {overall['model_total_return_pct']:.2f}%")
    lines.append(f"- Random total return: {overall['random_total_return_pct']:.2f}%")
    lines.append(f"- Nifty total return: {overall['nifty_total_return_pct']:.2f}%")
    lines.append(f"- Model avg Sharpe across windows: {overall['model_avg_sharpe']:.3f}")
    lines.append(f"- Random avg Sharpe across windows: {overall['random_avg_sharpe']:.3f}")
    lines.append(f"- Model avg IC across windows: {overall['model_avg_ic']:.5f}")
    lines.append(f"- Model avg alpha/trade across windows: {overall['model_avg_alpha_per_trade']:.5f}")
    lines.append("")
    lines.append("| Year | Test Start | Test End | Model Final | Model Sharpe | IC Mean | Avg Alpha | Profit Factor | Model DD % | Random Final | Nifty Final |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for row in results:
        lines.append(
            "| {year} | {start} | {end} | {model_final:.2f} | {model_sharpe:.3f} | {ic_mean:.5f} | {alpha:.5f} | {profit_factor:.3f} | {model_dd:.2f} | {random_final:.2f} | {nifty_final:.2f} |".format(
                year=row.get("year"),
                start=row.get("start"),
                end=row.get("end"),
                model_final=row.get("model_final") or float("nan"),
                model_sharpe=row.get("model_sharpe") or float("nan"),
                ic_mean=row.get("ic_mean") or float("nan"),
                alpha=row.get("avg_alpha_per_trade") or float("nan"),
                profit_factor=row.get("profit_factor") or float("nan"),
                model_dd=row.get("model_drawdown_pct") or float("nan"),
                random_final=row.get("random_final") or float("nan"),
                nifty_final=row.get("nifty_final") or float("nan"),
            )
        )

    out_md = LOG_DIR / "true_15y_walkforward_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(out_json)
    print(out_md)
    print("\nOverall summary:")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
