import csv
import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
WINDOWS_CSV = ROOT / "logs" / "standard_metrics" / "ten_window_10d_regime_safety_vs_prior_relaxed.csv"
OUT_DIR = ROOT / "logs" / "standard_metrics"
DETAIL_CSV = OUT_DIR / "regime_safety_sweep_10window_detail.csv"
SUMMARY_CSV = OUT_DIR / "regime_safety_sweep_10window_summary.csv"


REAL_BLOCK = re.compile(
    r"REAL MODEL\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+).*?Trade Days:\s*(\d+)\s+Trades Executed:\s*(\d+)",
    flags=re.S,
)


def load_windows(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"start": row["start"], "end": row["end"]})
    return rows


def load_prior(path: Path):
    out = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["start"], row["end"])
            out[key] = {
                "prior_final": float(row["prior_relaxed_final"]),
                "prior_sharpe": float(row["prior_relaxed_sharpe"]),
                "prior_trades": int(float(row["prior_relaxed_trades"])),
            }
    return out


def parse_real_metrics(stdout_text: str):
    m = REAL_BLOCK.search(stdout_text)
    if not m:
        return None
    return {
        "final": float(m.group(1)),
        "sharpe": float(m.group(2)),
        "trade_days": int(m.group(3)),
        "trades": int(m.group(4)),
    }


def run_window(py_exe: str, start: str, end: str, env: dict, timeout_sec: int):
    cmd = [
        py_exe,
        str(TRAIN_PY),
        "--backtest-only",
        "--start",
        start,
        "--end",
        end,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(60, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        return None, (exc.stdout or ""), f"timeout_after_{timeout_sec}s"
    if proc.returncode != 0:
        return None, proc.stdout, proc.stderr
    metrics = parse_real_metrics(proc.stdout)
    return metrics, proc.stdout, proc.stderr


def config_label(c):
    return (
        f"pct{c['strict_pct']:.2f}_"
        f"bad{c['strict_bad']:.3f}_neu{c['strict_neutral']:.3f}_trd{c['strict_trending']:.3f}_"
        f"kelly{c['strict_kelly']:.2f}"
    )


def build_configs():
    return [
        {
            "strict_pct": 0.75,
            "strict_bad": 0.011,
            "strict_neutral": 0.009,
            "strict_trending": 0.007,
            "strict_kelly": 0.51,
        },
        {
            "strict_pct": 0.75,
            "strict_bad": 0.012,
            "strict_neutral": 0.010,
            "strict_trending": 0.008,
            "strict_kelly": 0.52,
        },
        {
            "strict_pct": 0.80,
            "strict_bad": 0.012,
            "strict_neutral": 0.010,
            "strict_trending": 0.008,
            "strict_kelly": 0.52,
        },
        {
            "strict_pct": 0.80,
            "strict_bad": 0.013,
            "strict_neutral": 0.011,
            "strict_trending": 0.009,
            "strict_kelly": 0.53,
        },
        {
            "strict_pct": 0.85,
            "strict_bad": 0.012,
            "strict_neutral": 0.010,
            "strict_trending": 0.008,
            "strict_kelly": 0.53,
        },
        {
            "strict_pct": 0.85,
            "strict_bad": 0.014,
            "strict_neutral": 0.012,
            "strict_trending": 0.010,
            "strict_kelly": 0.54,
        },
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Regime safety sweep runner")
    p.add_argument("--max-configs", type=int, default=0, help="Limit number of configs (0 = all)")
    p.add_argument("--max-windows", type=int, default=0, help="Limit number of windows (0 = all)")
    p.add_argument("--timeout-sec", type=int, default=600, help="Per-window timeout in seconds")
    return p.parse_args()


def main():
    args = parse_args()
    if not WINDOWS_CSV.exists():
        raise SystemExit(f"Missing windows file: {WINDOWS_CSV}")

    py_exe = sys.executable
    windows = load_windows(WINDOWS_CSV)
    prior = load_prior(WINDOWS_CSV)
    configs = build_configs()

    if args.max_windows and args.max_windows > 0:
        windows = windows[: args.max_windows]
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []

    for cfg_idx, cfg in enumerate(configs, start=1):
        label = config_label(cfg)
        finals = []
        sharpes = []
        trades_total = 0
        better_final = 0
        better_sharpe = 0

        print(f"[{cfg_idx}/{len(configs)}] {label}")

        for w_idx, w in enumerate(windows, start=1):
            print(f"  window [{w_idx}/{len(windows)}] {w['start']} -> {w['end']}")
            env = {k: v for k, v in os.environ.items()}
            env.update(
                {
                    "QUANT_SIGNAL_MODE": "model",
                    "QUANT_ENSEMBLE_SEEDS": "42",
                    "QUANT_OBJECTIVE_MODE": "ranking",
                    "QUANT_SIMPLE_HIDDEN_SIZE": "64",
                    "QUANT_TARGET_MODE": "absolute",
                    "QUANT_TARGET_HORIZON_DAYS": "1",
                    "QUANT_HOLDING_DAYS": "1",
                    "QUANT_TARGET_ABS_THRESHOLD": "0.001",
                    "QUANT_TARGET_COST_BPS": "7",
                    "QUANT_FEATURE_SIGNAL_ENABLED": "0",
                    "QUANT_HYBRID_ENSEMBLE_ENABLED": "1",
                    "QUANT_XGBOOST_ENABLED": "1",
                    "QUANT_CARRY_FORWARD_OPEN": "1",
                    "QUANT_ADAPTIVE_REGIME_LOGIC": "1",
                    "QUANT_REGIME_SAFETY_ENABLED": "1",
                    "QUANT_REGIME_SAFETY_STRICT_PCT": str(cfg["strict_pct"]),
                    "QUANT_RANK_THRESHOLD_BAD": "0.65",
                    "QUANT_RANK_THRESHOLD_NEUTRAL": "0.60",
                    "QUANT_RANK_THRESHOLD_TRENDING": "0.68",
                    "QUANT_MIN_TOP_CONFIDENCE": "0.480",
                    "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.485",
                    "QUANT_MIN_TOP_CONFIDENCE_TRENDING": "0.495",
                    "QUANT_TOP_K": "3",
                    "QUANT_MAX_NEW_POSITIONS_PER_DAY": "3",
                    "QUANT_TURNOVER_COOLDOWN_DAYS": "2",
                    "QUANT_SIGNAL_SPREAD_BAD": "0.006",
                    "QUANT_SIGNAL_SPREAD_NEUTRAL": "0.005",
                    "QUANT_SIGNAL_SPREAD_TRENDING": "0.004",
                    "QUANT_STRICT_SIGNAL_SPREAD_BAD": str(cfg["strict_bad"]),
                    "QUANT_STRICT_SIGNAL_SPREAD_NEUTRAL": str(cfg["strict_neutral"]),
                    "QUANT_STRICT_SIGNAL_SPREAD_TRENDING": str(cfg["strict_trending"]),
                    "QUANT_KELLY_LITE_ENABLED": "1",
                    "QUANT_KELLY_BREAKEVEN": "0.50",
                    "QUANT_KELLY_BREAKEVEN_STRICT": str(cfg["strict_kelly"]),
                }
            )

            metrics, _stdout, stderr = run_window(py_exe, w["start"], w["end"], env, timeout_sec=args.timeout_sec)
            if metrics is None:
                detail_rows.append(
                    {
                        "config": label,
                        "start": w["start"],
                        "end": w["end"],
                        "status": "error",
                        "error": (stderr or "parse_failed").strip()[:400],
                    }
                )
                continue

            key = (w["start"], w["end"])
            prior_row = prior.get(key)
            prior_final = prior_row["prior_final"] if prior_row else float("nan")
            prior_sharpe = prior_row["prior_sharpe"] if prior_row else float("nan")

            final_delta = metrics["final"] - prior_final if prior_row else float("nan")
            sharpe_delta = metrics["sharpe"] - prior_sharpe if prior_row else float("nan")

            if prior_row and metrics["final"] > prior_final:
                better_final += 1
            if prior_row and metrics["sharpe"] > prior_sharpe:
                better_sharpe += 1

            finals.append(metrics["final"])
            sharpes.append(metrics["sharpe"])
            trades_total += metrics["trades"]

            detail_rows.append(
                {
                    "config": label,
                    "start": w["start"],
                    "end": w["end"],
                    "status": "ok",
                    "final": metrics["final"],
                    "sharpe": metrics["sharpe"],
                    "trade_days": metrics["trade_days"],
                    "trades": metrics["trades"],
                    "prior_final": prior_final,
                    "prior_sharpe": prior_sharpe,
                    "final_delta": final_delta,
                    "sharpe_delta": sharpe_delta,
                }
            )

        if finals:
            summary_rows.append(
                {
                    "config": label,
                    "windows_ok": len(finals),
                    "mean_final": statistics.mean(finals),
                    "median_final": statistics.median(finals),
                    "mean_sharpe": statistics.mean(sharpes),
                    "median_sharpe": statistics.median(sharpes),
                    "total_trades": trades_total,
                    "better_final_windows": better_final,
                    "better_sharpe_windows": better_sharpe,
                }
            )

    detail_fields = [
        "config",
        "start",
        "end",
        "status",
        "final",
        "sharpe",
        "trade_days",
        "trades",
        "prior_final",
        "prior_sharpe",
        "final_delta",
        "sharpe_delta",
        "error",
    ]
    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    summary_rows = sorted(summary_rows, key=lambda r: (r["mean_sharpe"], r["mean_final"]), reverse=True)
    summary_fields = [
        "config",
        "windows_ok",
        "mean_final",
        "median_final",
        "mean_sharpe",
        "median_sharpe",
        "total_trades",
        "better_final_windows",
        "better_sharpe_windows",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Sweep done")
    print(f"Detail CSV: {DETAIL_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    if summary_rows:
        top = summary_rows[0]
        print("Top config:", top)


if __name__ == "__main__":
    main()
