import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import FROZEN_OBJECTIVE_V1_PARAMS


TRAIN_PY = ROOT / "train.py"
WINDOWS_CSV = ROOT / "logs" / "standard_metrics" / "ten_window_10d_regime_safety_vs_prior_relaxed.csv"
OUT_DIR = ROOT / "logs" / "standard_metrics"
DETAIL_CSV = OUT_DIR / "ablation_matrix_10window_detail.csv"
SUMMARY_CSV = OUT_DIR / "ablation_matrix_10window_summary.csv"

REAL_BLOCK = re.compile(
    r"REAL MODEL\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+).*?Trade Days:\s*(\d+)\s+Trades Executed:\s*(\d+)",
    flags=re.S,
)
RANDOM_BLOCK = re.compile(
    r"RANDOM BASELINE\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+).*?Trade Days:\s*(\d+)\s+Trades Executed:\s*(\d+)",
    flags=re.S,
)


def load_windows(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"start": row["start"], "end": row["end"]})
    return rows


def parse_metrics(stdout_text: str, block: re.Pattern):
    m = block.search(stdout_text)
    if not m:
        return None
    return {
        "final": float(m.group(1)),
        "sharpe": float(m.group(2)),
        "trade_days": int(m.group(3)),
        "trades": int(m.group(4)),
    }


def run_window(py_exe: str, start: str, end: str, env: dict, timeout_sec: int):
    cmd = [py_exe, str(TRAIN_PY), "--backtest-only", "--start", start, "--end", end]
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
        return None, None, (exc.stdout or ""), f"timeout_after_{timeout_sec}s"

    if proc.returncode != 0:
        return None, None, proc.stdout, proc.stderr

    real = parse_metrics(proc.stdout, REAL_BLOCK)
    rand = parse_metrics(proc.stdout, RANDOM_BLOCK)
    return real, rand, proc.stdout, proc.stderr


def build_configs():
    configs = []
    for safety, feature, hybrid in product([0, 1], [0, 1], [0, 1]):
        label = f"safety{bool(safety)}_feature{bool(feature)}_hybrid{bool(hybrid)}"
        cfg = {
            "label": label,
            "safety": str(safety),
            "feature": str(feature),
            "hybrid": str(hybrid),
        }
        configs.append(cfg)
    return configs


def parse_args():
    p = argparse.ArgumentParser(description="Ablation matrix runner (safety/feature/hybrid)")
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
    configs = build_configs()

    if args.max_windows and args.max_windows > 0:
        windows = windows[: args.max_windows]
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []

    for cfg_idx, cfg in enumerate(configs, start=1):
        label = cfg["label"]
        real_finals = []
        real_sharpes = []
        edge_finals = []
        edge_sharpes = []
        total_trades = 0

        print(f"[{cfg_idx}/{len(configs)}] {label}")

        for w_idx, w in enumerate(windows, start=1):
            print(f"  window [{w_idx}/{len(windows)}] {w['start']} -> {w['end']}")
            env = {k: v for k, v in os.environ.items()}

            env.update(FROZEN_OBJECTIVE_V1_PARAMS)
            env.update(
                {
                    "QUANT_SIGNAL_MODE": "model",
                    "QUANT_CARRY_FORWARD_OPEN": "1",
                    "QUANT_ADAPTIVE_REGIME_LOGIC": "1",
                    "QUANT_TOP_K": "3",
                    "QUANT_MIN_TOP_CONFIDENCE": "0.480",
                    "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.485",
                    "QUANT_MIN_TOP_CONFIDENCE_TRENDING": "0.495",
                    "QUANT_RANK_THRESHOLD_BAD": "0.65",
                    "QUANT_RANK_THRESHOLD_NEUTRAL": "0.60",
                    "QUANT_RANK_THRESHOLD_TRENDING": "0.68",
                    "QUANT_SIGNAL_SPREAD_BAD": "0.006",
                    "QUANT_SIGNAL_SPREAD_NEUTRAL": "0.005",
                    "QUANT_SIGNAL_SPREAD_TRENDING": "0.004",
                    "QUANT_KELLY_LITE_ENABLED": "1",
                    "QUANT_KELLY_BREAKEVEN": "0.50",
                    "QUANT_MAX_NEW_POSITIONS_PER_DAY": "3",
                    "QUANT_TURNOVER_COOLDOWN_DAYS": "2",
                    "QUANT_REGIME_SAFETY_ENABLED": cfg["safety"],
                    "QUANT_FEATURE_SIGNAL_ENABLED": cfg["feature"],
                    "QUANT_HYBRID_ENSEMBLE_ENABLED": cfg["hybrid"],
                    "QUANT_XGBOOST_ENABLED": "1" if cfg["hybrid"] == "1" else "0",
                }
            )

            real, rand, _stdout, stderr = run_window(py_exe, w["start"], w["end"], env, timeout_sec=args.timeout_sec)
            if real is None or rand is None:
                detail_rows.append(
                    {
                        "config": label,
                        "start": w["start"],
                        "end": w["end"],
                        "status": "error",
                        "error": (stderr or "parse_failed").strip()[:500],
                    }
                )
                continue

            final_edge = real["final"] - rand["final"]
            sharpe_edge = real["sharpe"] - rand["sharpe"]

            real_finals.append(real["final"])
            real_sharpes.append(real["sharpe"])
            edge_finals.append(final_edge)
            edge_sharpes.append(sharpe_edge)
            total_trades += int(real["trades"])

            detail_rows.append(
                {
                    "config": label,
                    "start": w["start"],
                    "end": w["end"],
                    "status": "ok",
                    "real_final": real["final"],
                    "real_sharpe": real["sharpe"],
                    "real_trades": real["trades"],
                    "rand_final": rand["final"],
                    "rand_sharpe": rand["sharpe"],
                    "final_edge": final_edge,
                    "sharpe_edge": sharpe_edge,
                }
            )

        if real_finals:
            summary_rows.append(
                {
                    "config": label,
                    "windows_ok": len(real_finals),
                    "mean_real_final": statistics.mean(real_finals),
                    "mean_real_sharpe": statistics.mean(real_sharpes),
                    "mean_final_edge": statistics.mean(edge_finals),
                    "median_final_edge": statistics.median(edge_finals),
                    "mean_sharpe_edge": statistics.mean(edge_sharpes),
                    "median_sharpe_edge": statistics.median(edge_sharpes),
                    "win_final_edge_windows": int(sum(1 for x in edge_finals if x > 0)),
                    "win_sharpe_edge_windows": int(sum(1 for x in edge_sharpes if x > 0)),
                    "total_trades": total_trades,
                }
            )

    detail_fields = [
        "config",
        "start",
        "end",
        "status",
        "real_final",
        "real_sharpe",
        "real_trades",
        "rand_final",
        "rand_sharpe",
        "final_edge",
        "sharpe_edge",
        "error",
    ]
    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    summary_rows = sorted(summary_rows, key=lambda r: (r["mean_sharpe_edge"], r["mean_final_edge"]), reverse=True)
    summary_fields = [
        "config",
        "windows_ok",
        "mean_real_final",
        "mean_real_sharpe",
        "mean_final_edge",
        "median_final_edge",
        "mean_sharpe_edge",
        "median_sharpe_edge",
        "win_final_edge_windows",
        "win_sharpe_edge_windows",
        "total_trades",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Ablation matrix done")
    print(f"Detail CSV: {DETAIL_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    if summary_rows:
        print("Top config:", summary_rows[0])


if __name__ == "__main__":
    main()
