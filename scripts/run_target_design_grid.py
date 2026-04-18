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
DETAIL_CSV = OUT_DIR / "target_design_grid_10window_detail.csv"
SUMMARY_CSV = OUT_DIR / "target_design_grid_10window_summary.csv"

REAL_BLOCK = re.compile(
    r"REAL MODEL\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+).*?Trade Days:\s*(\d+)\s+Trades Executed:\s*(\d+)",
    flags=re.S,
)
RANDOM_BLOCK = re.compile(
    r"RANDOM BASELINE\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+).*?Trade Days:\s*(\d+)\s+Trades Executed:\s*(\d+)",
    flags=re.S,
)
IC_MEAN_RE = re.compile(r"Signal quality summary:\s*\{.*?'ic_mean':\s*([-0-9.]+)", flags=re.S)
WF_PASS_RE = re.compile(r"Walk-forward governance:.*?pass=\s*(True|False)", flags=re.S)
ROB_PASS_RE = re.compile(r"Robustness sensitivity:\s*\{.*?'overall_pass':\s*(True|False)", flags=re.S)


def load_windows(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"start": row["start"], "end": row["end"]})
    return rows


def parse_metrics(stdout_text: str, block: re.Pattern):
    match = block.search(stdout_text)
    if not match:
        return None
    return {
        "final": float(match.group(1)),
        "sharpe": float(match.group(2)),
        "trade_days": int(match.group(3)),
        "trades": int(match.group(4)),
    }


def parse_ic_mean(stdout_text: str):
    match = IC_MEAN_RE.search(stdout_text)
    return float(match.group(1)) if match else None


def parse_bool(stdout_text: str, pattern: re.Pattern):
    match = pattern.search(stdout_text)
    if not match:
        return None
    return match.group(1) == "True"


def config_label(cfg):
    threshold = f"{cfg['abs_threshold']:.4f}".rstrip("0").rstrip(".")
    return f"h{cfg['horizon_days']}_thr{threshold}_cost{cfg['cost_bps']}"


def build_configs():
    configs = []
    for horizon_days, abs_threshold, cost_bps in product([1, 3, 5], [0.0005, 0.0010, 0.0020], [5, 7, 10]):
        configs.append(
            {
                "label": config_label(
                    {
                        "horizon_days": horizon_days,
                        "abs_threshold": abs_threshold,
                        "cost_bps": cost_bps,
                    }
                ),
                "horizon_days": horizon_days,
                "abs_threshold": abs_threshold,
                "cost_bps": cost_bps,
            }
        )
    return configs


def run_window(py_exe: str, start: str, end: str, env: dict, timeout_sec: int, backtest_only: bool):
    cmd = [py_exe, str(TRAIN_PY)]
    if backtest_only:
        cmd.append("--backtest-only")
    cmd.extend(["--start", start, "--end", end])
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
        return None, None, None, None, (exc.stdout or ""), f"timeout_after_{timeout_sec}s"

    if proc.returncode != 0:
        return None, None, None, None, proc.stdout, proc.stderr

    real = parse_metrics(proc.stdout, REAL_BLOCK)
    rand = parse_metrics(proc.stdout, RANDOM_BLOCK)
    ic_mean = parse_ic_mean(proc.stdout)
    wf_pass = parse_bool(proc.stdout, WF_PASS_RE)
    robustness_pass = parse_bool(proc.stdout, ROB_PASS_RE)
    return real, rand, ic_mean, wf_pass, robustness_pass, proc.stdout, proc.stderr


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4 target/label design grid runner")
    parser.add_argument("--max-configs", type=int, default=0, help="Limit number of configs (0 = all)")
    parser.add_argument("--max-windows", type=int, default=0, help="Limit number of windows (0 = all)")
    parser.add_argument("--timeout-sec", type=int, default=1800, help="Per-window timeout in seconds")
    parser.add_argument("--backtest-only", action="store_true", help="Reuse existing checkpoint for smoke testing")
    return parser.parse_args()


def main():
    args = parse_args()
    if not WINDOWS_CSV.exists():
        raise SystemExit(f"Missing windows file: {WINDOWS_CSV}")

    py_exe = sys.executable
    windows = load_windows(WINDOWS_CSV)
    configs = build_configs()

    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []

    for cfg_idx, cfg in enumerate(configs, start=1):
        label = cfg["label"]
        real_finals = []
        real_sharpes = []
        final_edges = []
        sharpe_edges = []
        ic_means = []
        total_trades = 0
        wf_pass_count = 0
        robustness_pass_count = 0

        print(f"[{cfg_idx}/{len(configs)}] {label}")

        for w_idx, w in enumerate(windows, start=1):
            print(f"  window [{w_idx}/{len(windows)}] {w['start']} -> {w['end']}")
            env = dict(os.environ)
            env.update(FROZEN_OBJECTIVE_V1_PARAMS)
            env.update(
                {
                    "QUANT_OBJECTIVE_FREEZE_ALLOW_TARGET_GRID": "1",
                    "QUANT_TARGET_HORIZON_DAYS": str(cfg["horizon_days"]),
                    "QUANT_HOLDING_DAYS": str(cfg["horizon_days"]),
                    "QUANT_TARGET_COST_BPS": str(cfg["cost_bps"]),
                    "QUANT_TARGET_ABS_THRESHOLD": str(cfg["abs_threshold"]),
                    "QUANT_SIGNAL_MODE": "model",
                    "QUANT_CARRY_FORWARD_OPEN": "1",
                    "QUANT_ADAPTIVE_REGIME_LOGIC": "1",
                    "QUANT_REGIME_SAFETY_ENABLED": "0",
                    "QUANT_FEATURE_SIGNAL_ENABLED": "0",
                    "QUANT_HYBRID_ENSEMBLE_ENABLED": "0",
                    "QUANT_XGBOOST_ENABLED": "0",
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
                    "QUANT_MAX_NEW_POSITIONS_PER_DAY": "3",
                    "QUANT_TURNOVER_COOLDOWN_DAYS": "2",
                    "QUANT_SIGNAL_INVERSION_MODE": "none",
                    "QUANT_AUTO_INVERT_SIGNAL": "0",
                    "QUANT_ENSEMBLE_SEEDS": "42",
                }
            )

            real, rand, ic_mean, wf_pass, robustness_pass, _stdout, stderr = run_window(
                py_exe=py_exe,
                start=w["start"],
                end=w["end"],
                env=env,
                timeout_sec=args.timeout_sec,
                backtest_only=args.backtest_only,
            )

            if real is None or rand is None:
                detail_rows.append(
                    {
                        "config": label,
                        "horizon_days": cfg["horizon_days"],
                        "abs_threshold": cfg["abs_threshold"],
                        "cost_bps": cfg["cost_bps"],
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
            final_edges.append(final_edge)
            sharpe_edges.append(sharpe_edge)
            total_trades += int(real["trades"])
            if ic_mean is not None:
                ic_means.append(ic_mean)
            if wf_pass:
                wf_pass_count += 1
            if robustness_pass:
                robustness_pass_count += 1

            detail_rows.append(
                {
                    "config": label,
                    "horizon_days": cfg["horizon_days"],
                    "abs_threshold": cfg["abs_threshold"],
                    "cost_bps": cfg["cost_bps"],
                    "start": w["start"],
                    "end": w["end"],
                    "status": "ok",
                    "real_final": real["final"],
                    "real_sharpe": real["sharpe"],
                    "real_trade_days": real["trade_days"],
                    "real_trades": real["trades"],
                    "rand_final": rand["final"],
                    "rand_sharpe": rand["sharpe"],
                    "rand_trade_days": rand["trade_days"],
                    "rand_trades": rand["trades"],
                    "final_edge": final_edge,
                    "sharpe_edge": sharpe_edge,
                    "ic_mean": ic_mean,
                    "wf_pass": wf_pass,
                    "robustness_pass": robustness_pass,
                }
            )

        if real_finals:
            summary_rows.append(
                {
                    "config": label,
                    "horizon_days": cfg["horizon_days"],
                    "abs_threshold": cfg["abs_threshold"],
                    "cost_bps": cfg["cost_bps"],
                    "windows_ok": len(real_finals),
                    "mean_real_final": statistics.mean(real_finals),
                    "mean_real_sharpe": statistics.mean(real_sharpes),
                    "mean_final_edge": statistics.mean(final_edges),
                    "median_final_edge": statistics.median(final_edges),
                    "mean_sharpe_edge": statistics.mean(sharpe_edges),
                    "median_sharpe_edge": statistics.median(sharpe_edges),
                    "mean_ic_mean": statistics.mean(ic_means) if ic_means else None,
                    "win_final_edge_windows": sum(1 for x in final_edges if x > 0),
                    "win_sharpe_edge_windows": sum(1 for x in sharpe_edges if x > 0),
                    "wf_pass_windows": wf_pass_count,
                    "robustness_pass_windows": robustness_pass_count,
                    "total_trades": total_trades,
                }
            )

    detail_fields = [
        "config",
        "horizon_days",
        "abs_threshold",
        "cost_bps",
        "start",
        "end",
        "status",
        "real_final",
        "real_sharpe",
        "real_trade_days",
        "real_trades",
        "rand_final",
        "rand_sharpe",
        "rand_trade_days",
        "rand_trades",
        "final_edge",
        "sharpe_edge",
        "ic_mean",
        "wf_pass",
        "robustness_pass",
        "error",
    ]
    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (row["mean_sharpe_edge"], row["mean_final_edge"], row["mean_ic_mean"] or -999.0),
        reverse=True,
    )
    summary_fields = [
        "config",
        "horizon_days",
        "abs_threshold",
        "cost_bps",
        "windows_ok",
        "mean_real_final",
        "mean_real_sharpe",
        "mean_final_edge",
        "median_final_edge",
        "mean_sharpe_edge",
        "median_sharpe_edge",
        "mean_ic_mean",
        "win_final_edge_windows",
        "win_sharpe_edge_windows",
        "wf_pass_windows",
        "robustness_pass_windows",
        "total_trades",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Target design grid done")
    print(f"Detail CSV: {DETAIL_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    if summary_rows:
        print("Top config:", summary_rows[0])


if __name__ == "__main__":
    main()