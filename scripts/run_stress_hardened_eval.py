import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
WINDOWS_CSV = ROOT / "logs" / "standard_metrics" / "ten_window_10d_regime_safety_vs_prior_relaxed.csv"
OUT_DIR = ROOT / "logs" / "standard_metrics"
OUT_CSV = OUT_DIR / "stress_hardened_eval_12windows.csv"

REAL_BLOCK = re.compile(
    r"REAL MODEL\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+).*?Trade Days:\s*(\d+)\s+Trades Executed:\s*(\d+)",
    flags=re.S,
)
RANDOM_BLOCK = re.compile(
    r"RANDOM BASELINE\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+).*?Trade Days:\s*(\d+)\s+Trades Executed:\s*(\d+)",
    flags=re.S,
)
INVERTED_BLOCK = re.compile(
    r"INVERTED SIGNAL \(DIAGNOSTIC\)\s+Final Value:\s*([-0-9.]+)\s+Sharpe:\s*([-0-9.]+)",
    flags=re.S,
)
IC_RE = re.compile(r"Signal quality summary:\s*\{.*?'ic_mean':\s*([-0-9.]+)", flags=re.S)
WF_RE = re.compile(r"Walk-forward governance:.*?pass=\s*(True|False)", flags=re.S)
ROB_RE = re.compile(r"Robustness sensitivity:\s*\{.*?'overall_pass':\s*(True|False)", flags=re.S)
FREEZE_RE = re.compile(r"Objective freeze:\s*\{.*?'hash':\s*'([a-f0-9]+)'.*?'ok':\s*(True|False)", flags=re.S)


def parse_block(text, pattern):
    m = pattern.search(text)
    if not m:
        return None
    return m.groups()


def load_windows():
    windows = []
    with WINDOWS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            windows.append((row["start"], row["end"], "rolling10"))
    windows.append(("2025-09-01", "2025-09-30", "month_check"))
    windows.append(("2025-10-01", "2025-10-31", "month_check"))
    return windows


def build_env(base_env):
    env = dict(base_env)
    env.update(
        {
            "QUANT_OBJECTIVE_FREEZE_ALLOW_TARGET_GRID": "1",
            "QUANT_TARGET_HORIZON_DAYS": "3",
            "QUANT_HOLDING_DAYS": "3",
            "QUANT_TARGET_COST_BPS": "7",
            "QUANT_TARGET_ABS_THRESHOLD": "0.003",
            "QUANT_REGIME_SAFETY_ENABLED": "1",
            "QUANT_REGIME_SAFETY_STRICT_PCT": "0.85",
            "QUANT_FEATURE_SIGNAL_ENABLED": "0",
            "QUANT_HYBRID_ENSEMBLE_ENABLED": "0",
            "QUANT_XGBOOST_ENABLED": "0",
            "QUANT_MIN_TOP_CONFIDENCE": "0.480",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.485",
            "QUANT_MIN_TOP_CONFIDENCE_TRENDING": "0.495",
            "QUANT_TOP_K": "3",
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "3",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "2",
            "QUANT_SIGNAL_INVERSION_MODE": "none",
            "QUANT_AUTO_INVERT_SIGNAL": "0",
            "QUANT_ENSEMBLE_SEEDS": "42",
        }
    )
    return env


def run_one(py_exe, env, start, end):
    cmd = [py_exe, str(TRAIN_PY), "--backtest-only", "--start", start, "--end", end]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if proc.returncode != 0:
        return {"status": "error", "error": (proc.stderr or proc.stdout)[-800:]}

    text = proc.stdout
    real = parse_block(text, REAL_BLOCK)
    rand = parse_block(text, RANDOM_BLOCK)
    inv = parse_block(text, INVERTED_BLOCK)
    ic = parse_block(text, IC_RE)
    wf = parse_block(text, WF_RE)
    rob = parse_block(text, ROB_RE)
    freeze = parse_block(text, FREEZE_RE)

    if not real or not rand:
        return {"status": "error", "error": "parse_failed"}

    real_final, real_sharpe, real_trade_days, real_trades = real
    rand_final, rand_sharpe, rand_trade_days, rand_trades = rand

    inv_final = inv[0] if inv else ""
    inv_sharpe = inv[1] if inv else ""

    out = {
        "status": "ok",
        "real_final": float(real_final),
        "real_sharpe": float(real_sharpe),
        "real_trade_days": int(real_trade_days),
        "real_trades": int(real_trades),
        "rand_final": float(rand_final),
        "rand_sharpe": float(rand_sharpe),
        "rand_trade_days": int(rand_trade_days),
        "rand_trades": int(rand_trades),
        "final_edge": float(real_final) - float(rand_final),
        "sharpe_edge": float(real_sharpe) - float(rand_sharpe),
        "ic_mean": float(ic[0]) if ic else "",
        "wf_pass": (wf[0] == "True") if wf else "",
        "robustness_pass": (rob[0] == "True") if rob else "",
        "inv_final": float(inv_final) if inv_final != "" else "",
        "inv_sharpe": float(inv_sharpe) if inv_sharpe != "" else "",
        "freeze_hash": freeze[0] if freeze else "",
        "freeze_ok": (freeze[1] == "True") if freeze else "",
    }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    py_exe = sys.executable
    env = build_env(os.environ)
    windows = load_windows()

    rows = []
    for idx, (start, end, bucket) in enumerate(windows, start=1):
        print(f"[{idx}/{len(windows)}] {start} -> {end} ({bucket})", flush=True)
        row = {"start": start, "end": end, "bucket": bucket}
        row.update(run_one(py_exe, env, start, end))
        rows.append(row)

    fields = [
        "start",
        "end",
        "bucket",
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
        "inv_final",
        "inv_sharpe",
        "freeze_hash",
        "freeze_ok",
        "error",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    beat_final = sum(1 for r in ok_rows if r["final_edge"] > 0)
    beat_sharpe = sum(1 for r in ok_rows if r["sharpe_edge"] > 0)
    wf_pass = sum(1 for r in ok_rows if r.get("wf_pass") is True)
    rob_pass = sum(1 for r in ok_rows if r.get("robustness_pass") is True)

    print("\nStress-hardened summary")
    print(f"ok_windows={len(ok_rows)}/{len(rows)}")
    print(f"beat_random_final={beat_final}/{len(ok_rows) if ok_rows else 0}")
    print(f"beat_random_sharpe={beat_sharpe}/{len(ok_rows) if ok_rows else 0}")
    print(f"wf_pass={wf_pass}/{len(ok_rows) if ok_rows else 0}")
    print(f"robustness_pass={rob_pass}/{len(ok_rows) if ok_rows else 0}")
    print(f"csv={OUT_CSV}")


if __name__ == "__main__":
    main()
