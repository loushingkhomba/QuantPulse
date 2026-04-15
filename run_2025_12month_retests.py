import json
import os
import subprocess
import sys
from pathlib import Path

from run_10_window_retests import apply_month_type_profile, parse_metrics, render_report_table, THRESHOLDS


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


WINDOWS = [
    ("2025_01", "2025-01-01", "2025-01-31"),
    ("2025_02", "2025-02-01", "2025-02-28"),
    ("2025_03", "2025-03-01", "2025-03-31"),
    ("2025_04", "2025-04-01", "2025-04-30"),
    ("2025_05", "2025-05-01", "2025-05-31"),
    ("2025_06", "2025-06-01", "2025-06-30"),
    ("2025_07", "2025-07-01", "2025-07-31"),
    ("2025_08", "2025-08-01", "2025-08-31"),
    ("2025_09", "2025-09-01", "2025-09-30"),
    ("2025_10", "2025-10-01", "2025-10-31"),
    ("2025_11", "2025-11-01", "2025-11-30"),
    ("2025_12", "2025-12-01", "2025-12-31"),
]


def run_window(tag: str, start: str, end: str) -> dict:
    log_path = LOGS / f"walkforward_2025_dryrun_{tag}.log"
    env = os.environ.copy()
    env["QUANT_SIGNAL_MODE"] = "model"
    env["QUANT_CARRY_FORWARD_OPEN"] = "1"
    env["QUANT_TRANSACTION_COST"] = env.get("QUANT_TRANSACTION_COST", "0.0005")
    env["QUANT_TEST_START"] = start
    env["QUANT_TEST_END"] = end
    env["QUANT_SPLIT_DATE"] = start

    profile_args, profile_metadata = apply_month_type_profile(start)

    venv_python = ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    cmd = [
        str(venv_python),
        "-u",
        "train.py",
        "--backtest-only",
        "--start",
        start,
        "--end",
        end,
    ]
    cmd.extend(profile_args)

    print(f"\n=== Running {tag}: {start} to {end} ===")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for {tag}. See {log_path}")

    metrics = parse_metrics(combined)
    metrics["tag"] = tag
    metrics["start"] = start
    metrics["end"] = end
    metrics["profile"] = profile_metadata
    metrics["log_path"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return metrics


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = [run_window(tag, start, end) for tag, start, end in WINDOWS]

    report = {
        "windows": rows,
        "thresholds": THRESHOLDS,
        "worst_by_model_sharpe": min(rows, key=lambda r: float("inf") if r["model_sharpe"] is None else r["model_sharpe"]),
        "worst_by_model_final": min(rows, key=lambda r: float("inf") if r["model_final"] is None else r["model_final"]),
    }

    out_path = LOGS / "walkforward_2025_12month_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== 2025 12-month report saved ===")
    print(out_path)
    print("\n=== Latest report table ===")
    print(render_report_table(report))


if __name__ == "__main__":
    main()
