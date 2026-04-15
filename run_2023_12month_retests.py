import json
import os
import subprocess
import sys
from pathlib import Path

from run_10_window_retests import parse_metrics, render_report_table, THRESHOLDS


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


WINDOWS = [
    ("2023_01", "2023-01-01", "2023-01-31"),
    ("2023_02", "2023-02-01", "2023-02-28"),
    ("2023_03", "2023-03-01", "2023-03-31"),
    ("2023_04", "2023-04-01", "2023-04-30"),
    ("2023_05", "2023-05-01", "2023-05-31"),
    ("2023_06", "2023-06-01", "2023-06-30"),
    ("2023_07", "2023-07-01", "2023-07-31"),
    ("2023_08", "2023-08-01", "2023-08-31"),
    ("2023_09", "2023-09-01", "2023-09-30"),
    ("2023_10", "2023-10-01", "2023-10-31"),
    ("2023_11", "2023-11-01", "2023-11-30"),
    ("2023_12", "2023-12-01", "2023-12-31"),
]


def run_window(tag: str, start: str, end: str) -> dict:
    log_path = LOGS / f"walkforward_2023_dryrun_{tag}.log"
    env = os.environ.copy()
    env["QUANT_SIGNAL_MODE"] = "model"
    env["QUANT_CARRY_FORWARD_OPEN"] = "1"
    env["QUANT_TRANSACTION_COST"] = env.get("QUANT_TRANSACTION_COST", "0.0005")
    env["QUANT_TEST_START"] = start
    env["QUANT_TEST_END"] = end
    env["QUANT_SPLIT_DATE"] = start

    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--backtest-only",
        "--start",
        start,
        "--end",
        end,
    ]

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

    out_path = LOGS / "walkforward_2023_12month_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["report_path"] = str(out_path.relative_to(ROOT)).replace("\\", "/")
    table_text = render_report_table(report)
    table_path = LOGS / "walkforward_2023_12month_comparison_latest.md"
    table_path.write_text(table_text + "\n", encoding="utf-8")

    print("\n=== 2023 12-month report saved ===")
    print(out_path)
    print("\n=== Latest report table ===")
    print(table_text)
    print(f"\nMarkdown table saved: {table_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
