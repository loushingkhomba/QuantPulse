import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


WINDOWS = [
    ("oct2025", "2025-10-01", "2025-10-31"),
    ("jan2026", "2026-01-01", "2026-01-31"),
    ("march2026", "2026-03-01", "2026-03-31"),
]


def run_window(tag: str, start: str, end: str) -> None:
    log_path = LOGS / f"walkforward_dryrun_{tag}.log"
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


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    for tag, start, end in WINDOWS:
        run_window(tag, start, end)

    print("\n=== Regenerating comparison report ===")
    rep = subprocess.run(
        [sys.executable, "-u", "build_walkforward_report.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(rep.stdout)
    if rep.returncode != 0:
        print(rep.stderr)
        raise RuntimeError("Failed to regenerate walkforward comparison report")

    print("All phase re-tests complete.")


if __name__ == "__main__":
    main()
