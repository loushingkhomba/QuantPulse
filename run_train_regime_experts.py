import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

EXPERTS = [
    ("bad", "bad"),
    ("neutral", "neutral"),
    ("trending", "trending"),
]


def run_expert(expert_name: str, regime_filter: str, args: argparse.Namespace) -> tuple[int, Path]:
    env = os.environ.copy()
    env["QUANT_SIGNAL_MODE"] = "model"
    env["QUANT_OBJECTIVE_MODE"] = args.objective_mode
    env["QUANT_SINGLE_MODEL"] = "1"
    env["QUANT_ENSEMBLE_SEEDS"] = str(args.seed)
    env["QUANT_PRIMARY_MODEL_SEED"] = str(args.seed)
    env["QUANT_MODEL_TAG"] = expert_name
    env["QUANT_TRAIN_REGIME_FILTER"] = regime_filter

    if args.train_start:
        env["QUANT_TRAIN_START"] = args.train_start
    if args.train_end:
        env["QUANT_TRAIN_END"] = args.train_end
    if args.test_start:
        env["QUANT_TEST_START"] = args.test_start
    if args.test_end:
        env["QUANT_TEST_END"] = args.test_end
    if args.split_date:
        env["QUANT_SPLIT_DATE"] = args.split_date

    cmd = [sys.executable, "-u", "train.py"]
    if args.start:
        cmd.extend(["--start", args.start])
    if args.end:
        cmd.extend(["--end", args.end])

    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"train_regime_expert_{expert_name}.log"

    print(f"\n=== Training {expert_name.upper()} expert ===")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode == 0:
        print(f"Saved: {log_path.relative_to(ROOT).as_posix()}")
    else:
        print(f"FAILED: {expert_name} expert. See {log_path.relative_to(ROOT).as_posix()}")

    return proc.returncode, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bad/neutral/trending regime experts.")
    parser.add_argument("--objective-mode", default="ranking", choices=["ranking", "classification"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default="", help="Optional CLI start date forwarded to train.py")
    parser.add_argument("--end", type=str, default="", help="Optional CLI end date forwarded to train.py")
    parser.add_argument("--train-start", type=str, default="")
    parser.add_argument("--train-end", type=str, default="")
    parser.add_argument("--test-start", type=str, default="")
    parser.add_argument("--test-end", type=str, default="")
    parser.add_argument("--split-date", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = []

    for expert_name, regime_filter in EXPERTS:
        code, log_path = run_expert(expert_name, regime_filter, args)
        if code != 0:
            failures.append((expert_name, log_path))

    if failures:
        print("\nSome expert trainings failed:")
        for expert_name, log_path in failures:
            print(f"- {expert_name}: {log_path.relative_to(ROOT).as_posix()}")
        raise SystemExit(1)

    print("\nAll regime experts trained successfully.")


if __name__ == "__main__":
    main()
