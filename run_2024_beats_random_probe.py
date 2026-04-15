import json
import os
import subprocess
import sys
from pathlib import Path

from run_10_window_retests import parse_metrics


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

WINDOWS = [
    ("2024_01", "2024-01-01", "2024-01-31"),
    ("2024_02", "2024-02-01", "2024-02-29"),
    ("2024_09", "2024-09-01", "2024-09-30"),
    ("2024_11", "2024-11-01", "2024-11-30"),
]


def run_window(tag: str, start: str, end: str) -> dict:
    env = os.environ.copy()
    env["QUANT_SIGNAL_MODE"] = "model"
    env["QUANT_CARRY_FORWARD_OPEN"] = "1"
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

    print(f"\n=== Probe {tag}: {start} to {end} ===")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")

    log_path = LOGS / f"walkforward_2024_probe_{tag}.log"
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"Probe run failed for {tag}. See {log_path}")

    metrics = parse_metrics(combined)
    metrics["tag"] = tag
    metrics["start"] = start
    metrics["end"] = end
    metrics["log_path"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return metrics


def render_md(rows: list[dict]) -> str:
    lines = []
    lines.append("| Month | Model Final | Random Final | Sharpe | vs Random % | Result | First Failed Gate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for r in rows:
        lines.append(
            "| {m} | {mf:.2f} | {rf:.2f} | {sh:.3f} | {vr:.2f} | {res} | {gate} |".format(
                m=r["tag"],
                mf=float(r.get("model_final") or 0.0),
                rf=float(r.get("random_final") or 0.0),
                sh=float(r.get("model_sharpe") or 0.0),
                vr=float(r.get("model_vs_random_pct") or 0.0),
                res=r.get("pass_or_fail", "-"),
                gate=r.get("first_failed_gate") or "-",
            )
        )
    return "\n".join(lines)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = [run_window(tag, start, end) for tag, start, end in WINDOWS]

    out_json = LOGS / "walkforward_2024_beats_random_probe.json"
    out_json.write_text(json.dumps({"windows": rows}, indent=2), encoding="utf-8")

    out_md = LOGS / "walkforward_2024_beats_random_probe_latest.md"
    out_md.write_text(render_md(rows) + "\n", encoding="utf-8")

    print("\n=== Probe report saved ===")
    print(out_json)
    print(out_md)
    print("\n" + render_md(rows))


if __name__ == "__main__":
    main()
