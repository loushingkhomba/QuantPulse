import json
import os
import re
import subprocess
import sys
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
REPORT_GLOB = "walkforward_10window_comparison*.json"

# 10 monthly windows around the previously identified weak period.
WINDOWS = [
    ("2025_07", "2025-07-01", "2025-07-31"),
    ("2025_08", "2025-08-01", "2025-08-31"),
    ("2025_09", "2025-09-01", "2025-09-30"),
    ("2025_10", "2025-10-01", "2025-10-31"),
    ("2025_11", "2025-11-01", "2025-11-30"),
    ("2025_12", "2025-12-01", "2025-12-31"),
    ("2026_01", "2026-01-01", "2026-01-31"),
    ("2026_02", "2026-02-01", "2026-02-28"),
    ("2026_03", "2026-03-01", "2026-03-31"),
    ("2026_04", "2026-04-01", "2026-04-09"),
]

THRESHOLDS = {
    "beats_random_min_edge_pct": 2.0,
    "beats_nifty_min_edge_pct": 0.5,
    "min_sharpe": 1.0,
    "max_drawdown_abs_pct": 15.0,
    "max_trade_count": 50,
}


def _extract_section_value(text: str, section: str, key: str) -> float | None:
    pattern = rf"{re.escape(section)}.*?{re.escape(key)}\s*([-0-9.]+)"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_section_int(text: str, section: str, key: str) -> int | None:
    pattern = rf"{re.escape(section)}.*?{re.escape(key)}\s*(\d+)"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_metrics(log_text: str) -> dict:
    result = {
        "model_final": _extract_section_value(log_text, "REAL MODEL", "Final Value:"),
        "model_sharpe": _extract_section_value(log_text, "REAL MODEL", "Sharpe:"),
        "model_dd_pct": _extract_section_value(log_text, "REAL MODEL", "Max Drawdown:"),
        "model_trade_count": _extract_section_int(log_text, "REAL MODEL", "Trades Executed:"),
        "random_final": _extract_section_value(log_text, "RANDOM BASELINE", "Final Value:"),
        "random_sharpe": _extract_section_value(log_text, "RANDOM BASELINE", "Sharpe:"),
        "inverted_final": _extract_section_value(log_text, "INVERTED SIGNAL (DIAGNOSTIC)", "Final Value:"),
        "inverted_sharpe": _extract_section_value(log_text, "INVERTED SIGNAL (DIAGNOSTIC)", "Sharpe:"),
        "nifty_final": _extract_section_value(log_text, "NIFTY BUY & HOLD", "Final Value:"),
    }

    if result["model_final"] is not None and result["nifty_final"] is not None and result["nifty_final"] != 0:
        result["model_vs_nifty_pct"] = (result["model_final"] / result["nifty_final"] - 1.0) * 100.0
    else:
        result["model_vs_nifty_pct"] = None

    if result["model_final"] is not None and result["random_final"] is not None and result["random_final"] != 0:
        result["model_vs_random_pct"] = (result["model_final"] / result["random_final"] - 1.0) * 100.0
    else:
        result["model_vs_random_pct"] = None

    gate_order = ["beats_random", "beats_nifty", "positive_sharpe", "max_drawdown_ok", "trade_count_ok"]
    pass_flags = {
        "beats_random": (result["model_vs_random_pct"] is not None and result["model_vs_random_pct"] > THRESHOLDS["beats_random_min_edge_pct"]),
        "beats_nifty": (result["model_vs_nifty_pct"] is not None and result["model_vs_nifty_pct"] > THRESHOLDS["beats_nifty_min_edge_pct"]),
        "positive_sharpe": (result["model_sharpe"] is not None and result["model_sharpe"] > THRESHOLDS["min_sharpe"]),
        "max_drawdown_ok": (result["model_dd_pct"] is not None and abs(result["model_dd_pct"]) < THRESHOLDS["max_drawdown_abs_pct"]),
        "trade_count_ok": (result["model_trade_count"] is not None and result["model_trade_count"] <= THRESHOLDS["max_trade_count"]),
    }
    first_failed_gate = next((gate for gate in gate_order if not pass_flags[gate]), None)
    pass_flags["window_pass"] = all(pass_flags.values())
    result["pass_flags"] = pass_flags
    result["pass_or_fail"] = "PASS" if pass_flags["window_pass"] else "FAIL"
    result["first_failed_gate"] = None if pass_flags["window_pass"] else first_failed_gate

    return result


def run_window(tag: str, start: str, end: str) -> dict:
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

    metrics = parse_metrics(combined)
    metrics["tag"] = tag
    metrics["start"] = start
    metrics["end"] = end
    metrics["log_path"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return metrics


def parse_existing_window(tag: str, start: str, end: str) -> dict:
    log_path = LOGS / f"walkforward_dryrun_{tag}.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log for parse-only mode: {log_path}")

    text = log_path.read_text(encoding="utf-8")
    metrics = parse_metrics(text)
    metrics["tag"] = tag
    metrics["start"] = start
    metrics["end"] = end
    metrics["log_path"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return metrics


def find_latest_report_path() -> Path:
    candidates = [path for path in LOGS.glob(REPORT_GLOB) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No report files found matching {REPORT_GLOB} in {LOGS}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _fmt(value, precision: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def render_report_table(report: dict) -> str:
    windows = report.get("windows", [])
    lines = []
    lines.append("| Window | Model Final | Random Final | Nifty Final | Sharpe | Max DD % | Trades | vs Random % | vs Nifty % | Result | First Failed Gate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for window in windows:
        random_final = window.get("random_final")
        nifty_final = window.get("nifty_final")
        lines.append(
            "| {tag} | {model_final} | {random_final} | {nifty_final} | {model_sharpe} | {model_dd} | {trades} | {edge_random} | {edge_nifty} | {result} | {failed_gate} |".format(
                tag=window.get("tag", "-"),
                model_final=_fmt(window.get("model_final")),
                random_final=_fmt(random_final),
                nifty_final=_fmt(nifty_final),
                model_sharpe=_fmt(window.get("model_sharpe"), precision=3),
                model_dd=_fmt(window.get("model_dd_pct")),
                trades=_fmt(window.get("model_trade_count"), precision=0),
                edge_random=_fmt(window.get("model_vs_random_pct")),
                edge_nifty=_fmt(window.get("model_vs_nifty_pct")),
                result=window.get("pass_or_fail", "-"),
                failed_gate=window.get("first_failed_gate") or "-",
            )
        )

    windows_total = len(windows)
    windows_passed = sum(1 for window in windows if window.get("pass_or_fail") == "PASS")
    summary_lines = [
        f"Latest report: {report.get('report_path', '-')}" if report.get('report_path') else "Latest report:",
        f"Pass rate: {windows_passed}/{windows_total}",
        "",
        "\n".join(lines),
    ]
    return "\n".join(summary_lines)


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["report_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or parse 10-window frozen backtests.")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip execution and build report from existing per-window logs.",
    )
    parser.add_argument(
        "--latest-report",
        action="store_true",
        help="Print the newest comparison report JSON as a table without rerunning windows.",
    )
    args = parser.parse_args()

    if args.latest_report:
        latest_report_path = find_latest_report_path()
        report = load_report(latest_report_path)
        print(render_report_table(report))
        return

    LOGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, start, end in WINDOWS:
        if args.parse_only:
            rows.append(parse_existing_window(tag, start, end))
        else:
            rows.append(run_window(tag, start, end))

    report = {
        "windows": rows,
        "thresholds": THRESHOLDS,
        "worst_by_model_sharpe": min(rows, key=lambda r: float("inf") if r["model_sharpe"] is None else r["model_sharpe"]),
        "worst_by_model_final": min(rows, key=lambda r: float("inf") if r["model_final"] is None else r["model_final"]),
    }

    out_path = LOGS / "walkforward_10window_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["report_path"] = str(out_path.relative_to(ROOT)).replace("\\", "/")
    table_text = render_report_table(report)
    table_path = LOGS / "walkforward_10window_comparison_latest.md"
    table_path.write_text(table_text + "\n", encoding="utf-8")

    print("\n=== 10-window report saved ===")
    print(out_path)
    print("\n=== Latest report table ===")
    print(table_text)
    print(f"\nMarkdown table saved: {table_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
