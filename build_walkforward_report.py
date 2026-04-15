import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.zerodha_ohlc_loader import download_data


ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
OUT_PATH = LOGS_DIR / "walkforward_comparison_report.json"
REPORT_GLOB = "walkforward_comparison_report*.json"


WINDOWS = [
    {
        "name": "oct_2025_momentum",
        "start": "2025-10-01",
        "end": "2025-10-31",
        "log": LOGS_DIR / "walkforward_dryrun_oct2025.log",
    },
    {
        "name": "jan_2026_volatility",
        "start": "2026-01-01",
        "end": "2026-01-31",
        "log": LOGS_DIR / "walkforward_dryrun_jan2026.log",
    },
    {
        "name": "mar_2026_grinding",
        "start": "2026-03-01",
        "end": "2026-03-31",
        "log": LOGS_DIR / "walkforward_dryrun_march2026.log",
    },
]


THRESHOLDS = {
    "beats_random_min_edge_pct": 2.0,
    "beats_nifty_min_edge_pct": 0.5,
    "min_sharpe": 1.0,
    "max_drawdown_abs_pct": 15.0,
    "max_drawdown_vs_nifty_multiple": 1.5,
    "max_trade_count": 50,
}


def parse_section(text: str, section: str):
    pattern = re.compile(
        rf"{section}\n"
        r"Final Value: ([\d.]+)\n"
        r"Sharpe: ([\d.-]+)\n"
        r"Max Drawdown: ([\d.-]+) %\n"
        r"Annualized Return: ([\d.-]+) %\n"
        r"Trade Days: (\d+)\n"
        r"Trades Executed: (\d+)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not parse section: {section}")

    return {
        "final_value": float(match.group(1)),
        "sharpe": float(match.group(2)),
        "max_drawdown_pct": float(match.group(3)),
        "annualized_return_pct": float(match.group(4)),
        "trade_days": int(match.group(5)),
        "trade_count": int(match.group(6)),
    }


def parse_scalar(text: str, label: str, cast_fn=float):
    pattern = re.compile(rf"{re.escape(label)}: ([^\n]+)")
    match = pattern.search(text)
    if not match:
        return None
    try:
        return cast_fn(match.group(1).strip())
    except Exception:
        return None


def find_latest_report_path() -> Path:
    candidates = [path for path in LOGS_DIR.glob(REPORT_GLOB) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No report files found matching {REPORT_GLOB} in {LOGS_DIR}")
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


def render_report_table(payload: dict) -> str:
    runs = payload.get("runs", [])
    lines = []
    lines.append("| Window | Model Final | Random Final | Nifty Final | Sharpe | Max DD % | Trades | vs Random % | vs Nifty % | Result | First Failed Gate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for run in runs:
        window = run.get("window", {})
        edges = run.get("edges_pct", {})
        model = run.get("model", {})
        random_ = run.get("random", {})
        nifty = run.get("nifty", {})
        flags = run.get("flags", {})
        failed_gate = "-"
        if not flags.get("window_pass", False):
            for gate in ["beats_random", "beats_nifty", "positive_sharpe", "max_drawdown_ok", "trade_count_under_cap"]:
                if not flags.get(gate, False):
                    failed_gate = gate
                    break

        lines.append(
            "| {name} | {model_final} | {random_final} | {nifty_final} | {sharpe} | {dd} | {trades} | {edge_random} | {edge_nifty} | {result} | {failed_gate} |".format(
                name=window.get("name", "-"),
                model_final=_fmt(model.get("final_value")),
                random_final=_fmt(random_.get("final_value")),
                nifty_final=_fmt(nifty.get("final_value")),
                sharpe=_fmt(model.get("sharpe"), precision=3),
                dd=_fmt(model.get("max_drawdown_pct")),
                trades=_fmt(model.get("trade_count"), precision=0),
                edge_random=_fmt(edges.get("vs_random")),
                edge_nifty=_fmt(edges.get("vs_nifty")),
                result="PASS" if flags.get("window_pass") else "FAIL",
                failed_gate=failed_gate,
            )
        )

    summary = payload.get("summary", {})
    summary_lines = [
        f"Latest report: {payload.get('report_path', '-')}" if payload.get("report_path") else "Latest report:",
        f"Pass rate: {summary.get('windows_passed', 0)}/{summary.get('windows_total', 0)}",
        "",
        "\n".join(lines),
    ]
    return "\n".join(summary_lines)


def load_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return payload


def compute_nifty_drawdowns(windows):
    data = download_data()
    nifty = (
        data[["Date", "nifty_close"]]
        .dropna(subset=["nifty_close"])
        .drop_duplicates(subset=["Date"])
        .copy()
    )
    nifty["Date"] = pd.to_datetime(nifty["Date"])
    nifty = nifty.sort_values("Date").reset_index(drop=True)

    out = {}
    for w in windows:
        start = pd.to_datetime(w["start"])
        end = pd.to_datetime(w["end"])
        seg = nifty[(nifty["Date"] >= start) & (nifty["Date"] <= end)].copy()

        if seg.empty:
            out[w["name"]] = {
                "nifty_max_drawdown_pct": None,
                "nifty_start_value": None,
                "nifty_end_value": None,
            }
            continue

        seg["running_peak"] = seg["nifty_close"].cummax()
        seg["drawdown"] = seg["nifty_close"] / seg["running_peak"] - 1.0
        max_dd_abs = float(abs(seg["drawdown"].min()) * 100.0)
        out[w["name"]] = {
            "nifty_max_drawdown_pct": max_dd_abs,
            "nifty_start_value": float(seg["nifty_close"].iloc[0]),
            "nifty_end_value": float(seg["nifty_close"].iloc[-1]),
        }

    return out


def main():
    nifty_dd = compute_nifty_drawdowns(WINDOWS)
    runs = []

    import argparse
    parser = argparse.ArgumentParser(description="Build or print the 3-window walk-forward comparison report.")
    parser.add_argument(
        "--latest-report",
        action="store_true",
        help="Print the newest comparison report JSON as a table without rebuilding windows.",
    )
    args = parser.parse_args()

    if args.latest_report:
        latest_report_path = find_latest_report_path()
        payload = load_report(latest_report_path)
        print(render_report_table(payload))
        return

    for window in WINDOWS:
        text = window["log"].read_text(encoding="utf-8")
        real = parse_section(text, "REAL MODEL")
        random_ = parse_section(text, "RANDOM BASELINE")

        nifty_match = re.search(r"NIFTY BUY & HOLD\nFinal Value: ([\d.]+)", text, re.MULTILINE)
        if not nifty_match:
            raise ValueError(f"Could not parse NIFTY section in {window['log']}")
        nifty_final = float(nifty_match.group(1))

        transaction_cost = parse_scalar(text, "Transaction cost")
        slippage = parse_scalar(text, "Slippage")
        inversion_triggered = "[SIGNAL REVERSAL DETECTED]" in text

        random_edge_pct = ((real["final_value"] / random_["final_value"]) - 1.0) * 100.0
        nifty_edge_pct = ((real["final_value"] / nifty_final) - 1.0) * 100.0

        model_max_dd_abs = abs(real["max_drawdown_pct"])
        nifty_max_dd_abs = nifty_dd[window["name"]]["nifty_max_drawdown_pct"]
        max_dd_vs_nifty_limit = None
        pass_max_dd_vs_nifty = False
        if nifty_max_dd_abs is not None:
            max_dd_vs_nifty_limit = THRESHOLDS["max_drawdown_vs_nifty_multiple"] * nifty_max_dd_abs
            pass_max_dd_vs_nifty = model_max_dd_abs < max_dd_vs_nifty_limit

        flags = {
            "beats_random": random_edge_pct > THRESHOLDS["beats_random_min_edge_pct"],
            "beats_nifty": nifty_edge_pct > THRESHOLDS["beats_nifty_min_edge_pct"],
            "positive_sharpe": real["sharpe"] > THRESHOLDS["min_sharpe"],
            "max_drawdown_under_15pct": model_max_dd_abs < THRESHOLDS["max_drawdown_abs_pct"],
            "max_drawdown_under_1p5x_nifty": pass_max_dd_vs_nifty,
            "trade_count_under_cap": real["trade_count"] <= THRESHOLDS["max_trade_count"],
        }
        flags["max_drawdown_ok"] = flags["max_drawdown_under_15pct"] or flags["max_drawdown_under_1p5x_nifty"]
        flags["window_pass"] = (
            flags["beats_random"]
            and flags["beats_nifty"]
            and flags["positive_sharpe"]
            and flags["max_drawdown_ok"]
            and flags["trade_count_under_cap"]
        )

        runs.append(
            {
                "window": {
                    "name": window["name"],
                    "start": window["start"],
                    "end": window["end"],
                    "log_path": str(window["log"].relative_to(ROOT)).replace("\\", "/"),
                },
                "model": real,
                "random": random_,
                "nifty": {
                    "final_value": nifty_final,
                    "max_drawdown_pct": nifty_max_dd_abs,
                    "start_value": nifty_dd[window["name"]]["nifty_start_value"],
                    "end_value": nifty_dd[window["name"]]["nifty_end_value"],
                },
                "execution": {
                    "transaction_cost": transaction_cost,
                    "slippage": slippage,
                    "inversion_triggered": inversion_triggered,
                    "trade_count": real["trade_count"],
                    "trade_days": real["trade_days"],
                },
                "edges_pct": {
                    "vs_random": random_edge_pct,
                    "vs_nifty": nifty_edge_pct,
                },
                "drawdown_limits": {
                    "absolute_limit_pct": THRESHOLDS["max_drawdown_abs_pct"],
                    "vs_nifty_limit_pct": max_dd_vs_nifty_limit,
                },
                "flags": flags,
            }
        )

    total = len(runs)
    passed = sum(1 for r in runs if r["flags"]["window_pass"])

    payload = {
        "generated_by": "build_walkforward_report.py",
        "thresholds": THRESHOLDS,
        "summary": {
            "windows_total": total,
            "windows_passed": passed,
            "overall_pass": passed == total,
        },
        "runs": runs,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_path"] = str(OUT_PATH.relative_to(ROOT)).replace("\\", "/")
    table_text = render_report_table(payload)
    table_path = LOGS_DIR / "walkforward_comparison_report_latest.md"
    table_path.write_text(table_text + "\n", encoding="utf-8")
    print(f"Saved: {OUT_PATH}")
    print("\n=== Latest report table ===")
    print(table_text)
    print(f"\nMarkdown table saved: {table_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
