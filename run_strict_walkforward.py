import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


@dataclass
class Window:
    tag: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return month_start(ts) + pd.offsets.MonthEnd(1)


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return month_start(ts) + pd.DateOffset(months=months)


def build_windows(
    start_month: str,
    end_month: str,
    train_months: int,
    test_months: int,
    step_months: int,
) -> List[Window]:
    start_ts = month_start(pd.to_datetime(start_month))
    end_ts = month_end(pd.to_datetime(end_month))

    windows: List[Window] = []
    cursor = start_ts
    idx = 0

    while True:
        train_start = add_months(cursor, -train_months)
        train_end = month_end(add_months(cursor, -1))
        test_start = cursor
        test_end = month_end(add_months(cursor, test_months - 1))

        if test_end > end_ts:
            break

        tag = f"wf_{idx:02d}_{test_start.strftime('%Y%m')}_{test_end.strftime('%Y%m')}"
        windows.append(
            Window(
                tag=tag,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            )
        )

        idx += 1
        cursor = add_months(cursor, step_months)

    return windows


def parse_scalar(text: str, label: str, cast=float):
    m = re.search(rf"{re.escape(label)}\s*([-0-9.]+)", text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except Exception:
        return None


def parse_section_metrics(text: str, section: str) -> Dict[str, Optional[float]]:
    pattern = re.compile(
        rf"{re.escape(section)}\n"
        r"Final Value: ([\d.]+)\n"
        r"Sharpe: ([\d.-]+)\n"
        r"Max Drawdown: ([\d.-]+) %\n"
        r"Annualized Return: ([\d.-]+) %\n"
        r"Trade Days: (\d+)\n"
        r"Trades Executed: (\d+)",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return {
            "final_value": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "annualized_return_pct": None,
            "trade_days": None,
            "trades_executed": None,
        }

    return {
        "final_value": float(m.group(1)),
        "sharpe": float(m.group(2)),
        "max_drawdown_pct": float(m.group(3)),
        "annualized_return_pct": float(m.group(4)),
        "trade_days": int(m.group(5)),
        "trades_executed": int(m.group(6)),
    }


def parse_nifty_final(text: str) -> Optional[float]:
    m = re.search(r"NIFTY BUY & HOLD\nFinal Value: ([\d.]+)", text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "count": 0}

    rng = np.random.RandomState(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()

    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "mean": float(arr.mean()),
        "ci_low": low,
        "ci_high": high,
        "count": int(len(arr)),
    }


def run_window(window: Window, frozen_env: Dict[str, str], backtest_only: bool) -> Dict:
    cmd = [sys.executable, "-u", "train.py"]
    if backtest_only:
        cmd.append("--backtest-only")

    env = os.environ.copy()
    env.update(frozen_env)
    env["QUANT_TRAIN_START"] = window.train_start
    env["QUANT_TRAIN_END"] = window.train_end
    env["QUANT_TEST_START"] = window.test_start
    env["QUANT_TEST_END"] = window.test_end
    env["QUANT_SPLIT_DATE"] = window.test_start

    log_path = LOGS / f"strict_walkforward_{window.tag}.log"
    metrics_path = LOGS / f"strict_walkforward_{window.tag}_metrics.json"
    env["QUANT_METRICS_OUT"] = str(metrics_path)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(text, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"Window failed: {window.tag}. See {log_path}")

    model_metrics = parse_section_metrics(text, "REAL MODEL")
    random_metrics = parse_section_metrics(text, "RANDOM BASELINE")
    inv_metrics = parse_section_metrics(text, "INVERTED SIGNAL (DIAGNOSTIC)")

    standard_metrics = {}
    if metrics_path.exists():
        standard_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    signal_quality = standard_metrics.get("signal_quality", {})
    information_coefficient = signal_quality.get("information_coefficient", {})
    ic_decay = information_coefficient.get("ic_decay", {})
    return_quality = standard_metrics.get("return_quality", {})

    if model_metrics["final_value"] and random_metrics["final_value"]:
        edge_vs_random_pct = ((model_metrics["final_value"] / random_metrics["final_value"]) - 1.0) * 100.0
    else:
        edge_vs_random_pct = None

    nifty_final = parse_nifty_final(text)
    if model_metrics["final_value"] and nifty_final:
        edge_vs_nifty_pct = ((model_metrics["final_value"] / nifty_final) - 1.0) * 100.0
    else:
        edge_vs_nifty_pct = None

    return {
        "window": {
            "tag": window.tag,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "test_start": window.test_start,
            "test_end": window.test_end,
            "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/"),
            "metrics_path": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "model": model_metrics,
        "random": random_metrics,
        "inverted_diagnostic": inv_metrics,
        "nifty_final": nifty_final,
        "signal_quality": {
            "ic_mean": information_coefficient.get("mean_daily_ic"),
            "ic_decay": ic_decay,
            "spread_mean": signal_quality.get("confidence_spread", {}).get("mean_rank1_rank10_spread"),
            "calibration_ece": signal_quality.get("calibration", {}).get("expected_calibration_error"),
        },
        "trade_quality": {
            "avg_alpha_per_trade": return_quality.get("alpha_vs_nifty_per_trade", {}).get("average_alpha"),
            "profit_factor": return_quality.get("profit_factor"),
            "win_loss_ratio": return_quality.get("win_loss_ratio"),
        },
        "edges_pct": {
            "vs_random": edge_vs_random_pct,
            "vs_nifty": edge_vs_nifty_pct,
        },
    }


def save_outputs(payload: Dict, out_tag: str) -> Tuple[Path, Path]:
    LOGS.mkdir(parents=True, exist_ok=True)
    json_path = LOGS / f"strict_walkforward_{out_tag}.json"
    md_path = LOGS / f"strict_walkforward_{out_tag}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = []
    lines.append("| Window | Model Final | Random Final | Nifty Final | Model Sharpe | IC Mean | Avg Alpha | Profit Factor | Edge vs Random % | Edge vs Nifty % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["windows"]:
        w = row["window"]
        model = row["model"]
        random_ = row["random"]
        lines.append(
            "| {tag} | {mf:.2f} | {rf:.2f} | {nf:.2f} | {ms:.3f} | {ic:.5f} | {alpha:.5f} | {pf:.3f} | {er:.2f} | {en:.2f} |".format(
                tag=w["tag"],
                mf=float(model.get("final_value") or 0.0),
                rf=float(random_.get("final_value") or 0.0),
                nf=float(row.get("nifty_final") or 0.0),
                ms=float(model.get("sharpe") or 0.0),
                ic=float(row["signal_quality"].get("ic_mean") or 0.0),
                alpha=float(row.get("trade_quality", {}).get("avg_alpha_per_trade") or 0.0),
                pf=float(row.get("trade_quality", {}).get("profit_factor") or 0.0),
                er=float(row["edges_pct"].get("vs_random") or 0.0),
                en=float(row["edges_pct"].get("vs_nifty") or 0.0),
            )
        )

    summary = payload["aggregate"]
    lines.append("")
    lines.append("Aggregate 95% bootstrap CIs")
    lines.append(f"- Model sharpe mean: {summary['model_sharpe']['mean']:.4f} [{summary['model_sharpe']['ci_low']:.4f}, {summary['model_sharpe']['ci_high']:.4f}]")
    lines.append(f"- Model annualized return % mean: {summary['model_annualized_return_pct']['mean']:.4f} [{summary['model_annualized_return_pct']['ci_low']:.4f}, {summary['model_annualized_return_pct']['ci_high']:.4f}]")
    lines.append(f"- IC mean: {summary['ic_mean']['mean']:.5f} [{summary['ic_mean']['ci_low']:.5f}, {summary['ic_mean']['ci_high']:.5f}]")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict walk-forward runner with frozen hyperparameters and CI summary.")
    parser.add_argument("--start-month", type=str, default="2024-01")
    parser.add_argument("--end-month", type=str, default="2026-03")
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all windows.")
    parser.add_argument("--backtest-only", action="store_true", help="Use existing checkpoints per window (faster smoke mode).")
    parser.add_argument("--out-tag", type=str, default="phaseB_v1")
    args = parser.parse_args()

    windows = build_windows(
        start_month=args.start_month,
        end_month=args.end_month,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    if not windows:
        raise ValueError("No windows generated. Check date bounds and window sizes.")

    # Frozen across all windows: this is the strict no-retune contract.
    frozen_env = {
        "QUANT_SIGNAL_MODE": "model",
        "QUANT_OBJECTIVE_MODE": "ranking",
        "QUANT_RANKING_AUX_CE_WEIGHT": "0.25",
        "QUANT_EXECUTION_SIGNAL_DIRECTION": "normal",
        "QUANT_AUTO_ORIENT_SIGNAL": "1",
        "QUANT_ALLOW_ORIENT_EXECUTION": "0",
        "QUANT_AUTO_INVERT_SIGNAL": "0",
        "QUANT_RANDOM_BASELINE_RUNS": "7",
        "QUANT_CARRY_FORWARD_OPEN": "1",
    }

    print("Running strict walk-forward windows:", len(windows))
    print("Frozen hyperparameters:", frozen_env)

    rows = []
    for i, window in enumerate(windows, start=1):
        print(f"[{i}/{len(windows)}] {window.tag}: train {window.train_start}..{window.train_end}, test {window.test_start}..{window.test_end}")
        rows.append(run_window(window, frozen_env=frozen_env, backtest_only=args.backtest_only))

    agg = {
        "model_sharpe": bootstrap_ci([float(r["model"].get("sharpe") or np.nan) for r in rows]),
        "model_annualized_return_pct": bootstrap_ci([float(r["model"].get("annualized_return_pct") or np.nan) for r in rows]),
        "ic_mean": bootstrap_ci([float(r["signal_quality"].get("ic_mean") or np.nan) for r in rows]),
    }

    payload = {
        "meta": {
            "strict_walkforward": True,
            "frozen_hyperparameters": frozen_env,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "step_months": args.step_months,
            "backtest_only": bool(args.backtest_only),
            "max_windows": args.max_windows,
            "python": sys.executable,
        },
        "windows": rows,
        "aggregate": agg,
    }

    json_path, md_path = save_outputs(payload, args.out_tag)
    print("Saved JSON:", json_path)
    print("Saved markdown:", md_path)


if __name__ == "__main__":
    main()
