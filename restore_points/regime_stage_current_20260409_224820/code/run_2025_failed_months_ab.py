import json
import os
import subprocess
import sys
import argparse
from pathlib import Path

from run_10_window_retests import parse_metrics


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
DEFAULT_SCENARIO_FILE = ROOT / "experiments" / "configs" / "failed_months_scenarios.json"

FAILED_WINDOWS = [
    ("2025_01", "2025-01-01", "2025-01-31"),
    ("2025_02", "2025-02-01", "2025-02-28"),
    ("2025_07", "2025-07-01", "2025-07-31"),
    ("2025_10", "2025-10-01", "2025-10-31"),
    ("2025_11", "2025-11-01", "2025-11-30"),
]

SCENARIOS = [
    {
        "name": "baseline",
        "env": {},
    },
    {
        "name": "trade_control",
        "env": {
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "1",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "2",
            "QUANT_KILL_SWITCH_MAX_NEW_POSITIONS": "0",
        },
    },
    {
        "name": "defensive_bad_regime",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.50",
            "QUANT_RANK_THRESHOLD_BAD": "0.70",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.53",
            "QUANT_SIGNAL_SPREAD_BAD": "0.013",
        },
    },
    {
        "name": "stability_combo",
        "env": {
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "1",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "2",
            "QUANT_KILL_SWITCH_MAX_NEW_POSITIONS": "0",
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.55",
            "QUANT_RANK_THRESHOLD_BAD": "0.69",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.525",
            "QUANT_RANDOM_BASELINE_RUNS": "15",
        },
    },
]


def load_scenarios(scenario_file: Path) -> tuple[list[tuple[str, str, str]], list[dict]]:
    if not scenario_file.exists():
        print(f"[WARN] Scenario file not found: {scenario_file}. Using built-in defaults.")
        return FAILED_WINDOWS, SCENARIOS

    payload = json.loads(scenario_file.read_text(encoding="utf-8"))
    windows_raw = payload.get("windows", [])
    scenarios_raw = payload.get("scenarios", [])

    if not windows_raw or not scenarios_raw:
        raise ValueError(f"Invalid scenario config at {scenario_file}: missing windows/scenarios")

    windows = []
    for w in windows_raw:
        if len(w) != 3:
            raise ValueError(f"Invalid window entry: {w}")
        windows.append((w[0], w[1], w[2]))

    scenarios = []
    for s in scenarios_raw:
        if "name" not in s:
            raise ValueError(f"Invalid scenario entry: {s}")
        scenarios.append({
            "name": str(s["name"]),
            "env": dict(s.get("env", {})),
        })

    return windows, scenarios


def run_window(tag: str, start: str, end: str, scenario: dict) -> dict:
    env = os.environ.copy()
    env["QUANT_SIGNAL_MODE"] = "model"
    env["QUANT_SINGLE_MODEL"] = env.get("QUANT_SINGLE_MODEL", "1")
    env["QUANT_ENSEMBLE_SEEDS"] = env.get("QUANT_ENSEMBLE_SEEDS", env.get("QUANT_SEED", "42"))
    env["QUANT_CARRY_FORWARD_OPEN"] = "1"
    env["QUANT_TEST_START"] = start
    env["QUANT_TEST_END"] = end
    env["QUANT_SPLIT_DATE"] = start
    env.update(scenario["env"])

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

    print(f"\n=== {scenario['name']} / {tag} ({start} -> {end}) ===")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")

    log_path = LOGS / f"ab_2025_failed_{scenario['name']}_{tag}.log"
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for {scenario['name']} / {tag}. See {log_path}")

    metrics = parse_metrics(combined)
    metrics["tag"] = tag
    metrics["start"] = start
    metrics["end"] = end
    metrics["scenario"] = scenario["name"]
    metrics["log_path"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return metrics


def score(rows: list[dict]) -> dict:
    pass_count = sum(1 for r in rows if r.get("pass_or_fail") == "PASS")
    avg_vs_random = sum(float(r.get("model_vs_random_pct") or 0.0) for r in rows) / max(1, len(rows))
    avg_vs_nifty = sum(float(r.get("model_vs_nifty_pct") or 0.0) for r in rows) / max(1, len(rows))
    avg_sharpe = sum(float(r.get("model_sharpe") or 0.0) for r in rows) / max(1, len(rows))
    return {
        "pass_count": pass_count,
        "avg_vs_random": avg_vs_random,
        "avg_vs_nifty": avg_vs_nifty,
        "avg_sharpe": avg_sharpe,
    }


def render_markdown(payload: dict) -> str:
    lines = []
    lines.append("| Scenario | Month | Model Final | Random Final | Nifty Final | Sharpe | Trades | vs Random % | vs Nifty % | Result | First Failed Gate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")

    for scenario_name, rows in payload["rows_by_scenario"].items():
        for row in rows:
            lines.append(
                "| {scenario} | {month} | {model_final:.2f} | {random_final:.2f} | {nifty_final:.2f} | {sharpe:.3f} | {trades} | {vs_random:.2f} | {vs_nifty:.2f} | {result} | {failed_gate} |".format(
                    scenario=scenario_name,
                    month=row["tag"],
                    model_final=float(row.get("model_final") or 0.0),
                    random_final=float(row.get("random_final") or 0.0),
                    nifty_final=float(row.get("nifty_final") or 0.0),
                    sharpe=float(row.get("model_sharpe") or 0.0),
                    trades=int(row.get("model_trade_count") or 0),
                    vs_random=float(row.get("model_vs_random_pct") or 0.0),
                    vs_nifty=float(row.get("model_vs_nifty_pct") or 0.0),
                    result=row.get("pass_or_fail", "-"),
                    failed_gate=row.get("first_failed_gate") or "-",
                )
            )

    lines.append("")
    lines.append("| Scenario | Passes (out of 5) | Avg vs Random % | Avg vs Nifty % | Avg Sharpe |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for scenario_name, s in payload["scenario_scores"].items():
        lines.append(
            "| {scenario} | {pass_count} | {avg_vs_random:.2f} | {avg_vs_nifty:.2f} | {avg_sharpe:.3f} |".format(
                scenario=scenario_name,
                pass_count=s["pass_count"],
                avg_vs_random=s["avg_vs_random"],
                avg_vs_nifty=s["avg_vs_nifty"],
                avg_sharpe=s["avg_sharpe"],
            )
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B scenarios on 2025 failed months.")
    parser.add_argument(
        "--scenario-file",
        type=str,
        default=str(DEFAULT_SCENARIO_FILE),
        help="Path to scenario JSON file.",
    )
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    windows, scenarios = load_scenarios(Path(args.scenario_file))

    rows_by_scenario = {}
    scenario_scores = {}

    for scenario in scenarios:
        rows = []
        for tag, start, end in windows:
            rows.append(run_window(tag, start, end, scenario))
        rows_by_scenario[scenario["name"]] = rows
        scenario_scores[scenario["name"]] = score(rows)

    payload = {
        "windows": windows,
        "scenarios": scenarios,
        "rows_by_scenario": rows_by_scenario,
        "scenario_scores": scenario_scores,
    }

    out_json = LOGS / "ab_2025_failed_months_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = LOGS / "ab_2025_failed_months_comparison.md"
    out_md.write_text(render_markdown(payload) + "\n", encoding="utf-8")

    print("\n=== Failed-month A/B sweep saved ===")
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
