import json
import os
import subprocess
import sys
from pathlib import Path

from run_10_window_retests import parse_metrics


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

WINDOWS = [
    ("2025_02", "2025-02-01", "2025-02-28"),
    ("2025_05", "2025-05-01", "2025-05-31"),
    ("2025_06", "2025-06-01", "2025-06-30"),
    ("2025_11", "2025-11-01", "2025-11-30"),
]

SCENARIOS = [
    {
        "name": "baseline_current",
        "env": {},
    },
    {
        "name": "relax_spread_conf",
        "env": {
            "QUANT_SIGNAL_SPREAD_BAD": "0.012",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.52",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "1",
        },
    },
    {
        "name": "more_entries",
        "env": {
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "3",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "1",
            "QUANT_RANK_THRESHOLD_BAD": "0.66",
            "QUANT_SIGNAL_SPREAD_BAD": "0.012",
        },
    },
    {
        "name": "higher_bad_exposure",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.70",
            "QUANT_SIGNAL_SPREAD_BAD": "0.012",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.52",
        },
    },
]


def run_window_for_scenario(tag: str, start: str, end: str, scenario: dict) -> dict:
    env = os.environ.copy()
    env["QUANT_SIGNAL_MODE"] = "model"
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

    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")

    log_path = LOGS / f"sweep_2025_fail_{scenario['name']}_{tag}.log"
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


def score_scenario(rows: list[dict]) -> dict:
    pass_count = sum(1 for r in rows if r.get("pass_or_fail") == "PASS")
    avg_edge_random = sum(float(r.get("model_vs_random_pct") or 0.0) for r in rows) / max(1, len(rows))
    avg_edge_nifty = sum(float(r.get("model_vs_nifty_pct") or 0.0) for r in rows) / max(1, len(rows))
    avg_sharpe = sum(float(r.get("model_sharpe") or 0.0) for r in rows) / max(1, len(rows))
    return {
        "pass_count": pass_count,
        "avg_edge_random": avg_edge_random,
        "avg_edge_nifty": avg_edge_nifty,
        "avg_sharpe": avg_sharpe,
    }


def render_markdown(results: dict) -> str:
    lines = []
    lines.append("| Scenario | Month | Model Final | Random Final | Nifty Final | Sharpe | vs Random % | vs Nifty % | Result | First Failed Gate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for scenario_name, rows in results["rows_by_scenario"].items():
        for row in rows:
            lines.append(
                "| {scenario} | {month} | {model_final:.2f} | {random_final:.2f} | {nifty_final:.2f} | {sharpe:.3f} | {edge_r:.2f} | {edge_n:.2f} | {result} | {gate} |".format(
                    scenario=scenario_name,
                    month=row["tag"],
                    model_final=float(row.get("model_final") or 0.0),
                    random_final=float(row.get("random_final") or 0.0),
                    nifty_final=float(row.get("nifty_final") or 0.0),
                    sharpe=float(row.get("model_sharpe") or 0.0),
                    edge_r=float(row.get("model_vs_random_pct") or 0.0),
                    edge_n=float(row.get("model_vs_nifty_pct") or 0.0),
                    result=row.get("pass_or_fail", "-"),
                    gate=row.get("first_failed_gate") or "-",
                )
            )

    lines.append("")
    lines.append("| Scenario | Passes (out of 4) | Avg vs Random % | Avg vs Nifty % | Avg Sharpe |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for scenario_name, score in results["scenario_scores"].items():
        lines.append(
            "| {scenario} | {passes} | {edge_r:.2f} | {edge_n:.2f} | {sharpe:.3f} |".format(
                scenario=scenario_name,
                passes=score["pass_count"],
                edge_r=score["avg_edge_random"],
                edge_n=score["avg_edge_nifty"],
                sharpe=score["avg_sharpe"],
            )
        )

    return "\n".join(lines)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)

    rows_by_scenario = {}
    scenario_scores = {}

    for scenario in SCENARIOS:
        print(f"\n=== Scenario: {scenario['name']} ===")
        rows = []
        for tag, start, end in WINDOWS:
            print(f"  Running {tag} ({start} -> {end})")
            rows.append(run_window_for_scenario(tag, start, end, scenario))
        rows_by_scenario[scenario["name"]] = rows
        scenario_scores[scenario["name"]] = score_scenario(rows)

    out_payload = {
        "windows": WINDOWS,
        "scenarios": SCENARIOS,
        "rows_by_scenario": rows_by_scenario,
        "scenario_scores": scenario_scores,
    }

    out_json = LOGS / "sweep_2025_failures_comparison.json"
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    md = render_markdown(out_payload)
    out_md = LOGS / "sweep_2025_failures_comparison.md"
    out_md.write_text(md + "\n", encoding="utf-8")

    print("\n=== Sweep saved ===")
    print(out_json)
    print(out_md)
    print("\n" + md)


if __name__ == "__main__":
    main()