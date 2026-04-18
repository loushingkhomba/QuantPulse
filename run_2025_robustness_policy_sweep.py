import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from run_10_window_retests import THRESHOLDS, apply_month_type_profile, parse_metrics, render_report_table


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

# Focused third sweep around the current best candidate.
# Keep governance criteria fixed and search only anti-fragility controls.
SCENARIOS = [
    {"name": "afg_00_control", "env": {}},
    {
        "name": "afg_01_bad_exposure_050",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.50",
        },
    },
    {
        "name": "afg_02_bad_exposure_060",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.60",
        },
    },
    {
        "name": "afg_03_cooldown3_new1",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.55",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "3",
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "1",
        },
    },
    {
        "name": "afg_04_cooldown4_new1_cap042",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.55",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "4",
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "1",
            "QUANT_MAX_POSITION_WEIGHT": "0.42",
        },
    },
    {
        "name": "afg_05_spread013_conf054",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.55",
            "QUANT_SIGNAL_SPREAD_BAD": "0.013",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.54",
        },
    },
    {
        "name": "afg_06_spread014_conf055",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.55",
            "QUANT_SIGNAL_SPREAD_BAD": "0.014",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.55",
        },
    },
    {
        "name": "afg_07_fallback_guard",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.55",
            "QUANT_FALLBACK_CONF_THRESHOLD_BAD": "0.545",
            "QUANT_FALLBACK_REDUCE_FACTOR_BAD": "0.40",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "3",
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "1",
        },
    },
    {
        "name": "afg_08_killswitch_tight",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.50",
            "QUANT_KILL_SWITCH_DRAWDOWN_THRESHOLD": "-0.10",
            "QUANT_KILL_SWITCH_MAX_NEW_POSITIONS": "0",
            "QUANT_KILL_SWITCH_FORCE_EXIT": "1",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "3",
        },
    },
    {
        "name": "afg_09_weight_temp145_cap040",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.55",
            "QUANT_WEIGHT_TEMPERATURE": "1.45",
            "QUANT_MAX_POSITION_WEIGHT": "0.40",
        },
    },
    {
        "name": "afg_10_stack_strict",
        "env": {
            "QUANT_REGIME_EXPOSURE_SCALE_BAD": "0.50",
            "QUANT_TURNOVER_COOLDOWN_DAYS": "3",
            "QUANT_MAX_NEW_POSITIONS_PER_DAY": "1",
            "QUANT_SIGNAL_SPREAD_BAD": "0.014",
            "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.54",
            "QUANT_MAX_POSITION_WEIGHT": "0.42",
            "QUANT_WEIGHT_TEMPERATURE": "1.40",
        },
    },
]

ROBUSTNESS_PROMOTION_MIN = 3
ROLLBACK_MAX_FOR_ELIGIBLE = 8
PASS_MIN_FOR_ELIGIBLE = 8


def _build_clean_window_env(start: str, end: str, runtime_env: dict, scenario_env: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("QUANT_")}
    env["QUANT_SIGNAL_MODE"] = "model"
    env["QUANT_CARRY_FORWARD_OPEN"] = "1"
    env["QUANT_TRANSACTION_COST"] = os.environ.get("QUANT_TRANSACTION_COST", "0.0005")
    env["QUANT_TEST_START"] = start
    env["QUANT_TEST_END"] = end
    env["QUANT_SPLIT_DATE"] = start
    env.update(runtime_env or {})
    env.update(scenario_env or {})
    return env


def _extract_dict_after_prefix(log_text: str, prefix: str) -> dict:
    pattern = re.escape(prefix) + r"\s*(\{.*?\})"
    match = re.search(pattern, log_text, flags=re.S)
    if not match:
        return {}
    try:
        payload = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def run_window(scenario: dict, tag: str, start: str, end: str) -> dict:
    profile_args, profile_metadata = apply_month_type_profile(start)
    env = _build_clean_window_env(
        start,
        end,
        profile_metadata.get("runtime_env", {}),
        scenario.get("env", {}),
    )

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

    print(f"Running {scenario['name']} / {tag} ...")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")

    log_path = LOGS / f"sweep_2025_robust_policy_{scenario['name']}_{tag}.log"
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for {scenario['name']} / {tag}. See {log_path}")

    metrics = parse_metrics(combined)
    robustness = _extract_dict_after_prefix(combined, "Robustness sensitivity:")
    policy = _extract_dict_after_prefix(combined, "Promotion policy:")

    metrics["tag"] = tag
    metrics["start"] = start
    metrics["end"] = end
    metrics["scenario"] = scenario["name"]
    metrics["profile"] = profile_metadata
    metrics["robustness"] = {
        "overall_pass": bool(robustness.get("overall_pass")) if robustness else None,
        "scenario_count": robustness.get("scenario_count") if robustness else None,
        "failing_scenarios": robustness.get("failing_scenarios") if robustness else None,
    }
    metrics["policy"] = {
        "decision": policy.get("decision") if policy else None,
        "reasons": policy.get("reasons") if policy else None,
        "promote_streak": policy.get("promote_streak") if policy else None,
        "rollback_streak": policy.get("rollback_streak") if policy else None,
    }
    metrics["log_path"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return metrics


def aggregate_rows(rows: list[dict]) -> dict:
    policy_counts = {"promote": 0, "hold": 0, "rollback": 0, "unknown": 0}
    for row in rows:
        decision = (row.get("policy", {}).get("decision") or "unknown").strip().lower()
        if decision not in policy_counts:
            decision = "unknown"
        policy_counts[decision] += 1

    pass_count = sum(1 for row in rows if row.get("pass_or_fail") == "PASS")
    robustness_pass_count = sum(1 for row in rows if row.get("robustness", {}).get("overall_pass") is True)

    avg_final_ratio = sum(float(row.get("model_final") or 0.0) for row in rows) / max(1, len(rows))
    avg_sharpe = sum(float(row.get("model_sharpe") or 0.0) for row in rows) / max(1, len(rows))

    eligible = (
        robustness_pass_count >= ROBUSTNESS_PROMOTION_MIN
        and policy_counts["rollback"] <= ROLLBACK_MAX_FOR_ELIGIBLE
        and pass_count >= PASS_MIN_FOR_ELIGIBLE
    )

    # Robustness-first ranking for deployment readiness.
    ranking_tuple = (
        1 if eligible else 0,
        robustness_pass_count,
        -policy_counts["rollback"],
        pass_count,
        avg_sharpe,
    )

    return {
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "pass_rate_pct": (pass_count / max(1, len(rows))) * 100.0,
        "robustness_pass_count": robustness_pass_count,
        "robustness_fail_count": len(rows) - robustness_pass_count,
        "policy_counts": policy_counts,
        "eligible_for_promotion": bool(eligible),
        "avg_model_final": avg_final_ratio,
        "avg_model_sharpe": avg_sharpe,
        "ranking_tuple": ranking_tuple,
    }


def render_scenario_markdown(name: str, rows: list[dict], aggregate: dict) -> str:
    report = {
        "windows": rows,
        "thresholds": THRESHOLDS,
        "report_path": f"scenario:{name}",
    }
    lines = [f"## Scenario: {name}"]
    lines.append(
        "Summary: pass={pass_count}/12, robustness_pass={robust_pass}/12, policy(promote/hold/rollback)={promote}/{hold}/{rollback}".format(
            pass_count=aggregate["pass_count"],
            robust_pass=aggregate["robustness_pass_count"],
            promote=aggregate["policy_counts"]["promote"],
            hold=aggregate["policy_counts"]["hold"],
            rollback=aggregate["policy_counts"]["rollback"],
        )
    )
    lines.append("")
    lines.append(render_report_table(report))
    return "\n".join(lines)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)

    rows_by_scenario = {}
    aggregate_by_scenario = {}

    for scenario in SCENARIOS:
        rows = []
        print(f"\n=== Scenario: {scenario['name']} ===")
        for tag, start, end in WINDOWS:
            rows.append(run_window(scenario, tag, start, end))
        rows_by_scenario[scenario["name"]] = rows
        aggregate_by_scenario[scenario["name"]] = aggregate_rows(rows)

    ranked = sorted(
        aggregate_by_scenario.items(),
        key=lambda kv: kv[1]["ranking_tuple"],
        reverse=True,
    )

    out_payload = {
        "windows": WINDOWS,
        "scenarios": SCENARIOS,
        "rows_by_scenario": rows_by_scenario,
        "aggregate_by_scenario": aggregate_by_scenario,
        "ranked_scenarios": [
            {
                "name": name,
                "summary": summary,
            }
            for name, summary in ranked
        ],
    }

    out_json = LOGS / "sweep_2025_robustness_policy_comparison.json"
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    md_lines = ["# 2025 Robustness + Policy Sweep", ""]
    md_lines.append("## Ranking")
    md_lines.append(
        "Eligibility rule: robust_passes >= {rp}, rollback <= {rb}, pass_count >= {pc}".format(
            rp=ROBUSTNESS_PROMOTION_MIN,
            rb=ROLLBACK_MAX_FOR_ELIGIBLE,
            pc=PASS_MIN_FOR_ELIGIBLE,
        )
    )
    md_lines.append("| Rank | Scenario | Eligible | Passes | Robust Passes | Promote | Hold | Rollback | Avg Sharpe |")
    md_lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx, (name, summary) in enumerate(ranked, start=1):
        md_lines.append(
            "| {rank} | {name} | {eligible} | {passes}/12 | {robust}/12 | {promote} | {hold} | {rollback} | {sharpe:.3f} |".format(
                rank=idx,
                name=name,
                eligible="YES" if summary["eligible_for_promotion"] else "NO",
                passes=summary["pass_count"],
                robust=summary["robustness_pass_count"],
                promote=summary["policy_counts"]["promote"],
                hold=summary["policy_counts"]["hold"],
                rollback=summary["policy_counts"]["rollback"],
                sharpe=summary["avg_model_sharpe"],
            )
        )

    for name, _ in ranked:
        md_lines.append("")
        md_lines.append(render_scenario_markdown(name, rows_by_scenario[name], aggregate_by_scenario[name]))

    out_md = LOGS / "sweep_2025_robustness_policy_comparison.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    best_name, best_summary = ranked[0]
    print("\n=== Robustness/Policy sweep complete ===")
    print("Best scenario:", best_name)
    print(
        {
            "pass_count": best_summary["pass_count"],
            "robustness_pass_count": best_summary["robustness_pass_count"],
            "policy_counts": best_summary["policy_counts"],
            "avg_model_sharpe": round(best_summary["avg_model_sharpe"], 3),
        }
    )
    print("Saved JSON:", out_json)
    print("Saved Markdown:", out_md)


if __name__ == "__main__":
    main()
