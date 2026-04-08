#!/usr/bin/env python
"""Universal robustness sweep for the trading system.

Runs a scenario matrix across many outer seeds and multiple random baselines per seed,
and evaluates pass/fail gates that detect configuration overfitting.
"""

import json
import os
import re
import statistics
import subprocess
from datetime import datetime

PYTHON_EXE = ".\\venv\\Scripts\\python.exe"
TRAIN_SCRIPT = "train.py"

# Wider outer-seed set than standard Phase 4.
DEFAULT_OUTER_SEEDS = [
    42, 123, 777, 1337, 2024, 7, 17, 99, 31415, 2718,
    11, 19, 29, 37, 43, 53, 67, 79, 89, 101,
]

# Scenario matrix for robustness envelope.
SCENARIOS = [
    {
        "name": "base_costs",
        "env": {
            "QUANT_TRANSACTION_COST": "0.0005",
            "QUANT_SLIPPAGE": "0.0002",
            "QUANT_MAX_POSITION_WEIGHT": "0.50",
            "QUANT_EXECUTION_DELAY": "0",
        },
    },
    {
        "name": "high_costs",
        "env": {
            "QUANT_TRANSACTION_COST": "0.0010",
            "QUANT_SLIPPAGE": "0.0010",
            "QUANT_MAX_POSITION_WEIGHT": "0.33",
            "QUANT_EXECUTION_DELAY": "1",
        },
    },
    {
        "name": "stress_costs_delay2",
        "env": {
            "QUANT_TRANSACTION_COST": "0.0015",
            "QUANT_SLIPPAGE": "0.0012",
            "QUANT_MAX_POSITION_WEIGHT": "0.25",
            "QUANT_EXECUTION_DELAY": "2",
            "QUANT_MISSING_DATA_RATE": "0.0",
        },
    },
    {
        "name": "delay2_cap02",
        "env": {
            "QUANT_TRANSACTION_COST": "0.0010",
            "QUANT_SLIPPAGE": "0.0010",
            "QUANT_MAX_POSITION_WEIGHT": "0.20",
            "QUANT_EXECUTION_DELAY": "2",
            "QUANT_MISSING_DATA_RATE": "0.0",
        },
    },
    {
        "name": "partial_missing_data",
        "env": {
            "QUANT_TRANSACTION_COST": "0.0010",
            "QUANT_SLIPPAGE": "0.0010",
            "QUANT_MAX_POSITION_WEIGHT": "0.33",
            "QUANT_EXECUTION_DELAY": "1",
            "QUANT_MISSING_DATA_RATE": "0.15",
        },
    },
]

# Universal robustness gates.
GATES = {
    "min_seed_win_rate": 0.80,
    "min_avg_sharpe_edge": 0.0,
    "min_worst_seed_avg_sharpe_edge": -0.25,
    "min_avg_final_ratio": 1.05,
}


def parse_seed_list(raw):
    if not raw.strip():
        return DEFAULT_OUTER_SEEDS
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def selected_scenarios(all_scenarios, raw_filter):
    if not raw_filter.strip():
        return all_scenarios
    allowed = {x.strip() for x in raw_filter.split(",") if x.strip()}
    return [s for s in all_scenarios if s["name"] in allowed]

def parse_section_metrics(stdout, section_name):
    pattern = re.compile(
        rf"{section_name}\n"
        r"Final Value: ([\d.]+)\n"
        r"Sharpe: ([\d.-]+)\n"
        r"Max Drawdown: ([\d.-]+) %\n"
        r"Annualized Return: ([\d.-]+) %\n"
        r"Trade Days: (\d+)\n"
        r"Trades Executed: (\d+)",
        re.MULTILINE,
    )
    match = pattern.search(stdout)
    if not match:
        return None
    return {
        "final": float(match.group(1)),
        "sharpe": float(match.group(2)),
        "max_dd_pct": float(match.group(3)),
        "ann_pct": float(match.group(4)),
        "trade_days": int(match.group(5)),
        "trades": int(match.group(6)),
    }


def parse_metrics(stdout):
    stdout = stdout.replace("\r\n", "\n")
    real = parse_section_metrics(stdout, "REAL MODEL")
    rand = parse_section_metrics(stdout, "RANDOM BASELINE")
    if real is None or rand is None:
        return None

    return {
        "real_final": real["final"],
        "real_sharpe": real["sharpe"],
        "real_max_dd_pct": real["max_dd_pct"],
        "real_ann_pct": real["ann_pct"],
        "real_trade_days": real["trade_days"],
        "real_trades": real["trades"],
        "rand_final": rand["final"],
        "rand_sharpe": rand["sharpe"],
        "rand_max_dd_pct": rand["max_dd_pct"],
        "rand_ann_pct": rand["ann_pct"],
        "rand_trade_days": rand["trade_days"],
        "rand_trades": rand["trades"],
        "sharpe_edge": real["sharpe"] - rand["sharpe"],
        "final_ratio": (real["final"] / rand["final"]) if rand["final"] > 0 else 0.0,
        "beat_random": real["final"] > rand["final"],
    }


def run_once(outer_seed, baseline_seed, scenario_env):
    env = os.environ.copy()
    env["QUANT_SEED"] = str(outer_seed)
    env["QUANT_RANDOM_BASELINE_SEED"] = str(baseline_seed)
    env.update(scenario_env)

    result = subprocess.run(
        [PYTHON_EXE, "-u", TRAIN_SCRIPT],
        capture_output=True,
        text=True,
        timeout=220,
        env=env,
    )

    metrics = parse_metrics(result.stdout)
    return metrics, result


def evaluate_scenario(scenario_name, scenario_runs):
    per_seed = {}

    for run in scenario_runs:
        key = str(run["outer_seed"])
        per_seed.setdefault(key, [])
        per_seed[key].append(run)

    if not scenario_runs:
        return {
            "scenario": scenario_name,
            "outer_seed_count": 0,
            "total_runs": 0,
            "seed_win_rate": 0.0,
            "avg_sharpe_edge": -999.0,
            "worst_seed_avg_sharpe_edge": -999.0,
            "avg_final_ratio": 0.0,
            "gate_checks": {
                "seed_win_rate": False,
                "avg_sharpe_edge": False,
                "worst_seed_avg_sharpe_edge": False,
                "avg_final_ratio": False,
            },
            "passed": False,
            "seed_summaries": {},
        }

    seed_summaries = {}
    for seed, runs in per_seed.items():
        avg_edge = statistics.mean(x["sharpe_edge"] for x in runs)
        avg_ratio = statistics.mean(x["final_ratio"] for x in runs)
        win_rate = statistics.mean(1.0 if x["beat_random"] else 0.0 for x in runs)
        seed_summaries[seed] = {
            "avg_sharpe_edge": avg_edge,
            "avg_final_ratio": avg_ratio,
            "baseline_win_rate": win_rate,
        }

    seed_pass_count = sum(1 for s in seed_summaries.values() if s["avg_sharpe_edge"] > 0 and s["avg_final_ratio"] > 1.0)
    seed_win_rate = seed_pass_count / max(1, len(seed_summaries))

    avg_sharpe_edge = statistics.mean(x["sharpe_edge"] for x in scenario_runs)
    avg_final_ratio = statistics.mean(x["final_ratio"] for x in scenario_runs)
    worst_seed_avg_edge = min(s["avg_sharpe_edge"] for s in seed_summaries.values()) if seed_summaries else -999

    gate_checks = {
        "seed_win_rate": seed_win_rate >= GATES["min_seed_win_rate"],
        "avg_sharpe_edge": avg_sharpe_edge >= GATES["min_avg_sharpe_edge"],
        "worst_seed_avg_sharpe_edge": worst_seed_avg_edge >= GATES["min_worst_seed_avg_sharpe_edge"],
        "avg_final_ratio": avg_final_ratio >= GATES["min_avg_final_ratio"],
    }
    passed = all(gate_checks.values())

    return {
        "scenario": scenario_name,
        "outer_seed_count": len(seed_summaries),
        "total_runs": len(scenario_runs),
        "seed_win_rate": seed_win_rate,
        "avg_sharpe_edge": avg_sharpe_edge,
        "worst_seed_avg_sharpe_edge": worst_seed_avg_edge,
        "avg_final_ratio": avg_final_ratio,
        "gate_checks": gate_checks,
        "passed": passed,
        "seed_summaries": seed_summaries,
    }


def main():
    outer_seeds = parse_seed_list(os.getenv("QUANT_UNI_OUTER_SEEDS", ""))
    random_baseline_runs = int(os.getenv("QUANT_UNI_BASELINE_RUNS", "3"))
    scenarios = selected_scenarios(SCENARIOS, os.getenv("QUANT_UNI_SCENARIOS", ""))

    print("=" * 78)
    print("UNIVERSAL ROBUSTNESS CHECK")
    print("=" * 78)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outer seeds: {len(outer_seeds)}")
    print(f"Random baselines per seed: {random_baseline_runs}")
    print(f"Scenario count: {len(scenarios)}")
    print("Gates:")
    for k, v in GATES.items():
        print(f"  - {k}: {v}")

    full = {
        "timestamp": datetime.now().isoformat(),
        "outer_seeds": outer_seeds,
        "random_baseline_runs": random_baseline_runs,
        "gates": GATES,
        "scenarios": {},
    }

    for scenario in scenarios:
        name = scenario["name"]
        print("\n" + "-" * 78)
        print(f"SCENARIO: {name}")
        print("-" * 78)

        scenario_runs = []

        for outer_seed in outer_seeds:
            print(f"[seed {outer_seed}]", flush=True)
            for rb_idx in range(random_baseline_runs):
                baseline_seed = outer_seed * 1000 + rb_idx + 1
                metrics, result = run_once(outer_seed, baseline_seed, scenario["env"])
                if metrics is None:
                    print(f"  x rb={rb_idx + 1}: parse failed (rc={result.returncode})")
                    continue

                metrics["outer_seed"] = outer_seed
                metrics["baseline_seed"] = baseline_seed
                scenario_runs.append(metrics)
                print(
                    "  ok "
                    f"rb={rb_idx + 1} edge={metrics['sharpe_edge']:.3f} "
                    f"ratio={metrics['final_ratio']:.3f}"
                )

        summary = evaluate_scenario(name, scenario_runs)
        full["scenarios"][name] = {
            "env": scenario["env"],
            "summary": summary,
            "runs": scenario_runs,
        }

        print("\nScenario summary")
        print(f"  total runs: {summary['total_runs']}")
        print(f"  seed win rate: {summary['seed_win_rate']:.3f}")
        print(f"  avg sharpe edge: {summary['avg_sharpe_edge']:.3f}")
        print(f"  worst seed avg edge: {summary['worst_seed_avg_sharpe_edge']:.3f}")
        print(f"  avg final ratio: {summary['avg_final_ratio']:.3f}")
        print(f"  passed: {summary['passed']}")

    # Overall pass requires every scenario pass.
    overall_pass = all(x["summary"]["passed"] for x in full["scenarios"].values())
    full["overall_pass"] = overall_pass

    out_path = ".universal_robustness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)

    print("\n" + "=" * 78)
    print("OVERALL RESULT")
    print("=" * 78)
    for name, blob in full["scenarios"].items():
        print(f"{name}: {'PASS' if blob['summary']['passed'] else 'FAIL'}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
