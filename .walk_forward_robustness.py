#!/usr/bin/env python
"""Walk-forward robustness validation.

Windows:
- Train 2017-2020 -> Test 2021
- Train 2017-2021 -> Test 2022
- Train 2017-2022 -> Test 2023
- Train 2017-2023 -> Test 2024
"""

import json
import os
import re
import statistics
import subprocess
from datetime import datetime

PYTHON_EXE = ".\\venv\\Scripts\\python.exe"
TRAIN_SCRIPT = "train.py"

WINDOWS = [
    {"name": "train_2017_2020_test_2021", "train_start": "2017-01-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"name": "train_2018_2021_test_2022", "train_start": "2018-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"name": "train_2019_2022_test_2023", "train_start": "2019-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"name": "train_2020_2023_test_2024", "train_start": "2020-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
]

SCENARIOS = [
    {
        "name": "base_costs",
        "env": {
            "QUANT_TRANSACTION_COST": "0.0005",
            "QUANT_SLIPPAGE": "0.0002",
            "QUANT_MAX_POSITION_WEIGHT": "0.50",
            "QUANT_EXECUTION_DELAY": "0",
            "QUANT_MISSING_DATA_RATE": "0.0",
        },
    },
    {
        "name": "high_costs",
        "env": {
            "QUANT_TRANSACTION_COST": "0.0010",
            "QUANT_SLIPPAGE": "0.0010",
            "QUANT_MAX_POSITION_WEIGHT": "0.33",
            "QUANT_EXECUTION_DELAY": "1",
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

GATES = {
    "min_seed_win_rate": 0.80,
    "min_avg_sharpe_edge": 0.0,
    "min_worst_seed_avg_sharpe_edge": -0.25,
    "min_avg_final_ratio": 1.05,
}

DEFAULT_OUTER_SEEDS = [42, 123, 777, 1337, 2024]


def parse_seed_list(raw):
    if not raw.strip():
        return DEFAULT_OUTER_SEEDS
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def filter_by_names(items, raw_filter):
    if not raw_filter.strip():
        return items
    allowed = {x.strip() for x in raw_filter.split(",") if x.strip()}
    return [x for x in items if x["name"] in allowed]


def parse_section_metrics(stdout, section_name):
    pat = re.compile(
        rf"{section_name}\n"
        r"Final Value: ([\d.]+)\n"
        r"Sharpe: ([\d.-]+)\n"
        r"Max Drawdown: ([\d.-]+) %\n"
        r"Annualized Return: ([\d.-]+) %\n"
        r"Trade Days: (\d+)\n"
        r"Trades Executed: (\d+)",
        re.MULTILINE,
    )
    m = pat.search(stdout)
    if not m:
        return None
    return {
        "final": float(m.group(1)),
        "sharpe": float(m.group(2)),
        "max_dd_pct": float(m.group(3)),
        "ann_pct": float(m.group(4)),
        "trade_days": int(m.group(5)),
        "trades": int(m.group(6)),
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
        "rand_final": rand["final"],
        "rand_sharpe": rand["sharpe"],
        "sharpe_edge": real["sharpe"] - rand["sharpe"],
        "final_ratio": (real["final"] / rand["final"]) if rand["final"] > 0 else 0.0,
        "beat_random": real["final"] > rand["final"],
    }


def evaluate(runs):
    if not runs:
        return {
            "seed_win_rate": 0.0,
            "avg_sharpe_edge": -999.0,
            "worst_seed_avg_sharpe_edge": -999.0,
            "avg_final_ratio": 0.0,
            "passed": False,
            "gate_checks": {k: False for k in ["seed_win_rate", "avg_sharpe_edge", "worst_seed_avg_sharpe_edge", "avg_final_ratio"]},
        }

    by_seed = {}
    for r in runs:
        by_seed.setdefault(r["outer_seed"], []).append(r)

    seed_avg_edges = {}
    seed_passes = 0
    for seed, items in by_seed.items():
        avg_edge = statistics.mean(x["sharpe_edge"] for x in items)
        avg_ratio = statistics.mean(x["final_ratio"] for x in items)
        seed_avg_edges[seed] = avg_edge
        if avg_edge > 0 and avg_ratio > 1.0:
            seed_passes += 1

    seed_win_rate = seed_passes / max(1, len(by_seed))
    avg_sharpe_edge = statistics.mean(x["sharpe_edge"] for x in runs)
    avg_final_ratio = statistics.mean(x["final_ratio"] for x in runs)
    worst_seed_avg_sharpe_edge = min(seed_avg_edges.values())

    gate_checks = {
        "seed_win_rate": seed_win_rate >= GATES["min_seed_win_rate"],
        "avg_sharpe_edge": avg_sharpe_edge >= GATES["min_avg_sharpe_edge"],
        "worst_seed_avg_sharpe_edge": worst_seed_avg_sharpe_edge >= GATES["min_worst_seed_avg_sharpe_edge"],
        "avg_final_ratio": avg_final_ratio >= GATES["min_avg_final_ratio"],
    }

    return {
        "seed_win_rate": seed_win_rate,
        "avg_sharpe_edge": avg_sharpe_edge,
        "worst_seed_avg_sharpe_edge": worst_seed_avg_sharpe_edge,
        "avg_final_ratio": avg_final_ratio,
        "passed": all(gate_checks.values()),
        "gate_checks": gate_checks,
    }


def main():
    outer_seeds = parse_seed_list(os.getenv("QUANT_WF_OUTER_SEEDS", ""))
    baseline_runs = int(os.getenv("QUANT_WF_BASELINE_RUNS", "2"))
    windows = filter_by_names(WINDOWS, os.getenv("QUANT_WF_WINDOWS", ""))
    scenarios = filter_by_names(SCENARIOS, os.getenv("QUANT_WF_SCENARIOS", ""))

    output = {
        "timestamp": datetime.now().isoformat(),
        "outer_seeds": outer_seeds,
        "baseline_runs": baseline_runs,
        "gates": GATES,
        "windows": {},
    }

    print("=" * 78)
    print("WALK-FORWARD ROBUSTNESS CHECK")
    print("=" * 78)
    print(f"Outer seeds: {len(outer_seeds)}")
    print(f"Random baselines per seed: {baseline_runs}")

    for win in windows:
        print("\n" + "-" * 78)
        print(f"WINDOW: {win['name']}")
        print("-" * 78)
        output["windows"][win["name"]] = {"scenarios": {}}

        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")
            runs = []

            for seed in outer_seeds:
                print(f"  seed {seed}")
                for rb in range(baseline_runs):
                    env = os.environ.copy()
                    env["QUANT_SEED"] = str(seed)
                    env["QUANT_RANDOM_BASELINE_SEED"] = str(seed * 1000 + rb + 1)
                    env["QUANT_SPLIT_DATE"] = win["test_start"]
                    env["QUANT_TRAIN_START"] = win["train_start"]
                    env["QUANT_TRAIN_END"] = win["train_end"]
                    env["QUANT_TEST_START"] = win["test_start"]
                    env["QUANT_TEST_END"] = win["test_end"]
                    env["QUANT_SIGNAL_MODE"] = "model"
                    env.update(scenario["env"])

                    metrics = None
                    max_attempts = 2
                    for attempt in range(1, max_attempts + 1):
                        proc = subprocess.run(
                            [PYTHON_EXE, "-u", TRAIN_SCRIPT],
                            capture_output=True,
                            text=True,
                            timeout=260,
                            env=env,
                        )
                        metrics = parse_metrics(proc.stdout)
                        if metrics is not None:
                            break
                        if attempt < max_attempts:
                            print(f"    retry rb={rb + 1} attempt={attempt + 1}")

                    if metrics is None:
                        print(f"    x rb={rb + 1}: parse failed (rc={proc.returncode})")
                        continue
                    metrics["outer_seed"] = seed
                    runs.append(metrics)
                    print(f"    ok rb={rb + 1}: edge={metrics['sharpe_edge']:.3f}, ratio={metrics['final_ratio']:.3f}")

            summary = evaluate(runs)
            output["windows"][win["name"]]["scenarios"][scenario["name"]] = {
                "summary": summary,
                "run_count": len(runs),
            }

            print("  Summary")
            print(f"    runs: {len(runs)}")
            print(f"    seed_win_rate: {summary['seed_win_rate']:.3f}")
            print(f"    avg_sharpe_edge: {summary['avg_sharpe_edge']:.3f}")
            print(f"    worst_seed_avg_sharpe_edge: {summary['worst_seed_avg_sharpe_edge']:.3f}")
            print(f"    avg_final_ratio: {summary['avg_final_ratio']:.3f}")
            print(f"    passed: {summary['passed']}")

    all_pass = True
    for w in output["windows"].values():
        for s in w["scenarios"].values():
            if not s["summary"]["passed"]:
                all_pass = False

    output["overall_pass"] = all_pass

    out_path = ".walk_forward_robustness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 78)
    print(f"OVERALL WALK-FORWARD PASS: {all_pass}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
