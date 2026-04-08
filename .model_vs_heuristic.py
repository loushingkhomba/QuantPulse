#!/usr/bin/env python
"""Model vs heuristic vs random ranking comparison.

Pass condition:
model > momentum > random
for average final value and average Sharpe across tested seeds/scenarios.
"""

import json
import os
import re
import statistics
import subprocess
from datetime import datetime

PYTHON_EXE = ".\\venv\\Scripts\\python.exe"
TRAIN_SCRIPT = "train.py"

SEEDS = [42, 123, 777, 1337, 2024]

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

STRATEGIES = ["model", "momentum", "random"]


def parse_seed_list(raw):
    if not raw.strip():
        return SEEDS
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def filter_scenarios(raw):
    if not raw.strip():
        return SCENARIOS
    allowed = {x.strip() for x in raw.split(",") if x.strip()}
    return [s for s in SCENARIOS if s["name"] in allowed]


def parse_section_metrics(stdout, section_name):
    pat = re.compile(
        rf"{section_name}\n"
        r"Final Value: ([\d.]+)\n"
        r"Sharpe: ([\d.-]+)",
        re.MULTILINE,
    )
    m = pat.search(stdout.replace("\r\n", "\n"))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def main():
    seeds = parse_seed_list(os.getenv("QUANT_MVH_SEEDS", ""))
    scenarios = filter_scenarios(os.getenv("QUANT_MVH_SCENARIOS", ""))

    results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": {},
    }

    print("=" * 78)
    print("MODEL VS HEURISTIC CHECK")
    print("=" * 78)

    for scenario in scenarios:
        name = scenario["name"]
        print("\n" + "-" * 78)
        print(f"SCENARIO: {name}")
        print("-" * 78)

        strat_runs = {s: [] for s in STRATEGIES}

        for seed in seeds:
            print(f"seed {seed}")
            for strat in STRATEGIES:
                env = os.environ.copy()
                env["QUANT_SEED"] = str(seed)
                env["QUANT_RANDOM_BASELINE_SEED"] = str(seed * 1000 + 99)
                env["QUANT_SIGNAL_MODE"] = strat
                env.update(scenario["env"])

                parsed = None
                max_attempts = 2
                for attempt in range(1, max_attempts + 1):
                    proc = subprocess.run(
                        [PYTHON_EXE, "-u", TRAIN_SCRIPT],
                        capture_output=True,
                        text=True,
                        timeout=260,
                        env=env,
                    )
                    parsed = parse_section_metrics(proc.stdout, "REAL MODEL")
                    if parsed is not None:
                        break
                    if attempt < max_attempts:
                        print(f"  retry {strat} attempt={attempt + 1}")

                if parsed is None:
                    print(f"  x {strat}: parse failed (rc={proc.returncode})")
                    continue

                final_v, sharpe_v = parsed
                strat_runs[strat].append({"final": final_v, "sharpe": sharpe_v, "seed": seed})
                print(f"  ok {strat}: final={final_v:.2f}, sharpe={sharpe_v:.3f}")

        summary = {}
        for strat, items in strat_runs.items():
            if items:
                summary[strat] = {
                    "avg_final": statistics.mean(x["final"] for x in items),
                    "avg_sharpe": statistics.mean(x["sharpe"] for x in items),
                    "count": len(items),
                }
            else:
                summary[strat] = {"avg_final": -1.0, "avg_sharpe": -1.0, "count": 0}

        model_gt_mom_final = summary["model"]["avg_final"] > summary["momentum"]["avg_final"]
        mom_gt_rand_final = summary["momentum"]["avg_final"] > summary["random"]["avg_final"]
        model_gt_mom_sharpe = summary["model"]["avg_sharpe"] > summary["momentum"]["avg_sharpe"]
        mom_gt_rand_sharpe = summary["momentum"]["avg_sharpe"] > summary["random"]["avg_sharpe"]

        ordering_pass = model_gt_mom_final and mom_gt_rand_final and model_gt_mom_sharpe and mom_gt_rand_sharpe

        results["scenarios"][name] = {
            "summary": summary,
            "ordering": {
                "model_gt_momentum_final": model_gt_mom_final,
                "momentum_gt_random_final": mom_gt_rand_final,
                "model_gt_momentum_sharpe": model_gt_mom_sharpe,
                "momentum_gt_random_sharpe": mom_gt_rand_sharpe,
                "pass": ordering_pass,
            },
        }

        print("Summary")
        print(f"  model avg final/sharpe: {summary['model']['avg_final']:.2f} / {summary['model']['avg_sharpe']:.3f}")
        print(f"  momentum avg final/sharpe: {summary['momentum']['avg_final']:.2f} / {summary['momentum']['avg_sharpe']:.3f}")
        print(f"  random avg final/sharpe: {summary['random']['avg_final']:.2f} / {summary['random']['avg_sharpe']:.3f}")
        print(f"  pass model>momentum>random: {ordering_pass}")

    overall_pass = all(v["ordering"]["pass"] for v in results["scenarios"].values())
    results["overall_pass"] = overall_pass

    out_path = ".model_vs_heuristic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print(f"OVERALL ORDERING PASS: {overall_pass}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
