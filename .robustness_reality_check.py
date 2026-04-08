#!/usr/bin/env python
"""Run two robustness tests across 10 seeds:
1) transaction costs only
2) transaction costs + slippage + position limits + execution delay
"""

import json
import os
import re
import subprocess
from datetime import datetime


SEEDS = [42, 123, 777, 1337, 2024, 7, 17, 99, 31415, 2718]
SCENARIOS = [
    {
        "name": "cost_only",
        "env": {
            "QUANT_TRANSACTION_COST": "0.001",
            "QUANT_SLIPPAGE": "0.0002",
            "QUANT_MAX_POSITION_WEIGHT": "0.5",
            "QUANT_EXECUTION_DELAY": "0",
        },
    },
    {
        "name": "cost_slippage_constraints_delay",
        "env": {
            "QUANT_TRANSACTION_COST": "0.001",
            "QUANT_SLIPPAGE": "0.001",
            "QUANT_MAX_POSITION_WEIGHT": "0.33",
            "QUANT_EXECUTION_DELAY": "1",
        },
    },
]


def run_seed(seed, extra_env):
    env = os.environ.copy()
    env["QUANT_SEED"] = str(seed)
    env["QUANT_ENSEMBLE_SEEDS"] = "42,123,777"
    env.update(extra_env)

    result = subprocess.run(
        [".\\venv\\Scripts\\python.exe", "-u", "train.py"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )

    stdout = result.stdout
    real_match = re.search(
        r"REAL MODEL\nFinal Value: ([\d.]+)\nSharpe: ([\d.-]+).*?Annualized Return: ([\d.]+) %",
        stdout,
        re.DOTALL,
    )
    rand_match = re.search(
        r"RANDOM BASELINE\nFinal Value: ([\d.]+)\nSharpe: ([\d.-]+).*?Annualized Return: ([\d.]+) %",
        stdout,
        re.DOTALL,
    )

    if not real_match or not rand_match:
        return None, stdout

    real_final = float(real_match.group(1))
    real_sharpe = float(real_match.group(2))
    real_ann = float(real_match.group(3))
    rand_final = float(rand_match.group(1))
    rand_sharpe = float(rand_match.group(2))
    rand_ann = float(rand_match.group(3))

    return {
        "real_final": real_final,
        "real_sharpe": real_sharpe,
        "real_ann": real_ann,
        "rand_final": rand_final,
        "rand_sharpe": rand_sharpe,
        "rand_ann": rand_ann,
        "beat_random": real_final > rand_final,
        "sharpe_edge": real_sharpe - rand_sharpe,
    }, stdout


def main():
    print("=" * 70)
    print("ROBUSTNESS REALITY CHECK")
    print("=" * 70)
    print(f"Seeds: {len(SEEDS)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Ensemble seeds: 42,123,777")

    all_results = {}

    for scenario in SCENARIOS:
        print("\n" + "=" * 70)
        print(f"SCENARIO: {scenario['name']}")
        print("=" * 70)

        scenario_results = {}

        for seed in SEEDS:
            print(f"\n[SEED {seed}] Running...", flush=True)
            attempts = 1
            max_attempts = 3

            while attempts <= max_attempts:
                result, stdout = run_seed(seed, scenario["env"])

                if result is not None:
                    scenario_results[seed] = result
                    print(f"✓ Seed {seed} (attempt {attempts}):")
                    print(f"  Real:   Final={result['real_final']:.2f}, Sharpe={result['real_sharpe']:.3f}, Ann={result['real_ann']:.2f}%")
                    print(f"  Random: Final={result['rand_final']:.2f}, Sharpe={result['rand_sharpe']:.3f}, Ann={result['rand_ann']:.2f}%")
                    print(f"  Beat Random: {result['beat_random']}, Sharpe Edge: {result['sharpe_edge']:.3f}")
                    break

                attempts += 1
                if attempts > max_attempts:
                    print(f"✗ Seed {seed} failed after {max_attempts} attempts")

        if scenario_results:
            beat_count = sum(1 for r in scenario_results.values() if r["beat_random"])
            avg_sharpe_edge = sum(r["sharpe_edge"] for r in scenario_results.values()) / len(scenario_results)
            avg_real_final = sum(r["real_final"] for r in scenario_results.values()) / len(scenario_results)
            avg_rand_final = sum(r["rand_final"] for r in scenario_results.values()) / len(scenario_results)

            passed = (beat_count >= 4) or (avg_sharpe_edge > 0) or (avg_real_final > avg_rand_final * 1.05)

            print("\nAGGREGATE RESULTS")
            print(f"Seeds Beat Random: {beat_count}/{len(scenario_results)}")
            print(f"Average Sharpe Edge (Real - Random): {avg_sharpe_edge:.3f}")
            print(f"Average Real Final: ${avg_real_final:.2f}")
            print(f"Average Random Final: ${avg_rand_final:.2f}")
            print(f"{'✓ PASSED' if passed else '✗ FAILED'}")

            all_results[scenario["name"]] = {
                "seeds": scenario_results,
                "aggregate": {
                    "beat_count": beat_count,
                    "avg_sharpe_edge": avg_sharpe_edge,
                    "avg_real_final": avg_real_final,
                    "avg_rand_final": avg_rand_final,
                    "passed": passed,
                },
            }

    with open(".robustness_reality_check.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to .robustness_reality_check.json")


if __name__ == "__main__":
    main()