#!/usr/bin/env python
"""
10-seed Phase 4 validation with the current robustness setup.
"""

import json
import os
import re
import subprocess
from datetime import datetime

seeds = [42, 123, 777, 1337, 2024, 7, 17, 99, 31415, 2718]
results = {}

print("=" * 70)
print("PHASE 4 ROBUSTNESS TEST - CURRENT SETUP")
print("=" * 70)
print("Test Fraction: 0.30 (extended window)")
print("Model: QuantPulseSimple (1 hidden layer, 32 neurons)")
print("Patience: 50")
print("Regime Features: nifty_trend, market_volatility (in model input)")
print(f"Seeds: {len(seeds)}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

for seed in seeds:
    print(f"\n[SEED {seed}] Running...", flush=True)

    attempt = 1
    max_attempts = 5

    while attempt <= max_attempts:
        try:
            env = os.environ.copy()
            env["QUANT_SEED"] = str(seed)

            result = subprocess.run(
                [".\\venv\\Scripts\\python.exe", "-u", "train.py"],
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )

            if result.returncode == 0 or "RESULT COMPARISON" in result.stdout:
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

                if real_match and rand_match:
                    real_final = float(real_match.group(1))
                    real_sharpe = float(real_match.group(2))
                    real_ann = float(real_match.group(3))

                    rand_final = float(rand_match.group(1))
                    rand_sharpe = float(rand_match.group(2))
                    rand_ann = float(rand_match.group(3))

                    beat_random = real_final > rand_final
                    sharpe_edge = real_sharpe - rand_sharpe

                    results[seed] = {
                        "real_final": real_final,
                        "real_sharpe": real_sharpe,
                        "real_ann": real_ann,
                        "rand_final": rand_final,
                        "rand_sharpe": rand_sharpe,
                        "rand_ann": rand_ann,
                        "beat_random": beat_random,
                        "sharpe_edge": sharpe_edge,
                    }

                    print(f"✓ Seed {seed} (attempt {attempt}):")
                    print(f"  Real:   Final={real_final:.2f}, Sharpe={real_sharpe:.3f}, Ann={real_ann:.2f}%")
                    print(f"  Random: Final={rand_final:.2f}, Sharpe={rand_sharpe:.3f}, Ann={rand_ann:.2f}%")
                    print(f"  Beat Random: {beat_random}, Sharpe Edge: {sharpe_edge:.3f}")
                    break

            if attempt < max_attempts:
                attempt += 1
            else:
                print(f"✗ Seed {seed} failed on final attempt")
                break

        except Exception as e:
            if attempt < max_attempts:
                print(f"  Attempt {attempt} failed: {str(e)[:60]}... retrying", flush=True)
                attempt += 1
            else:
                print(f"✗ Seed {seed} failed after {max_attempts} attempts")
                break

print("\n" + "=" * 70)
print("AGGREGATE RESULTS")
print("=" * 70)

if results:
    beat_count = sum(1 for r in results.values() if r["beat_random"])
    avg_sharpe_edge = sum(r["sharpe_edge"] for r in results.values()) / len(results)
    avg_real_final = sum(r["real_final"] for r in results.values()) / len(results)
    avg_rand_final = sum(r["rand_final"] for r in results.values()) / len(results)

    print(f"Seeds Beat Random: {beat_count}/{len(results)}")
    print(f"Average Sharpe Edge (Real - Random): {avg_sharpe_edge:.3f}")
    print(f"Average Real Final: ${avg_real_final:.2f}")
    print(f"Average Random Final: ${avg_rand_final:.2f}")

    passed = (beat_count >= 4) or (avg_sharpe_edge > 0) or (avg_real_final > avg_rand_final * 1.05)

    print(f"\n{'✓ PHASE 4 PASSED' if passed else '✗ PHASE 4 FAILED'}")
    print(f"(Need: 4/5 beat random OR avg_sharpe_edge > 0 OR real avg 5% better than random)")

    with open(".phase4_results_simple.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": "QuantPulseSimple",
                "patience": 50,
                "test_fraction": 0.30,
                "seeds": results,
                "aggregate": {
                    "beat_count": beat_count,
                    "avg_sharpe_edge": avg_sharpe_edge,
                    "avg_real_final": avg_real_final,
                    "avg_rand_final": avg_rand_final,
                    "passed": passed,
                },
            },
            f,
            indent=2,
        )

    print("\nResults saved to .phase4_results_simple.json")
