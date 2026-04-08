#!/usr/bin/env python
"""Controlled bad-regime defense sweep.

Uses a small fixed grid and evaluates:
- 2021 edge threshold
- 2022-2024 non-degradation constraints
- score = 2*edge_2021 + edge_2022 + edge_2023 + edge_2024
"""

import json
import os
import subprocess
from datetime import datetime

PYTHON_EXE = ".\\venv\\Scripts\\python.exe"
WF_SCRIPT = ".walk_forward_robustness.py"

WINDOW_2021 = "train_2017_2020_test_2021"
WINDOW_2022 = "train_2018_2021_test_2022"
WINDOW_2023 = "train_2019_2022_test_2023"
WINDOW_2024 = "train_2020_2023_test_2024"
SCENARIO = "base_costs"

COMBOS = [
    {"id": "B0_baseline", "rank_bad": 0.67, "spread_bad": 0.014, "fallback_bad": 0.55, "exposure_bad": 0.65},
    {"id": "C01", "rank_bad": 0.68, "spread_bad": 0.015, "fallback_bad": 0.50, "exposure_bad": 0.60},
    {"id": "C02", "rank_bad": 0.68, "spread_bad": 0.017, "fallback_bad": 0.50, "exposure_bad": 0.60},
    {"id": "C03", "rank_bad": 0.70, "spread_bad": 0.015, "fallback_bad": 0.50, "exposure_bad": 0.60},
    {"id": "C04", "rank_bad": 0.70, "spread_bad": 0.017, "fallback_bad": 0.50, "exposure_bad": 0.60},
    {"id": "C05", "rank_bad": 0.72, "spread_bad": 0.017, "fallback_bad": 0.50, "exposure_bad": 0.60},
    {"id": "C06", "rank_bad": 0.70, "spread_bad": 0.020, "fallback_bad": 0.50, "exposure_bad": 0.60},
    {"id": "C07", "rank_bad": 0.68, "spread_bad": 0.015, "fallback_bad": 0.45, "exposure_bad": 0.60},
    {"id": "C08", "rank_bad": 0.70, "spread_bad": 0.017, "fallback_bad": 0.45, "exposure_bad": 0.60},
    {"id": "C09", "rank_bad": 0.72, "spread_bad": 0.020, "fallback_bad": 0.45, "exposure_bad": 0.60},
    {"id": "C10", "rank_bad": 0.68, "spread_bad": 0.017, "fallback_bad": 0.50, "exposure_bad": 0.55},
    {"id": "C11", "rank_bad": 0.70, "spread_bad": 0.017, "fallback_bad": 0.50, "exposure_bad": 0.55},
    {"id": "C12", "rank_bad": 0.72, "spread_bad": 0.020, "fallback_bad": 0.45, "exposure_bad": 0.55},
]


def get_edge_ratio(blob, window_name):
    ws = blob["windows"][window_name]["scenarios"][SCENARIO]["summary"]
    return float(ws["avg_sharpe_edge"]), float(ws["avg_final_ratio"])


def run_combo(combo):
    env = os.environ.copy()
    env["QUANT_WF_OUTER_SEEDS"] = "42"
    env["QUANT_WF_BASELINE_RUNS"] = "1"
    env["QUANT_WF_SCENARIOS"] = SCENARIO

    env["QUANT_RANK_THRESHOLD_BAD"] = str(combo["rank_bad"])
    env["QUANT_SIGNAL_SPREAD_BAD"] = str(combo["spread_bad"])
    env["QUANT_FALLBACK_REDUCE_FACTOR_BAD"] = str(combo["fallback_bad"])
    env["QUANT_REGIME_EXPOSURE_SCALE_BAD"] = str(combo["exposure_bad"])

    proc = subprocess.run(
        [PYTHON_EXE, "-u", WF_SCRIPT],
        capture_output=True,
        text=True,
        timeout=7200,
        env=env,
    )

    if proc.returncode != 0:
        return {"id": combo["id"], "error": f"non-zero return code {proc.returncode}"}

    with open(".walk_forward_robustness_results.json", "r", encoding="utf-8") as f:
        wf = json.load(f)

    e2021, r2021 = get_edge_ratio(wf, WINDOW_2021)
    e2022, r2022 = get_edge_ratio(wf, WINDOW_2022)
    e2023, r2023 = get_edge_ratio(wf, WINDOW_2023)
    e2024, r2024 = get_edge_ratio(wf, WINDOW_2024)

    pass_2021 = e2021 >= -0.25
    pass_22_24 = (e2022 > 0.5 and r2022 > 1.2 and e2023 > 0.5 and r2023 > 1.2 and e2024 > 0.5 and r2024 > 1.2)
    accept = pass_2021 and pass_22_24

    score = (2.0 * e2021) + e2022 + e2023 + e2024

    return {
        "id": combo["id"],
        "params": combo,
        "metrics": {
            "2021": {"edge": e2021, "ratio": r2021},
            "2022": {"edge": e2022, "ratio": r2022},
            "2023": {"edge": e2023, "ratio": r2023},
            "2024": {"edge": e2024, "ratio": r2024},
        },
        "pass_2021": pass_2021,
        "pass_2022_2024": pass_22_24,
        "accept": accept,
        "score": score,
    }


def main():
    print("=" * 78)
    print("TARGETED 2021 DEFENSE SWEEP")
    print("=" * 78)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Combos: {len(COMBOS)}")

    results = []
    for idx, combo in enumerate(COMBOS, start=1):
        print(f"\n[{idx}/{len(COMBOS)}] Running {combo['id']} ...", flush=True)
        out = run_combo(combo)
        results.append(out)
        if "error" in out:
            print(f"  ERROR: {out['error']}")
            continue
        m = out["metrics"]
        print(
            "  "
            f"2021 edge={m['2021']['edge']:.3f} ratio={m['2021']['ratio']:.3f} | "
            f"2022 edge={m['2022']['edge']:.3f} ratio={m['2022']['ratio']:.3f} | "
            f"2023 edge={m['2023']['edge']:.3f} ratio={m['2023']['ratio']:.3f} | "
            f"2024 edge={m['2024']['edge']:.3f} ratio={m['2024']['ratio']:.3f}"
        )
        print(f"  accept={out['accept']} score={out['score']:.3f}")

    valid = [x for x in results if "error" not in x]
    ranked = sorted(valid, key=lambda x: x["score"], reverse=True)
    accepted = [x for x in ranked if x["accept"]]

    payload = {
        "timestamp": datetime.now().isoformat(),
        "count": len(results),
        "results": results,
        "ranked_ids": [x["id"] for x in ranked],
        "accepted_ids": [x["id"] for x in accepted],
    }

    out_path = ".sweep_2021_defense_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 78)
    print("SWEEP SUMMARY")
    print("=" * 78)
    print(f"Valid runs: {len(valid)}")
    print(f"Accepted runs: {len(accepted)}")
    if accepted:
        print(f"Best accepted: {accepted[0]['id']} score={accepted[0]['score']:.3f}")
    elif ranked:
        print(f"No accepted configs. Best overall: {ranked[0]['id']} score={ranked[0]['score']:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
