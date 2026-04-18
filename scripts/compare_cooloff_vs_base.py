import json
import statistics
from pathlib import Path


def load(path):
    j = json.loads(Path(path).read_text())
    rows = []
    for w in j["windows"]:
        test = w["window"]["test_start"][:7]
        mf = float(w["model"]["final_value"])
        rf = float(w["random"]["final_value"])
        ms = float(w["model"]["sharpe"])
        rs = float(w["random"]["sharpe"])
        rows.append(
            {
                "test": test,
                "final_edge": mf - rf,
                "sharpe_edge": ms - rs,
            }
        )

    final_edges = [r["final_edge"] for r in rows]
    sharpe_edges = [r["sharpe_edge"] for r in rows]
    sorted_edges = sorted(final_edges)
    n = len(sorted_edges)
    q1 = sorted_edges[int((n - 1) * 0.25)]
    q3 = sorted_edges[int((n - 1) * 0.75)]
    worst_decile = sorted_edges[max(0, int(n * 0.1) - 1)]

    return {
        "rows": rows,
        "median_final_edge": statistics.median(final_edges),
        "mean_final_edge": statistics.mean(final_edges),
        "iqr_final_edge": q3 - q1,
        "worst_decile_final_edge": worst_decile,
        "win_final": sum(1 for x in final_edges if x > 0),
        "win_sharpe": sum(1 for x in sharpe_edges if x > 0),
    }


base = load("logs/strict_walkforward_step5_12m_h3_thr25bps.json")
cool = load("logs/strict_walkforward_step5_12m_h3_thr25bps_cooloff.json")

print("BASE", {k: round(v, 2) if isinstance(v, float) else v for k, v in base.items() if k != "rows"})
print("COOL", {k: round(v, 2) if isinstance(v, float) else v for k, v in cool.items() if k != "rows"})

print("\nWorst 3 months by final_edge (BASE):")
for r in sorted(base["rows"], key=lambda x: x["final_edge"])[:3]:
    print(r["test"], round(r["final_edge"], 2), round(r["sharpe_edge"], 3))

print("\nWorst 3 months by final_edge (COOL):")
for r in sorted(cool["rows"], key=lambda x: x["final_edge"])[:3]:
    print(r["test"], round(r["final_edge"], 2), round(r["sharpe_edge"], 3))

print("\nPer-month delta (COOL - BASE) on final_edge:")
base_map = {r["test"]: r for r in base["rows"]}
for r in cool["rows"]:
    b = base_map[r["test"]]
    d = r["final_edge"] - b["final_edge"]
    if abs(d) > 1e-9:
        print(r["test"], round(d, 2), "base", round(b["final_edge"], 2), "cool", round(r["final_edge"], 2))
