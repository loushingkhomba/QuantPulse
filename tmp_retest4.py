"""Targeted retest of the 4 previously-failing months.
   Reads profile from run_10_window_retests.apply_month_type_profile,
   runs train.py --backtest-only, and reports gate results.
"""
import os, subprocess, sys
from pathlib import Path
from run_10_window_retests import apply_month_type_profile, parse_metrics, THRESHOLDS

ROOT     = Path(__file__).resolve().parent
LOGS     = ROOT / "logs"
VENV_PY  = ROOT / "venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = Path(sys.executable)

TARGETS = [
    ("2025_02", "2025-02-01", "2025-02-28", "event_risk"),
    ("2025_07", "2025-07-01", "2025-07-31", "monsoon_chop"),
    ("2025_11", "2025-11-01", "2025-11-30", "festival_trend"),
    ("2025_12", "2025-12-01", "2025-12-31", "year_end"),
]

print("=" * 70)
print("TARGETED RETEST — 4 FAILING MONTHS")
print(f"beats_random_min_edge_pct : {THRESHOLDS['beats_random_min_edge_pct']}%")
print(f"beats_nifty_min_edge_pct  : {THRESHOLDS['beats_nifty_min_edge_pct']}%")
print(f"min_sharpe                : {THRESHOLDS['min_sharpe']}")
print(f"max_drawdown_abs_pct      : {THRESHOLDS['max_drawdown_abs_pct']}%")
print("=" * 70)

results = []
for tag, start, end, profile_label in TARGETS:
    log_path = LOGS / f"rectify_v3_{tag}.log"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["QUANT_SIGNAL_MODE"] = "model"
    env["QUANT_CARRY_FORWARD_OPEN"] = "1"
    env["QUANT_TRANSACTION_COST"] = "0.0005"
    env["QUANT_TEST_START"] = start
    env["QUANT_TEST_END"] = end
    env["QUANT_SPLIT_DATE"] = start

    profile_args, profile_metadata = apply_month_type_profile(start)
    env.update(profile_metadata.get("runtime_env", {}))

    cmd = [str(VENV_PY), "-u", "train.py", "--backtest-only", "--start", start, "--end", end]
    cmd.extend(profile_args)

    print(f"\n[{tag}] profile={profile_label}  start={start}  end={end}", flush=True)
    print(f"  runtime_env applied: {profile_metadata.get('runtime_env', {})}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    combined = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        print(f"  ERROR: run failed — see {log_path.name}")
        continue

    m = parse_metrics(combined)
    pf = m["pass_flags"]
    verdict = m["pass_or_fail"]
    first_fail = m["first_failed_gate"] or "—"

    print(f"  final={m['model_final']}  sharpe={m['model_sharpe']}  trades={m['model_trade_count']}", flush=True)
    print(f"  vs_random={m['model_vs_random_pct']}%  vs_nifty={m['model_vs_nifty_pct']}%  dd={m['model_dd_pct']}%", flush=True)
    print(f"  gates: {pf}", flush=True)
    print(f"  >> {verdict}  (first fail: {first_fail})", flush=True)

    results.append({"tag": tag, "verdict": verdict, "first_fail": first_fail, **m})

print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
passed = [r for r in results if r["verdict"] == "PASS"]
failed = [r for r in results if r["verdict"] == "FAIL"]
fail_summary = ', '.join('{} ({})'.format(r['tag'], r['first_fail']) for r in failed)
print(f"  PASS: {len(passed)}/4  ->  {[r['tag'] for r in passed]}")
print(f"  FAIL: {len(failed)}/4  ->  [{fail_summary}]")
