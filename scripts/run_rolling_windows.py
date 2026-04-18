"""Run N rolling backtest windows and print a summary table."""
import subprocess, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

GATES = {
    "QUANT_FEATURE_SIGNAL_ENABLED": "0",
    "QUANT_MIN_TOP_CONFIDENCE": "0.480",
    "QUANT_MIN_TOP_CONFIDENCE_BAD": "0.485",
    "QUANT_MIN_TOP_CONFIDENCE_TRENDING": "0.495",
    "QUANT_TOP_K": "3",
    "QUANT_MAX_NEW_POSITIONS_PER_DAY": "3",
    "QUANT_TURNOVER_COOLDOWN_DAYS": "2",
    "QUANT_SIGNAL_INVERSION_MODE": "none",
    "QUANT_AUTO_INVERT_SIGNAL": "0",
}

WINDOWS = [
    ("2026-03-24", "2026-04-06"),
    ("2026-03-23", "2026-04-05"),
    ("2026-03-22", "2026-04-04"),
    ("2026-03-21", "2026-04-03"),
    ("2026-03-20", "2026-04-02"),
    ("2026-03-19", "2026-04-01"),
    ("2026-03-18", "2026-03-31"),
    ("2026-03-17", "2026-03-30"),
    ("2026-03-16", "2026-03-28"),
    ("2026-03-15", "2026-03-27"),
]

import os
env = os.environ.copy()
env.update(GATES)

def parse(text):
    def g(pattern, default="N/A"):
        m = re.search(pattern, text)
        return m.group(1) if m else default
    # All "Final Value:" lines in order
    finals = re.findall(r"Final Value[:\s]+([\d.]+)", text)
    real_final = finals[0] if len(finals) > 0 else "N/A"
    rand_final = finals[1] if len(finals) > 1 else "N/A"
    # All "Sharpe:" lines — first = real, second = random
    sharpes = re.findall(r"Sharpe[:\s]+([\d.\-]+)", text)
    real_sharpe = sharpes[0] if len(sharpes) > 0 else "N/A"
    rand_sharpe = sharpes[1] if len(sharpes) > 1 else "N/A"
    trades = g(r"Trades Executed[:\s]+(\d+)")
    wf = "PASS" if re.search(r"pass=\s*True", text) else "FAIL"
    rob = "PASS" if re.search(r"overall_pass.*True", text) else "FAIL"
    return real_final, rand_final, real_sharpe, rand_sharpe, trades, wf, rob

rows = []
import csv
out_csv = ROOT / "logs" / "standard_metrics" / "rolling_10window_results.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["start","end","real_final","rand_final","real_sharpe","rand_sharpe","trades","wf","robustness"])
    for start, end in WINDOWS:
        print(f"Running {start} -> {end}...", flush=True)
        log_file = ROOT / "logs" / "standard_metrics" / f"window_{start}_{end}.log"
        with open(log_file, 'w') as lf:
            r = subprocess.run(
                [PY, "train.py", "--backtest-only", "--start", start, "--end", end],
                cwd=str(ROOT), env=env, stdout=lf, stderr=lf,
                text=True, timeout=120
            )
        out = log_file.read_text(errors='replace')
        real_f, rand_f, real_s, rand_s, trades, wf, rob = parse(out)
        rows.append((start, end, real_f, rand_f, real_s, rand_s, trades, wf, rob))
        writer.writerow([start, end, real_f, rand_f, real_s, rand_s, trades, wf, rob])
        f.flush()
        print(f"  Real={real_f} Sh={real_s} | Rand={rand_f} Sh={rand_s} | Trades={trades} WF={wf} Rob={rob}", flush=True)

print("\n{:<12} {:<12} {:>10} {:>10} {:>8} {:>8} {:>6} {:>4} {:>4}".format(
    "Start","End","RealFinal","RandFinal","RealSh","RandSh","Trades","WF","Rob"))
print("-"*90)
for r in rows:
    print("{:<12} {:<12} {:>10} {:>10} {:>8} {:>8} {:>6} {:>4} {:>4}".format(*r))

beats = sum(1 for r in rows if r[2] != "N/A" and r[3] != "N/A" and float(r[2]) > float(r[3]))
print(f"\nBeats random: {beats}/{len(rows)}")
