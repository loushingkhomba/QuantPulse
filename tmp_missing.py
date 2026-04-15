import os, subprocess
from pathlib import Path

BACKUP_ROOT = Path(r'C:\Users\fioli\OneDrive\Desktop\project\tradingAIAgent\backup\QuantPulse')
VENV_PY     = Path('venv/Scripts/python.exe')
LOGS        = Path('logs')

env = os.environ.copy()
env['PYTHONUTF8'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
env['QUANT_CARRY_FORWARD_OPEN'] = '1'

for tag, start, end in [('2025_02','2025-02-01','2025-02-28'), ('2025_04','2025-04-01','2025-04-30')]:
    print(f'\n=== backup {tag} ===', flush=True)
    proc = subprocess.run(
        [str(VENV_PY), '-u', 'train.py', '--backtest-only', '--start', start, '--end', end],
        cwd=BACKUP_ROOT, env=env, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    (LOGS / f'cmp_backup_{tag}.log').write_text(out, encoding='utf-8')
    # extract key lines
    for line in out.splitlines():
        l = line.strip()
        if any(k in l for k in ['Final Value:','Sharpe:','RANDOM BASELINE','NIFTY BUY','REAL MODEL']):
            print(' ', l)
