import json, os, re, subprocess
from pathlib import Path

WINDOWS = [
    ('2025_07', '2025-07-01', '2025-07-31'),
    ('2025_08', '2025-08-01', '2025-08-31'),
    ('2025_09', '2025-09-01', '2025-09-30'),
    ('2025_10', '2025-10-01', '2025-10-31'),
    ('2025_11', '2025-11-01', '2025-11-30'),
    ('2025_12', '2025-12-01', '2025-12-31'),
    ('2026_01', '2026-01-01', '2026-01-31'),
    ('2026_02', '2026-02-01', '2026-02-28'),
    ('2026_03', '2026-03-01', '2026-03-31'),
    ('2026_04', '2026-04-01', '2026-04-09'),
]

CURRENT_ROOT = Path('.')
BACKUP_ROOT  = Path(r'C:\Users\fioli\OneDrive\Desktop\project\tradingAIAgent\backup\QuantPulse')
VENV_PY      = CURRENT_ROOT / 'venv' / 'Scripts' / 'python.exe'
LOGS         = CURRENT_ROOT / 'logs'
LOGS.mkdir(exist_ok=True)


def parse(text):
    m = {}
    # Only grab values from the REAL MODEL section to avoid picking up random/inverted numbers
    real_section = re.search(r'REAL MODEL(.*?)(?:RANDOM BASELINE|$)', text, re.DOTALL)
    src = real_section.group(1) if real_section else text
    for label, key in [
        ('Final Value:', 'final'),
        ('Sharpe:', 'sharpe'),
        ('Max Drawdown:', 'dd'),
        ('Annualized Return:', 'ann'),
        ('Trade Days:', 'trade_days'),
        ('Trades Executed:', 'trades'),
    ]:
        pat = re.escape(label) + r'\s*([\-\d\.]+)'
        hit = re.search(pat, src)
        if hit:
            try:
                m[key] = float(hit.group(1))
            except ValueError:
                pass
    # Nifty - always at bottom
    hit = re.search(r'NIFTY BUY & HOLD\s*\nFinal Value:\s*([\d\.]+)', text)
    if hit:
        try:
            m['nifty'] = float(hit.group(1))
        except ValueError:
            pass
    return m


results = []
for tag, start, end in WINDOWS:
    print(f'\n=== {tag}: {start} -> {end} ===', flush=True)

    base_env = os.environ.copy()
    base_env['QUANT_CARRY_FORWARD_OPEN'] = '1'
    base_env['PYTHONUTF8'] = '1'
    base_env['PYTHONIOENCODING'] = 'utf-8'

    # ---- CURRENT ----
    cmd_cur = [str(VENV_PY), '-u', 'train.py', '--backtest-only', '--start', start, '--end', end]
    proc = subprocess.run(cmd_cur, cwd=CURRENT_ROOT, env=base_env, capture_output=True, text=True, timeout=600)
    cur_out = proc.stdout + proc.stderr
    (LOGS / ('cmp_current_' + tag + '.log')).write_text(cur_out, encoding='utf-8')
    cur = parse(cur_out)
    print('  CURRENT  final={:>9}  sharpe={:>6}  dd={}%  trades={}'.format(
        cur.get('final', '?'), cur.get('sharpe', '?'), cur.get('dd', '?'), cur.get('trades', '?')), flush=True)

    # ---- BACKUP ----
    cmd_bak = [str(VENV_PY), '-u', 'train.py', '--backtest-only', '--start', start, '--end', end]
    proc = subprocess.run(cmd_bak, cwd=BACKUP_ROOT, env=base_env, capture_output=True, text=True, timeout=600)
    bak_out = proc.stdout + proc.stderr
    (LOGS / ('cmp_backup_' + tag + '.log')).write_text(bak_out, encoding='utf-8')
    bak = parse(bak_out)
    print('  BACKUP   final={:>9}  sharpe={:>6}  dd={}%  trades={}'.format(
        bak.get('final', '?'), bak.get('sharpe', '?'), bak.get('dd', '?'), bak.get('trades', '?')), flush=True)

    edge = None
    if 'final' in cur and 'final' in bak and bak['final']:
        edge = round((cur['final'] - bak['final']) / bak['final'] * 100, 2)
    print('  EDGE (current vs backup): {}%'.format(edge), flush=True)
    results.append({'tag': tag, 'start': start, 'end': end, 'current': cur, 'backup': bak, 'edge_pct': edge})

# ---- Summary ----
print('\n\n========================================')
print('SIDE-BY-SIDE SUMMARY')
print('========================================')
print('  {:<10} {:>10} {:>10} {:>8} {:>11} {:>11}'.format(
    'Window', 'Cur Final', 'Bak Final', 'Edge%', 'Cur Sharpe', 'Bak Sharpe'))
print('  ' + '-' * 65)
cur_wins = bak_wins = ties = 0
for r in results:
    c, b = r['current'], r['backup']
    cf = c.get('final', '?')
    bf = b.get('final', '?')
    cs = c.get('sharpe', '?')
    bs = b.get('sharpe', '?')
    e  = r['edge_pct']
    if isinstance(e, float):
        mark = '<<CURRENT' if e > 0 else ('>>BACKUP' if e < 0 else '=TIE')
        if e > 0:   cur_wins += 1
        elif e < 0: bak_wins += 1
        else:       ties += 1
    else:
        mark = '?'
    print('  {:<10} {:>10} {:>10} {:>7}% {:>11} {:>11}  {}'.format(
        r['tag'], str(cf), str(bf), str(e), str(cs), str(bs), mark))

print('')
print(f'  Current WINS: {cur_wins}  |  Backup WINS: {bak_wins}  |  Ties: {ties}')

Path('logs/backup_vs_current_comparison.json').write_text(json.dumps({'results': results}, indent=2))
print('\nArtifact saved: logs/backup_vs_current_comparison.json')
