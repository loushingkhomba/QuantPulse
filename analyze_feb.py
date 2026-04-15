import json
import numpy as np
import pandas as pd

# Load the full 12-month comparison
with open('logs/walkforward_2025_12month_comparison.json') as f:
    data = json.load(f)

# Find Feb window
feb = [w for w in data['windows'] if w['tag'] == '2025_02'][0]

print('='*70)
print('FEBRUARY 2025 DEEP DIVE ANALYSIS')
print('='*70)

print('\n1. PASS/FAIL GATES:')
print(f'   Result: {feb.get("pass_or_fail")}')
print(f'   First Failed Gate: {feb.get("first_failed_gate")}')
print(f'   All Pass Flags: {feb.get("pass_flags", {})}')

print('\n2. PERFORMANCE METRICS:')
print(f'   vs Random: {feb.get("model_vs_random_pct", 0):.2f}% (need 2.0%)')
print(f'   vs Nifty: {feb.get("model_vs_nifty_pct", 0):.2f}% (need 0.5%)')
print(f'   Sharpe: {feb.get("model_sharpe", 0):.3f} (need 1.0)')
print(f'   Max DD: {feb.get("model_dd_pct", 0):.2f}% (need <= 15%)')
print(f'   Trade Count: {feb.get("model_trade_count", 0)} (need <= 50)')
print(f'   Nifty Final: {feb.get("nifty_final", 0):.2f}')
print(f'   Model Final: {feb.get("model_final", 0):.2f}')
print(f'   Random Final: {feb.get("random_final", 0):.2f}')

print('\n3. PORTFOLIO STATS:')
print(f'   Trade Days: {feb.get("trade_days", 0)}')
print(f'   Initial Capital: {feb.get("initial_capital", 0):.2f}')
print(f'   Annualized Return: {feb.get("annualized_return_pct", 0):.2f}%')

print('\n4. SIGNAL QUALITY METRICS:')
sig_qual = feb.get('signal_quality', {})
conf_spread = sig_qual.get('confidence_spread', {})
rank_stab = sig_qual.get('rank_stability', {})
ic_info = sig_qual.get('information_coefficient', {})

print(f'   Mean Rank1-Rank10 Spread: {conf_spread.get("mean_rank1_rank10_spread", 0):.5f}')
print(f'   Median Spread: {conf_spread.get("median_rank1_rank10_spread", 0):.5f}')
print(f'   Low Quality Fraction: {conf_spread.get("warning_low_quality_fraction", 0):.3f}')
print(f'   Top1 Changes Per Week: {rank_stab.get("top1_changes_per_week", 0):.2f}')
print(f'   Top1 Persistence: {rank_stab.get("top1_persistence_fraction", 0):.3f}')
print(f'   Mean Daily IC (holding days): {ic_info.get("mean_daily_ic", 0):.5f}')

print('\n5. TRADE EXECUTION QUALITY:')
ret_qual = feb.get('return_quality', {})
alpha_trade = ret_qual.get('alpha_vs_nifty_per_trade', {})
print(f'   Trade Count: {alpha_trade.get("trade_count", 0)}')
print(f'   Avg Alpha Per Trade: {alpha_trade.get("average_alpha", 0):.5f}')
print(f'   Positive Alpha Fraction: {alpha_trade.get("positive_alpha_fraction", 0):.3f}')
print(f'   Avg Win: {ret_qual.get("average_win", 0):.5f}')
print(f'   Avg Loss: {ret_qual.get("average_loss", 0):.5f}')
print(f'   Profit Factor: {ret_qual.get("profit_factor", 0):.3f}')

print('\n6. REJECTION COUNTS (Signal Selection Process):')
reject = feb.get('rejection_counts', {})
total_rejections = sum(reject.values())
print(f'   Total Rejections: {total_rejections}')
for key, val in sorted(reject.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f'   - {key}: {val}')

print('\n7. REGIME & MARKET CONTEXT:')
regime = feb.get('regime_metrics', {})
print(f'   Days in Bearish regime: {regime.get("BEARISH", {}).get("days", 0)}')
print(f'   Days in Neutral regime: {regime.get("NEUTRAL", {}).get("days", 0)}')
print(f'   Days in Bullish regime: {regime.get("BULLISH", {}).get("days", 0)}')
for regime_name in ['BEARISH', 'NEUTRAL', 'BULLISH']:
    if regime_name in regime and regime[regime_name]:
        print(f'   {regime_name} avg return: {regime[regime_name].get("avg_net_return", 0):.5f}')

print('\n8. MODEL HEALTH:')
model_health = feb.get('model_health', {})
recent_acc = model_health.get('recent_accuracy_check', {})
print(f'   Validation Accuracy: {recent_acc.get("validation_accuracy", 0):.3f}')
print(f'   Recent 20-day Accuracy: {recent_acc.get("recent_20day_accuracy", 0):.3f}')
print(f'   Accuracy Drift: {recent_acc.get("accuracy_drift_vs_validation", 0):.3f}')

print('\n9. SIGNAL BUCKET ANALYSIS:')
bucket = feb.get('signal_bucket_metrics', {})
for bucket_name in ['HIGH', 'MEDIUM', 'LOW']:
    if bucket_name in bucket:
        b = bucket[bucket_name]
        print(f'   {bucket_name}: trades={b.get("trades", 0)}, hit_rate={b.get("hit_rate", 0):.3f}, avg_return={b.get("avg_net_return", 0):.5f}')

print('\n' + '='*70)
print('SUMMARY OF ROOT CAUSES:')
print('='*70)
