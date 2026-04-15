import json

with open('logs/walkforward_2025_12month_comparison.json') as f:
    data = json.load(f)
feb = [w for w in data['windows'] if w['tag'] == '2025_02'][0]
log_path = feb.get('log_path')
print('Feb log path:', log_path)

# Load the feb log
with open(log_path) as f:
    feb_log = json.load(f)

print('\nBacktest summary keys:', list(feb_log.get('backtest_summary', {}).keys()))
bs = feb_log.get('backtest_summary', {})
print('\n=== BACKTEST SUMMARY ===')
print(f'Final Value: {bs.get("final_value", 0):.2f}')
print(f'Sharpe: {bs.get("sharpe", 0):.3f}')
print(f'Max Drawdown: {bs.get("max_drawdown_pct", 0):.2f} percent')
print(f'Annualized Return: {bs.get("annualized_return_pct", 0):.2f} percent')
print(f'Trade Days: {bs.get("trade_days", 0)}')
print(f'Trades Executed: {bs.get("trades_executed", 0)}')
print(f'Nifty Final: {bs.get("nifty_final", 0):.2f}')
print(f'Random Final: {bs.get("random_final", 0):.2f}')

print('\n=== TRADE QUALITY ===')
return_qual = feb_log.get('return_quality', {})
alpha_info = return_qual.get('alpha_vs_nifty_per_trade', {})
print(f'Trades: {alpha_info.get("trade_count", 0)}')
print(f'Avg Alpha: {alpha_info.get("average_alpha", 0):.5f}')
print(f'Positive Alpha Fraction: {alpha_info.get("positive_alpha_fraction", 0):.3f}')
print(f'Avg Win: {return_qual.get("average_win", 0):.5f}')
print(f'Avg Loss: {return_qual.get("average_loss", 0):.5f}')
print(f'Profit Factor: {return_qual.get("profit_factor", 0):.3f}')

print('\n=== SIGNAL QUALITY ===')
sig_qual = feb_log.get('signal_quality', {})
conf_spread = sig_qual.get('confidence_spread', {})
ic_info = sig_qual.get('information_coefficient', {})
rank_stab = sig_qual.get('rank_stability', {})
print(f'Mean Rank1-Rank10 Spread: {conf_spread.get("mean_rank1_rank10_spread", 0):.5f}')
print(f'Daily Count (spread): {conf_spread.get("daily_count", 0)}')
print(f'IC (holding period): {ic_info.get("mean_daily_ic", 0):.5f}')
print(f'IC Std Dev: {ic_info.get("full_horizon", {}).get("std_daily_ic", 0):.5f}')
print(f'Top1 Changes Per Week: {rank_stab.get("top1_changes_per_week", 0):.2f}')
print(f'Top1 Persistence: {rank_stab.get("top1_persistence_fraction", 0):.3f}')

print('\n=== REGIME PERFORMANCE ===')
regime_metrics = feb_log.get('regime_metrics', {})
for regime_name, stats in sorted(regime_metrics.items()):
    trades = stats.get('trades', 0)
    print(f'{regime_name}: trades={trades}, hit_rate={stats.get("hit_rate", 0):.3f}, avg_return={stats.get("avg_net_return", 0):.5f}, avg_alpha={stats.get("avg_alpha", 0):.5f}')

print('\n=== MODEL HEALTH ===')
model_health = feb_log.get('model_health', {})
recent_acc = model_health.get('recent_accuracy_check', {})
print(f'Validation Accuracy: {recent_acc.get("validation_accuracy", 0):.3f}')
print(f'Recent 20-day Accuracy: {recent_acc.get("recent_20day_accuracy", 0):.3f}')
print(f'Accuracy Drift: {recent_acc.get("accuracy_drift_vs_validation", 0):.3f}')

print('\n=== SIGNAL BUCKET ===')
bucket_metrics = feb_log.get('signal_bucket_metrics', {})
for bucket_name in ['HIGH', 'MEDIUM', 'LOW']:
    if bucket_name in bucket_metrics:
        b = bucket_metrics[bucket_name]
        print(f'{bucket_name}: trades={b.get("trades", 0)}, hit_rate={b.get("hit_rate", 0):.3f}, avg_return={b.get("avg_net_return", 0):.5f}, avg_alpha={b.get("avg_alpha", 0):.5f}')
    else:
        print(f'{bucket_name}: no trades')

print('\n=== CONCENTRATION RISK ===')
conc = feb_log.get('portfolio_metrics', {}).get('concentration_risk', {})
print(f'Max Signal Day Fraction: {conc.get("max_signal_day_fraction", 0):.3f}')
print(f'By Ticker:')
for ticker, info in list(conc.get('by_ticker', {}).items())[:5]:
    print(f'  {ticker}: signal_days={info.get("signal_days", 0)}, fraction={info.get("signal_day_fraction", 0):.3f}')

# Also check the trade records themselves
print('\n=== SAMPLE TRADES ===')
trade_records = feb_log.get('backtest_summary', {}).get('trade_records', []) or feb_log.get('trade_records', [])
print(f'Total trade records: {len(trade_records)}')
if trade_records:
    for i, trade in enumerate(trade_records[:3]):
        print(f'\nTrade {i+1}:')
        print(f'  Ticker: {trade.get("ticker")}')
        print(f'  Date: {trade.get("signal_date")}')
        print(f'  Confidence: {trade.get("confidence", 0):.4f}')
        print(f'  Return: {trade.get("net_return", 0):.5f}')
        print(f'  Alpha: {trade.get("alpha", 0):.5f}')
        print(f'  Win: {trade.get("win", False)}')
        print(f'  Regime: {trade.get("regime_label")}')
        print(f'  Signal Bucket: {trade.get("signal_bucket")}')
