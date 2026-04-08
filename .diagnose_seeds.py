#!/usr/bin/env python
"""
Diagnostic script for seeds 42 and 777 underperformance.
Analyzes prediction distribution, trade characteristics, and model behavior.
"""

import subprocess
import os
import re
import json
import numpy as np
import pandas as pd
import torch

output_template = """
========== SEED {seed} DIAGNOSTIC ==========

RUNNING TRAINING AND INFERENCE...
"""

problem_seeds = [42, 777]

for seed in problem_seeds:
    print(f"\n{'='*60}")
    print(f"DIAGNOSING SEED {seed}")
    print(f"{'='*60}")
    
    # Run training
    env = os.environ.copy()
    env["QUANT_SEED"] = str(seed)
    
    result = subprocess.run(
        [".\\venv\\Scripts\\python.exe", "-u", "train.py"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env
    )
    
    stdout = result.stdout
    stderr = result.stderr
    
    # Parse key diagnostics from output
    print("\n--- MODEL CONFIDENCE DISTRIBUTION ---")
    
    # Find predictions
    pred_match = re.search(r"First 20 predictions:\n\[(.*?)\]", stdout, re.DOTALL)
    if pred_match:
        pred_str = pred_match.group(1)
        preds = [float(x.strip()) for x in pred_str.split() if x.strip()]
        print(f"Sample predictions: {preds[:10]}")
        print(f"Mean confidence: {np.mean(preds):.4f}")
        print(f"Min confidence: {np.min(preds):.4f}")
        print(f"Max confidence: {np.max(preds):.4f}")
        print(f"Std dev: {np.std(preds):.4f}")
    
    print("\n--- BACKTEST RESULTS ---")
    
    # Parse backtest results
    real_match = re.search(
        r"REAL MODEL\nFinal Value: ([\d.]+)\nSharpe: ([\d.-]+).*?Annualized Return: ([\d.]+) %.*?Trade Days: (\d+)\nTrades Executed: (\d+)",
        stdout, re.DOTALL
    )
    
    rand_match = re.search(
        r"RANDOM BASELINE\nFinal Value: ([\d.]+)\nSharpe: ([\d.-]+).*?Annualized Return: ([\d.]+) %.*?Trade Days: (\d+)\nTrades Executed: (\d+)",
        stdout, re.DOTALL
    )
    
    inv_match = re.search(
        r"INVERTED SIGNAL.*?\nFinal Value: ([\d.]+)\nSharpe: ([\d.-]+).*?Trade Days: (\d+)\nTrades Executed: (\d+)",
        stdout, re.DOTALL
    )
    
    if real_match and rand_match:
        real_final, real_sharpe, real_ann, real_trade_days, real_trades = real_match.groups()
        rand_final, rand_sharpe, rand_ann, rand_trade_days, rand_trades = rand_match.groups()
        
        real_final, real_ann, real_trade_days, real_trades = float(real_final), float(real_ann), int(real_trade_days), int(real_trades)
        rand_final, rand_ann, rand_trade_days, rand_trades = float(rand_final), float(rand_ann), int(rand_trade_days), int(rand_trades)
        real_sharpe, rand_sharpe = float(real_sharpe), float(rand_sharpe)
        
        print(f"\nREAL MODEL:")
        print(f"  Final: ${real_final:.2f} | Sharpe: {real_sharpe:.3f} | Ann Return: {real_ann:.2f}%")
        print(f"  Trade Days: {real_trade_days} | Trades: {real_trades}")
        if real_trades > 0:
            print(f"  Avg trades per trade day: {real_trades / max(real_trade_days, 1):.2f}")
        
        print(f"\nRANDOM BASELINE:")
        print(f"  Final: ${rand_final:.2f} | Sharpe: {rand_sharpe:.3f} | Ann Return: {rand_ann:.2f}%")
        print(f"  Trade Days: {rand_trade_days} | Trades: {rand_trades}")
        if rand_trades > 0:
            print(f"  Avg trades per trade day: {rand_trades / max(rand_trade_days, 1):.2f}")
        
        if inv_match:
            inv_final, inv_sharpe, inv_trade_days, inv_trades = inv_match.groups()
            inv_final, inv_trade_days, inv_trades = float(inv_final), int(inv_trade_days), int(inv_trades)
            inv_sharpe = float(inv_sharpe)
            
            print(f"\nINVERTED SIGNAL (diagnostic):")
            print(f"  Final: ${inv_final:.2f} | Sharpe: {inv_sharpe:.3f}")
            print(f"  Trade Days: {inv_trade_days} | Trades: {inv_trades}")
        
        print(f"\n--- ANALYSIS ---")
        print(f"Real vs Random Delta: ${real_final - rand_final:.2f}")
        print(f"Sharpe Delta: {real_sharpe - rand_sharpe:.3f}")
        
        # Diagnose
        if real_final < rand_final:
            print(f"[UNDERPERFORMING] vs random by ${rand_final - real_final:.2f}")
            
            # Check if it's a signal issue or execution issue
            if inv_final < rand_final:
                print(f"   -> Signals appear VALID (inverted is worse than random)")
            elif inv_final > rand_final:
                print(f"   -> POTENTIAL SIGNAL REVERSAL (inverted beats random!)")
                print(f"   -> Consider signal inversion or threshold adjustment")
            
            # Check trade count
            if real_trades < rand_trades * 0.5:
                print(f"   -> Too few trades vs random ({real_trades} vs {rand_trades})")
                print(f"   -> May be over-filtering candidates (min_rank_threshold too high)")
            elif real_trades > rand_trades * 1.5:
                print(f"   -> Too many trades vs random ({real_trades} vs {rand_trades})")
                print(f"   -> May be taking low-confidence trades")
            
            # Check average return per trade
            real_ret_per_trade = ((real_final / 10000 - 1) * 100) / max(real_trade_days, 1) if real_trade_days > 0 else 0
            rand_ret_per_trade = ((rand_final / 10000 - 1) * 100) / max(rand_trade_days, 1) if rand_trade_days > 0 else 0
            print(f"   -> Real return per trade day: {real_ret_per_trade:.3f}%")
            print(f"   -> Random return per trade day: {rand_ret_per_trade:.3f}%")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("""
1. Check if inverted signal outperforms → may indicate signal is reversed
2. Compare model prediction confidence distribution vs other seeds
3. Check if min_rank_threshold needs adjustment
4. Consider reducing top_k or adjusting min_signal_spread
5. Profile which stocks are being picked on bad days
""")
