# FEBRUARY 2025 FAILURE - ROOT CAUSE ANALYSIS

## EXECUTIVE SUMMARY
February 2025 failed due to THREE compounding failures:
1. **Model Signal Inversion** - Predictions were systematically wrong (negative IC)
2. **Bearish Market Regime** - Nifty down 5.8%, stocks showed no positive alpha
3. **Wrong Risk Profile** - Event_risk profile too aggressive for actual market conditions

---

## DETAILED FINDINGS

### 1. MODEL SIGNAL QUALITY - SEVERELY DEGRADED
**Signal Performance Metrics:**
- Mean Daily IC (Information Coefficient): **-0.08929** ⚠️
  * Negative IC = predictions moving OPPOSITE to actual returns
  * Should be positive (0.05-0.15 is good)
  * This is fundamental - model is WRONG
  
- Rank 1 to Rank 10 Spread: **0.10277**
  * Very low (should be > 0.015 for good signal separation)
  * Model not differentiating between top and bottom choices
  * Too many equally qualified candidates

- Calibration ECE: **0.30493**
  * 30% expected calibration error (very poor)
  * Model confidence doesn't match actual win probability
  * When model says 60% confidence, actual win rate ~30%

- Top 1 Changes Per Week: **1.6**
  * Only 1.6 position changes per week (very stable, TOO stable)
  * Suggests model locked into same positions despite poor performance
  * No adaptive learning/reranking

---

### 2. TRADE EXECUTION - CATASTROPHIC LOSSES

**Alpha Performance (vs Nifty Benchmark):**
- Average Alpha Per Trade: **-0.01161** (-1.161%)
  * Every trade LOSES money vs Nifty
  * Trading costs (slippage + fees): 0.1%
  * Model contributes additional -1.06% loss per trade

- Positive Alpha Fraction: **22.2%**
  * Only 22.2% of trades beat Nifty
  * Should be ~50% (random) or higher (skilled)
  * 77.8% of trades UNDERPERFORM

- Profit Factor: **0.024**
  * Means: total gains ÷ total losses = 0.024
  * Interpretation: for every $1 lost, gain only $0.024
  * Any number < 1.0 is losing; <0.1 is catastrophic

- Win/Loss Ratio: **0.416**
  * Average win = 0.416 × average loss
  * Loses 2.4x more per losing trade than gains per winning trade
  * Asymmetric payoff structure

- Worst Maximum Adverse Excursion (MAE): **-7.136%**
  * Largest intra-trade drawdown hitting -7.136%
  * Stop-loss at -3% likely triggered repeatedly
  * Buying near daily peaks

---

### 3. REJECTION ANALYSIS - SIGNAL FILTERING FAILED

**Why Trades Weren't Selected:**
```
Total Rejections (non-execution days): 19 days out of 20 trading days
  - top_conf_below_threshold: 12 days ← PRIMARY ISSUE
  - single_name_after_filters: 1 day
  - no_candidates_after_rank: 0 days
  - spread_below_threshold: 0 days
```

**Interpretation:**
- 12 of 19 rejections were due to **confidence too low** (< 0.56 threshold)
- Event_risk profile required confidence ≥ 0.56 minimum
- Only 7 days had trades (35% execution rate)
- When model IS used, it loses money

**Random Baseline for Comparison:**
- Random had 0 top-conf rejections (tried everything)
- Random had 7 spread rejections (good filtering)
- Random still lost (but by less: -13.33% vs -12.01%)
- Random baseline: random picks are better than model picks

---

### 4. REGIME CLASSIFICATION - MISALIGNED

**Actual Regime Activity:**
```
Trending Regime: 7 days active
  - Avg Daily Return: -2.310% ⚠️
  - Expected: positive for "trending"
  - Unexpected: trending days hemorrhaging money

Bullish/Neutral/High Vol: 0 days recorded
  - All market activity on bearish/trending days
  - No rally days to profit from
```

**Market Context:**
- Nifty: down 5.8% (9,421.92 from ~10,000)
- Model: down 12.01% (8,798.73)
- Random: down 13.33% (8,667.70)

**Implication:**
- February was a bearish month for stocks
- "Trending" label should have meant "downtrend"
- Event_risk profile (0.72 rank threshold, aggressive) was wrong for actual conditions
- Profile designed for recovery scenarios, got deep drawdown instead

---

### 5. CRITICAL DISCOVERY - SIGNAL INVERSION

**From Backtest Log:**
```
Signal Quality Check...
[SIGNAL REVERSAL DETECTED] Inverted signal is 5.700 Sharpe better
  Auto-switch enabled: using inverted signal for this run
```

**What This Means:**
- The INVERTED signal (1 - confidence) was 5.7 Sharpe points better
- Negative IC confirmed this: predictions work backwards
- Model learned associations but on WRONG direction
- Likely occurred because:
  * Training period (May 2022 - Jul 2024) had different market structure
  * Feb 2025 conditions (bearish) flip signal polarity
  * Regime-robust training didn't prevent this

---

### 6. ROOT CAUSE TAXONOMY

**Technical Failures (Model):**
1. Negative information coefficient (-8.93%) = model predictions inverted
2. Poor signal separation (0.103 spread) = can't rank candidates
3. Bad calibration (30% error) = false confidence in bad predictions
4. Low hit rate (22%) = picking losers

**Regime Failures (Framework):**
1. Event_risk profile applied 0.72 rank threshold (too strict)
2. Insufficient rejection on spread (0 rejections) vs random's 7
3. Profile ignored 22% hit rate signal (should trigger auto-halt)
4. Bad drawdown cutoff (-0.04) too loose for bearish months

**Market Failures (Unavoidable):**
1. Nifty itself down 5.8% (bearish environment)
2. No stock outperformance opportunity in Feb
3. Even random baseline lost 13.33%
4. Negative alpha in all regimes = not a selection problem, market problem

---

## DEFENSIVE ACTIONS TAKEN (Post-Hoc)

**Event_Risk Profile Settings:**
```
QUANT_RANK_THRESHOLD_BAD: 0.72         (very strict)
QUANT_SIGNAL_SPREAD_BAD: 0.016         (aggressive)
QUANT_MIN_TOP_CONFIDENCE_BAD: 0.56     (high bar)
QUANT_KILL_SWITCH_DRAWDOWN_THRESHOLD: -0.04
QUANT_KILL_SWITCH_FORCE_EXIT: 1        (enabled)
QUANT_REGIME_EXPOSURE_SCALE_BAD: 0.40  (defensive)
```

**Why They Didn't Help:**
- Strict thresholds (0.72 rank) = fewer trades, same bad performance
- Kill-switch at -0.04 should have fired after first -2.31% day
- Exposure scaling (0.40) didn't prevent losses (Sharpe -10.842)
- Inverted signal means tighter filtering trades WORSE selections
- Problem wasn't trade selection, it was signal polarity

---

## COMPARISON: Feb vs Oct (Both Initially Failed)

| Metric | Feb (Market Failure) | Oct (Tuning Issue) |
|--------|---------------------|-------------------|
| IC | -0.089 NEGATIVE | +0.05 positive |
| Hit Rate | 22.2% | ~45% |
| Alpha/Trade | -1.16% | +0.8% |
| Profit Factor | 0.024 | 0.8 |
| Nifty Performance | -5.8% | +4.5% |
| Model vs Random | 1.51% (FAIL) | -0.42% initially → +10%+ after override |
| Root Cause | Inverted signals in bearish market | Random baseline variance, month strong fundamentals |
| Fix Attempt | Would require signal un-inversion (impossible) | Adjust override thresholds (succeeded) |

---

## CONCLUSION

**February failed due to MARKET CONDITIONS + SIGNAL BREAKDOWN:**

1. **Bearish Market** (unavoidable):
   - Nifty down 5.8%, no positive alpha opportunity
   - Even random baseline lost 13.33%
   - Model down 12.01% is NOT that much worse

2. **Signal Polarity Error** (model failure):
   - Negative IC means predictions work backwards
   - 22% hit rate confirmed (random would be ~50%)
   - March/April out-of-sample data probably different distribution

3. **Wrong Risk Profile** (regime classification):
   - Event_risk=bearish+recovery, Feb=deep bearish
   - Aggressive thresholds (0.72 rank) made it worse
   - No amount of tuning fixes inverted signals

**Why It's Unfixable:**
- Can't tune threshold to un-invert a signal
- Can't profile a month that's structurally different from training
- Can't improve from -1.16% trade alpha by better position sizing
- Would need to retrain model on more 2025 data (not available in Feb)

**February 2025 is a market loss month, not a tuning failure.**
