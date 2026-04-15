# Updated Execution Plan (Honest Performance-First)

## Restore Point
- Name: `pre_rawsignal_walkforward_plan_20260410_01`
- Location: `restore_points/pre_rawsignal_walkforward_plan_20260410_01/`
- Snapshot includes:
  - `train.py`
  - `src/trainer.py`
  - `src/features.py`
  - `src/dataset.py`
  - `README.txt`

## Objective
Prioritize improvements that increase real forward performance and reduce false positives from evaluation artifacts.

## Phase A: Raw-Signal Execution Integrity (Highest Priority)
Goal: Make native model direction the default and keep inversion/orientation diagnostic-only.

### Tasks
- Change default `QUANT_EXECUTION_SIGNAL_DIRECTION` to `normal`.
- Allow inversion only through explicit override.
- Ensure headline "REAL MODEL" backtest always uses raw signal path unless explicitly overridden.
- Keep oriented/inverted outputs as diagnostics, never automatic execution replacement.
- Add explicit run metadata in diagnostics JSON:
  - `headline_execution_mode`
  - `manual_override_used`
  - `manual_override_value`

### Acceptance Criteria
- Default run does not invert or auto-switch signal.
- Any inversion requires explicit env override.
- Diagnostics clearly distinguish execution path vs diagnostic paths.

## Phase B: Strict Walk-Forward Validation (Second Priority)
Goal: Enforce no-leak evaluation and reduce overfitting risk.

### Tasks
- Add walk-forward mode in `train.py` using rolling windows:
  - train window
  - validation window
  - test window
- Freeze hyperparameters globally before first window.
- Disable window-by-window retuning.
- For each window, persist:
  - IC mean, IC-IR
  - rolling IC consistency
  - final value, sharpe, max drawdown
- Aggregate across windows with confidence intervals (bootstrap on windows):
  - mean and 95% CI for sharpe, annualized return, IC mean

### Acceptance Criteria
- Each window strictly uses past data for training and future data for testing.
- Same hyperparameters used in all windows.
- Output includes per-window table + aggregate CI summary.

## Phase C: Governance Gates (Promotion Controls)
Goal: Prevent unstable models from promotion.

### Tasks
- Implement gate evaluation module (in `train.py` first, refactor later):
  - Gate 1: IC drop <= 50% vs previous validation window.
  - Gate 2: Max drawdown jump limited window-to-window.
  - Gate 3: Rolling pass-rate floor over last 12 windows.
- Add final decision output:
  - `promotion_status`: pass/fail
  - `failed_gates`: list

### Acceptance Criteria
- Failed gate automatically marks model as non-promotable.
- Gate outputs are serialized to logs for audit.

## Phase D: Ranking Objective Upgrade (After A-C)
Goal: Improve top-k selection quality with better objective alignment.

### Tasks
- Move from in-batch pairwise ranking to date-grouped ranking objective.
- Keep optional CE auxiliary loss but secondary to ranking.
- Add confidence concentration diagnostics:
  - top-decile concentration
  - score entropy proxy

### Acceptance Criteria
- Improved top-vs-bottom spread consistency across walk-forward windows.
- No governance gate regression.

## Phase E: Feature Layer and Robustness Harness
Goal: Improve signal breadth only after validation/governance is stable.

### Tasks
- Add new cross-sectional/regime-aware features.
- Add stability-based feature selection across windows.
- Build mandatory stress sweeps:
  - costs, slippage, delay, missing data
- Add month-level failure attribution by regime/decile.

### Acceptance Criteria
- Feature changes survive strict walk-forward and all gates.
- Stress sweeps do not collapse edge.

## Execution Order (Do Not Skip)
1. Phase A
2. Phase B
3. Phase C
4. Phase D
5. Phase E

## Immediate Next Implementation Block
- Implement Phase A completely.
- Then run one baseline command and one explicit inversion command to confirm behavior separation.

### Validation Commands
```powershell
# Baseline (must be raw signal execution)
$env:QUANT_EXECUTION_SIGNAL_DIRECTION='normal'; c:/Users/fioli/OneDrive/Desktop/project/tradingAIAgent/ai_trading_agent/venv/Scripts/python.exe train.py --backtest-only

# Diagnostic override (manual inversion path)
$env:QUANT_EXECUTION_SIGNAL_DIRECTION='inverted'; c:/Users/fioli/OneDrive/Desktop/project/tradingAIAgent/ai_trading_agent/venv/Scripts/python.exe train.py --backtest-only
```
