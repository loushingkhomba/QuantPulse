# QuantPulse Improvement Execution Checklist

## Phase 1: Evaluation Integrity and Baseline Trust
- [x] Create implementation checklist
- [x] Keep IC-based orientation as diagnostic-only by default
- [x] Keep inverted-signal comparison as diagnostic-only by default
- [x] Recompute inverted confidence after any execution transform decision
- [x] Persist execution-transform flags in diagnostics JSON
- [x] Validate script syntax

## Phase 2: Backtest Accounting Realism
- [x] Replace single-step horizon PnL shortcut with overlapping position ledger
- [x] Compute annualization using elapsed calendar/trading horizon
- [x] Add accounting consistency checks (exposure, cash, position lifecycle)

## Phase 3: Walk-Forward Governance
- [ ] Add pass/fail gates for Sharpe, DD, turnover, and baseline delta
- [ ] Add stability gate: IC drop must not exceed 50% vs previous window
- [ ] Emit machine-readable gate report per window

## Phase 4: Robustness and Sensitivity
- [ ] Run multi-seed random baseline sweeps and summarize dispersion
- [ ] Add stress toggles (missing data, slippage shock, transaction cost shock)
- [ ] Persist robustness summary artifact

## Phase 4.5: Feature and Signal Layer Upgrade
- [ ] Add cross-sectional relative-strength features
- [ ] Add regime-conditioned feature variants
- [ ] Add feature stability filter (retain consistently positive IC features)
- [ ] Re-train and compare uplift vs corrected baseline

## Phase 5: Promotion Criteria
- [ ] Freeze promotion thresholds after corrected pipeline rerun
- [ ] Generate before/after comparison report
- [ ] Define rollback criteria for production deployment
