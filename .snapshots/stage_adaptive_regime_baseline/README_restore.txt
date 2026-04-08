Stage baseline: adaptive-regime-pre-tuning
Created: 2026-04-08
Purpose: Restore point before new adaptive regime and threshold experiments.

Files captured:
- train.py
- src/features.py
- src/dataset.py
- .walk_forward_robustness.py
- .universal_robustness_check.py
- .robustness_reality_check.py

Restore commands (PowerShell from repo root):
Copy-Item .\.snapshots\stage_adaptive_regime_baseline\train.py .\train.py -Force
Copy-Item .\.snapshots\stage_adaptive_regime_baseline\features.py .\src\features.py -Force
Copy-Item .\.snapshots\stage_adaptive_regime_baseline\dataset.py .\src\dataset.py -Force
Copy-Item .\.snapshots\stage_adaptive_regime_baseline\walk_forward_robustness.py .\.walk_forward_robustness.py -Force
Copy-Item .\.snapshots\stage_adaptive_regime_baseline\universal_robustness_check.py .\.universal_robustness_check.py -Force
Copy-Item .\.snapshots\stage_adaptive_regime_baseline\robustness_reality_check.py .\.robustness_reality_check.py -Force

Guardrails for next changes:
- Do not modify execution layer logic (position entry/exit mechanics, slippage/fee math, holding horizon sequencing).
- Restrict experiments to signal conditioning and thresholding only.
- Re-run walk-forward + universal + reality checks after each tuning step.
