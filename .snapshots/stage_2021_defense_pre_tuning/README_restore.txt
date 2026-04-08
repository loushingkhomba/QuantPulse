Stage baseline: pre-2021-defense-tuning
Created: 2026-04-08
Purpose: Restore point before targeted bad-regime defense changes.

Files captured:
- train.py
- src/features.py
- src/dataset.py
- .walk_forward_robustness.py

Restore commands (PowerShell from repo root):
Copy-Item .\.snapshots\stage_2021_defense_pre_tuning\train.py .\train.py -Force
Copy-Item .\.snapshots\stage_2021_defense_pre_tuning\features.py .\src\features.py -Force
Copy-Item .\.snapshots\stage_2021_defense_pre_tuning\dataset.py .\src\dataset.py -Force
Copy-Item .\.snapshots\stage_2021_defense_pre_tuning\walk_forward_robustness.py .\.walk_forward_robustness.py -Force
