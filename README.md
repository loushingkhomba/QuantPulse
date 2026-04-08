# QuantPulse

QuantPulse is a machine learning-based stock trading project that trains a sequence model on historical market data, generates daily stock confidence scores, and evaluates a top-k portfolio strategy against a random baseline.

## Features

- Downloads market data with `yfinance`
- Builds technical indicators and a 5-day forward target
- Trains a QuantPulse classification model
- Runs a simple backtest and compares equity curves
- Saves charts and model checkpoints locally

## Project Structure

- `train.py` - end-to-end training and backtesting script
- `src/data_loader.py` - downloads market and index data
- `src/features.py` - creates technical indicators and labels
- `src/dataset.py` - prepares sequences for model training
- `src/model.py` - model architecture
- `src/trainer.py` - training loop
- `src/backtest.py` - performance metrics and backtest helpers
- `src/walkforward.py` - walk-forward evaluation utilities

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

## Zerodha Connection

1. Copy `.env.example` to `.env` and fill in your Zerodha API values.
2. Install dependencies from `requirements.txt`.
3. Run the bootstrap helper.

Show login URL:

```bash
python zerodha_bootstrap.py --show-login-url
```

Near-zero-touch login flow (auto-capture request token on localhost redirect):

```bash
python zerodha_bootstrap.py --mode rest --auto-login --redirect-url http://127.0.0.1:8765/callback --persist-token --save-session
```

Notes:

- Set the same redirect URL in Kite Connect app settings.
- Zerodha 2FA login is mandatory and cannot be bypassed programmatically.
- This flow removes manual token copy-paste by capturing the callback automatically.

After login redirect, use request token to generate access token:

```bash
python zerodha_bootstrap.py --request-token <REQUEST_TOKEN>
```

Validate broker connection:

```bash
python zerodha_bootstrap.py --mode rest --profile --margins --ltp NSE:RELIANCE NSE:TCS
```

Fetch segment-specific margins:

```bash
python zerodha_bootstrap.py --mode rest --margins --margins-segment equity
```

Invalidate API session token:

```bash
python zerodha_bootstrap.py --mode rest --logout
```

## 60-Day Paper Live Campaign

Track daily model signals in paper mode and compare matured returns against a random baseline.

```bash
python paper_live_60d.py
```

Start a true-forward campaign from a specific date with freshness checks:

```bash
python paper_live_60d.py --campaign-name forward_phaseA_2026 --campaign-start-date 2026-04-08 --max-data-lag-days 3
```

Bootstrap historical replay for the last 60 trading days:

```bash
python paper_live_60d.py --replay-last-n-days 60
```

Outputs:

- `logs/paper_live_signals.csv` - per-day model/random picks and realized returns once matured
- `logs/paper_live_summary.json` - campaign performance summary
- `logs/paper_live_state.json` - progress toward 60 signal days

## Go-Live Gating

Machine-readable live gating policy:

- `go_live_policy.json`
- `locked_live_config.json`

Generate weekly pass/fail go-live report:

```bash
python go_live_weekly_report.py
```

Campaign-specific weekly report:

```bash
set QUANT_PAPER_CAMPAIGN_NAME=forward_phaseA_2026
python go_live_weekly_report.py
```

Output:

- `logs/go_live_weekly_report.json` - metrics, gate checks, incidents, slippage monitor, model-decay monitor
- `logs/go_live_gate_state.json` - consecutive weekly pass counter

Incident and fill templates for live operation:

- `logs/operational_incidents.csv`
- `logs/live_fills.csv`

Windows Task Scheduler setup:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
.\setup_go_live_tasks.ps1
```

Plan-date scheduler setup with campaign namespace:

```powershell
.\setup_go_live_plan_tasks.ps1 -CampaignName forward_phaseA_2026
```

Optional custom times:

```powershell
.\setup_go_live_tasks.ps1 -DailyTime 08:45 -WeeklyTime 17:15
```

## Notes

- The project uses a top-k selection strategy for backtesting.
- Generated files such as model checkpoints, logs, and charts are ignored by Git.
