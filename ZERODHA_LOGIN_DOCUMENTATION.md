# Zerodha Login Documentation

## Purpose
This runbook explains how to authenticate Zerodha for this project and recover quickly when the token expires.

## Prerequisites
- Project root: `c:/Users/fioli/OneDrive/Desktop/project/tradingAIAgent/ai_trading_agent`
- Python environment: `venv`
- Zerodha Kite app configured with redirect URL:
  - `http://127.0.0.1:8765/callback`

## Required Environment Variables
The `.env` file must contain:
- `ZERODHA_API_KEY`
- `ZERODHA_API_SECRET`
- `ZERODHA_ACCESS_TOKEN`

## Full Auto-Login (Recommended)
Run this from project root:

```powershell
c:/Users/fioli/OneDrive/Desktop/project/tradingAIAgent/ai_trading_agent/venv/Scripts/python.exe zerodha_bootstrap.py --mode rest --auto-login --redirect-url http://127.0.0.1:8765/callback --persist-token --save-session
```

What this does:
1. Starts a local callback server at `127.0.0.1:8765`
2. Opens/prints Zerodha login URL
3. Captures `request_token` after browser redirect
4. Generates fresh `ZERODHA_ACCESS_TOKEN`
5. Persists token into `.env`
6. Saves session metadata in `logs/zerodha_session.json`

## Manual Login URL Only
If you only need to print the login URL:

```powershell
c:/Users/fioli/OneDrive/Desktop/project/tradingAIAgent/ai_trading_agent/venv/Scripts/python.exe zerodha_bootstrap.py --mode rest --show-login-url
```

## Health Check After Login
Use this to confirm auth and market quote access:

```powershell
c:/Users/fioli/OneDrive/Desktop/project/tradingAIAgent/ai_trading_agent/venv/Scripts/python.exe -c "from src.zerodha_ohlc_loader import get_zerodha_client; c=get_zerodha_client(); p=c.profile(); q=c.quote_ohlc(['NSE:RELIANCE']); print('profile_ok', p.get('user_name','unknown')); print('quote_ok', list(q.keys())[:1])"
```

Expected success output includes:
- `profile_ok <your_name>`
- `quote_ok ['NSE:RELIANCE']`

## Common Failure and Fix
### Error: HTTP 403 Forbidden
Cause: expired/invalid `ZERODHA_ACCESS_TOKEN`

Fix:
1. Run the full auto-login command again
2. Complete Zerodha login + 2FA in browser
3. Re-run health check

## Notes
- Zerodha access tokens are short-lived and may require daily refresh.
- This project uses `ZERODHA_*` variables (not `KITE_*`).
- Training and backtests that fetch fresh data will fail until auth is valid.
