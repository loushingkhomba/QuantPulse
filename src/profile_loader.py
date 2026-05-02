"""
Profile loader: reads configs/profile_<QUANT_PROFILE>.json.
QUANT_PROFILE defaults to 'live' so existing runs are unaffected.

Usage:
    from src.profile_loader import get_active_profile, get_ticker_map, get_cache_tag

Switch profile:
    $env:QUANT_PROFILE = "expanded"   # PowerShell
    export QUANT_PROFILE=expanded      # bash
"""
import json
import os
from pathlib import Path
from typing import Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

# Hardcoded live fallback in case config file missing (keeps backward compat).
_LIVE_FALLBACK_TICKERS: Dict[str, str] = {
    "RELIANCE.NS": "RELIANCE",
    "TCS.NS": "TCS",
    "INFY.NS": "INFY",
    "HDFCBANK.NS": "HDFCBANK",
    "ICICIBANK.NS": "ICICIBANK",
    "SBIN.NS": "SBIN",
    "LT.NS": "LT",
    "ITC.NS": "ITC",
    "BHARTIARTL.NS": "BHARTIARTL",
    "HINDUNILVR.NS": "HINDUNILVR",
}


def get_active_profile() -> dict:
    """Load and return the active profile dict."""
    profile_name = os.getenv("QUANT_PROFILE", "live").strip().lower()
    config_path = _CONFIGS_DIR / f"profile_{profile_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Profile config not found: {config_path}. "
            f"Set QUANT_PROFILE to a name matching configs/profile_<name>.json"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ticker_map() -> Dict[str, str]:
    """
    Return {yfinance_ticker: nse_symbol} for active profile.
    Falls back to hardcoded live list if config missing.
    """
    try:
        profile = get_active_profile()
        return dict(profile["tickers"])
    except FileNotFoundError:
        return _LIVE_FALLBACK_TICKERS.copy()


def get_cache_tag() -> str:
    """Return cache suffix for active profile (e.g. 'live', 'expanded')."""
    try:
        profile = get_active_profile()
        return str(profile.get("cache_tag", os.getenv("QUANT_PROFILE", "live")))
    except FileNotFoundError:
        return "live"


def get_model_path() -> str:
    """Return model .pth path for active profile."""
    try:
        profile = get_active_profile()
        return str(_PROJECT_ROOT / profile["model"]["path"])
    except (FileNotFoundError, KeyError):
        return str(_PROJECT_ROOT / "models" / "quantpulse_model.pth")


def get_hidden_size() -> int:
    """Return model hidden_size for active profile."""
    try:
        profile = get_active_profile()
        return int(profile["model"]["hidden_size"])
    except (FileNotFoundError, KeyError, ValueError):
        return 64


def print_active_profile_summary():
    profile_name = os.getenv("QUANT_PROFILE", "live").strip().lower()
    ticker_map = get_ticker_map()
    cache_tag = get_cache_tag()
    print(f"QUANT_PROFILE={profile_name} | tickers={len(ticker_map)} | cache_tag={cache_tag}")
    print(f"  Tickers: {', '.join(ticker_map.keys())}")
