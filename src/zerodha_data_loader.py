"""
Hybrid data loader: yfinance historical + Zerodha live for current day.
Ensures forward paper trading has fresh OHLCV data for today.
"""

import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import Zerodha REST client
from .zerodha_rest_client import ZerodhaRestClient, ZerodhaRestConfig


def get_zerodha_client():
    """Initialize Zerodha REST client from environment variables."""
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    
    if not all([api_key, api_secret, access_token]):
        raise RuntimeError(
            "Zerodha credentials missing. Set ZERODHA_API_KEY, "
            "ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN in .env"
        )
    
    config = ZerodhaRestConfig(
        api_key=api_key,
        api_secret=api_secret,
        access_token=access_token
    )
    
    return ZerodhaRestClient(config=config)


def get_ticker_mapping():
    """Map ticker symbols to Zerodha NSE identifiers."""
    return {
        "RELIANCE.NS": "NSE:RELIANCE",
        "TCS.NS": "NSE:TCS",
        "INFY.NS": "NSE:INFY",
        "HDFCBANK.NS": "NSE:HDFCBANK",
        "ICICIBANK.NS": "NSE:ICICIBANK",
        "SBIN.NS": "NSE:SBIN",
        "LT.NS": "NSE:LT",
        "ITC.NS": "NSE:ITC",
        "BHARTIARTL.NS": "NSE:BHARTIARTL",
        "HINDUNILVR.NS": "NSE:HINDUNILVR",
    }


def fetch_zerodha_live_data():
    """
    Fetch latest LTP from Zerodha for today.
    Returns dict with ticker -> OHLCV row for today.
    """
    try:
        client = get_zerodha_client()
        ticker_map = get_ticker_mapping()
        zerodha_symbols = list(ticker_map.values())
        
        # Fetch latest prices
        ltp_data = client.ltp(zerodha_symbols)
        
        today = pd.Timestamp.now()  # Use pd.Timestamp for consistency
        live_rows = {}
        
        for yf_ticker, z_symbol in ticker_map.items():
            if z_symbol in ltp_data:
                quote = ltp_data[z_symbol]
                # Use LTP as close; for OHLC, use LTP for all (conservative estimate)
                live_rows[yf_ticker] = {
                    "Date": today,  # pd.Timestamp
                    "Open": quote.get("last_price", quote.get("ltp")),
                    "High": quote.get("last_price", quote.get("ltp")),
                    "Low": quote.get("last_price", quote.get("ltp")),
                    "Close": quote.get("last_price", quote.get("ltp")),
                    "Adj Close": quote.get("last_price", quote.get("ltp")),
                    "Volume": quote.get("volume", 0),
                    "Ticker": yf_ticker,
                }
            else:
                print(f"Warning: {z_symbol} not found in Zerodha LTP response")
        
        return live_rows, today
    
    except Exception as e:
        print(f"Warning: Failed to fetch Zerodha live data: {e}")
        print("Falling back to yfinance only")
        return {}, None


def download_data(use_zerodha_live: bool = True):
    """
    Download data combining yfinance historical + Zerodha live.
    
    Args:
        use_zerodha_live: If True, append today's Zerodha LTP data.
                         If False, use yfinance only.
    
    Returns:
        DataFrame with Date, Open, High, Low, Close, Adj Close, Volume, Ticker
    """
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "LT.NS",
        "ITC.NS",
        "BHARTIARTL.NS",
        "HINDUNILVR.NS"
    ]

    frames = []

    # Download yfinance data for historical features
    for ticker in tickers:
        print(f"Downloading {ticker} (yfinance historical)")
        df = yf.download(ticker, start="2015-01-01", progress=False)

        if df is None or df.empty:
            print(f"Skipping {ticker} (download returned empty data)")
            continue

        # Flatten any multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["Ticker"] = ticker
        frames.append(df)

    if not frames:
        raise RuntimeError("No stock data downloaded successfully from Yahoo Finance.")

    data = pd.concat(frames, ignore_index=True)

    # Enhance with Zerodha live data for today if available
    if use_zerodha_live:
        print("Fetching live data from Zerodha...")
        live_rows, today = fetch_zerodha_live_data()
        
        if live_rows and today:
            live_frames = [pd.DataFrame([row]) for row in live_rows.values()]
            live_df = pd.concat(live_frames, ignore_index=True)
            
            # Append live data (will be the "latest" date in data)
            data = pd.concat([data, live_df], ignore_index=True)
            print(f"Appended {len(live_df)} rows of live Zerodha data for {today}")
    
    # Download Nifty index
    print("Downloading Nifty index (yfinance)")
    nifty = yf.download("^NSEI", start="2015-01-01", progress=False)

    if nifty is None or nifty.empty:
        raise RuntimeError("Failed to download Nifty index (^NSEI) data.")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    nifty = nifty.reset_index()

    nifty["nifty_return"] = nifty["Close"].pct_change()
    nifty["nifty_close"] = nifty["Close"]

    data = data.merge(
        nifty[["Date", "nifty_return", "nifty_close"]],
        on="Date",
        how="left"
    )

    return data
