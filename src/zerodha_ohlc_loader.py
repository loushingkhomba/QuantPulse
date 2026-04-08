"""
Hybrid Zerodha data loader.
Uses historical daily candles for feature history and appends a live
same-day OHLC snapshot from Zerodha quote endpoints.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .zerodha_rest_client import ZerodhaRestClient, ZerodhaRestConfig

# Load environment variables
load_dotenv()


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


def _strip_timezone(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def download_data():
    """
    Download OHLCV data from Zerodha for all tracked stocks.
    Returns DataFrame with Date, Open, High, Low, Close, Adj Close, Volume, Ticker
    
    Note: Fetches last 5 years of daily data for feature engineering.
    """
    client = get_zerodha_client()
    
    # Map yfinance tickers to NSE symbols
    ticker_map = {
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
    
    frames = []
    instrument_cache = {}
    
    # Calculate date range (5 years back for historical features)
    today = datetime.now().date()
    start_date = today - timedelta(days=365*5)
    # Zerodha API requires time format: yyyy-mm-dd hh:mm:ss
    date_str_from = start_date.strftime("%Y-%m-%d") + " 00:00:00"
    date_str_to = today.strftime("%Y-%m-%d") + " 23:59:59"
    
    print(f"Fetching Zerodha candles (5 years): {date_str_from} to {date_str_to}")
    
    for yfinance_ticker, nse_symbol in ticker_map.items():
        try:
            print(f"Fetching {nse_symbol}...")
            
            # Get instrument token for NSE symbol
            token = client.get_instrument_token(
                exchange="NSE",
                tradingsymbol=nse_symbol,
                cache=instrument_cache
            )
            
            # Fetch daily candles
            candles = client.historical_candles(
                instrument_token=token,
                interval="day",
                from_date=date_str_from,
                to_date=date_str_to
            )
            
            if not candles:
                print(f"Warning: No candles fetched for {nse_symbol}")
                continue
            
            # Convert candles to DataFrame
            # Each candle: [timestamp, open, high, low, close, volume]
            rows = []
            for candle in candles:
                if len(candle) >= 6:
                    rows.append({
                        "Date": candle[0],  # ISO format timestamp
                        "Open": float(candle[1]),
                        "High": float(candle[2]),
                        "Low": float(candle[3]),
                        "Close": float(candle[4]),
                        "Adj Close": float(candle[4]),  # Use close as adj close
                        "Volume": int(candle[5]),
                        "Ticker": yfinance_ticker,
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                df["Date"] = pd.to_datetime(df["Date"])
                frames.append(df)
                print(f"  ✓ {len(rows)} candles")
            
        except Exception as e:
            print(f"Warning: Failed to fetch {nse_symbol}: {e}")
            continue
    
    if not frames:
        raise RuntimeError("No stock data fetched from Zerodha. Check credentials and instrument availability.")
    
    # Combine all stock data
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = data["Date"].apply(_strip_timezone)

    # Append a same-day live snapshot so the latest row reflects today's market.
    print("Fetching live Zerodha OHLC snapshot...")
    snapshot_rows = []
    snapshot_symbols = {ticker: f"NSE:{symbol}" for ticker, symbol in ticker_map.items()}
    snapshot_keys = list(snapshot_symbols.values())
    live_nifty_close = None

    try:
        live_quotes = client.quote_ohlc(snapshot_keys)
        live_date = pd.Timestamp.now().normalize()

        for yfinance_ticker, quote_symbol in snapshot_symbols.items():
            payload = live_quotes.get(quote_symbol)
            if not payload:
                continue

            ohlc = payload.get("ohlc", {}) or {}
            price = float(payload.get("last_price", ohlc.get("close", 0.0)))
            snapshot_rows.append({
                "Date": live_date,
                "Open": float(ohlc.get("open", price)),
                "High": float(ohlc.get("high", price)),
                "Low": float(ohlc.get("low", price)),
                "Close": price,
                "Adj Close": price,
                "Volume": int(payload.get("volume", 0) or 0),
                "Ticker": yfinance_ticker,
            })

        nifty_payload = live_quotes.get("NSE:NIFTY 50") or live_quotes.get("NSE:NIFTY 50 ")
        if nifty_payload:
            nifty_ohlc = nifty_payload.get("ohlc", {}) or {}
            live_nifty_close = float(nifty_payload.get("last_price", nifty_ohlc.get("close", 0.0)))

        if snapshot_rows:
            live_df = pd.DataFrame(snapshot_rows)
            data = pd.concat([data, live_df], ignore_index=True)
            print(f"  ✓ Appended {len(live_df)} live snapshot rows for {live_date.date()}")
    except Exception as e:
        print(f"Warning: Failed to fetch live Zerodha snapshot: {e}")
    
    # Fetch Nifty index data or create synthetic
    print("Fetching Nifty index...")
    nifty_fetched = False
    
    # Try different Nifty symbol variants
    for nifty_symbol in ["NIFTY50", "NIFTYNXT50", "NIFTY 50", "NIFTY"]:
        try:
            nifty_token = client.get_instrument_token(
                exchange="NSE",
                tradingsymbol=nifty_symbol,
                cache=instrument_cache
            )
            print(f"  Found Nifty: {nifty_symbol}")
            
            nifty_candles = client.historical_candles(
                instrument_token=nifty_token,
                interval="day",
                from_date=date_str_from,
                to_date=date_str_to
            )
            
            if nifty_candles:
                nifty_rows = []
                for candle in nifty_candles:
                    if len(candle) >= 5:
                        nifty_rows.append({
                            "Date": candle[0],
                            "nifty_close": float(candle[4]),
                        })
                
                if nifty_rows:
                    nifty = pd.DataFrame(nifty_rows)
                    nifty["Date"] = pd.to_datetime(nifty["Date"])
                    nifty["Date"] = nifty["Date"].apply(_strip_timezone)
                    nifty["nifty_return"] = nifty["nifty_close"].pct_change()
                    
                    # Merge with main data
                    data = data.merge(
                        nifty[["Date", "nifty_return", "nifty_close"]],
                        on="Date",
                        how="left"
                    )
                    print(f"  ✓ {len(nifty_rows)} nifty candles")
                    nifty_fetched = True
                    break
        except (ValueError, Exception):
            continue
    
    # Fallback: Create synthetic Nifty from top stocks if real Nifty not found
    if not nifty_fetched:
        print("  Nifty not found. Creating synthetic Nifty from top 3 stocks...")
        nifty = data[data["Ticker"].isin(["RELIANCE.NS", "TCS.NS", "INFY.NS"])].copy()
        nifty = nifty.groupby("Date")["Close"].mean().reset_index()
        nifty.columns = ["Date", "nifty_close"]
        nifty["nifty_return"] = nifty["nifty_close"].pct_change()
        
        data = data.merge(
            nifty[["Date", "nifty_return", "nifty_close"]],
            on="Date",
            how="left"
        )
        print(f"  ✓ {len(nifty)} synthetic nifty rows")
    elif live_nifty_close is not None:
        live_nifty_date = pd.Timestamp.now().normalize()
        historical_nifty_close = float(nifty["nifty_close"].dropna().iloc[-1]) if "nifty_close" in nifty.columns and not nifty["nifty_close"].dropna().empty else live_nifty_close
        live_nifty_row = pd.DataFrame([
            {
                "Date": live_nifty_date,
                "nifty_close": live_nifty_close,
                "nifty_return": (live_nifty_close / historical_nifty_close) - 1 if historical_nifty_close else 0.0,
            }
        ])

        data = data.merge(
            pd.concat([nifty[["Date", "nifty_return", "nifty_close"]], live_nifty_row], ignore_index=True),
            on="Date",
            how="left"
        )
    
    print(f"\nTotal records: {len(data)}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

    # Make sure duplicate ticker/date pairs keep the freshest snapshot row.
    data = data.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"], keep="last")
    
    return data
