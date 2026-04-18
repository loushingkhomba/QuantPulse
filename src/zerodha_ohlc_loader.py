"""
Hybrid Zerodha data loader.
Uses historical daily candles for feature history and appends a live
same-day OHLC snapshot from Zerodha quote endpoints.
"""

import os
import shutil
import time
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


def _format_dt(date_value, is_end=False):
    suffix = "23:59:59" if is_end else "00:00:00"
    return f"{date_value.strftime('%Y-%m-%d')} {suffix}"


def _fetch_candles_chunked(client, instrument_token, start_date, end_date):
    """Fetch historical candles in smaller windows to avoid API range-limit failures."""
    base_chunk_days = max(30, int(os.getenv("QUANT_HISTORY_CHUNK_DAYS", "365")))
    min_chunk_days = max(15, int(os.getenv("QUANT_HISTORY_MIN_CHUNK_DAYS", "30")))
    sleep_seconds = float(os.getenv("QUANT_ZERODHA_CALL_SLEEP_SEC", "0.35"))
    max_retries = max(1, int(os.getenv("QUANT_ZERODHA_MAX_RETRIES", "4")))
    all_candles = []
    cursor = start_date

    while cursor <= end_date:
        chunk_days = base_chunk_days
        fetched_chunk = None
        chunk_end_used = min(end_date, cursor + timedelta(days=chunk_days - 1))

        while chunk_days >= min_chunk_days:
            chunk_end = min(end_date, cursor + timedelta(days=chunk_days - 1))
            chunk_end_used = chunk_end
            try:
                candles = None
                for attempt in range(max_retries):
                    try:
                        candles = client.historical_candles(
                            instrument_token=instrument_token,
                            interval="day",
                            from_date=_format_dt(cursor, is_end=False),
                            to_date=_format_dt(chunk_end, is_end=True),
                        )
                        break
                    except Exception as retry_error:
                        retryable = ("429" in str(retry_error)) or ("Too Many Requests" in str(retry_error))
                        if ("403" in str(retry_error)) and (attempt < (max_retries - 1)):
                            retryable = True

                        if (not retryable) or (attempt >= (max_retries - 1)):
                            raise

                        backoff = min(6.0, sleep_seconds * (2 ** attempt))
                        print(
                            "Retrying chunk fetch after transient API error",
                            {
                                "from": cursor.isoformat(),
                                "to": chunk_end.isoformat(),
                                "attempt": int(attempt + 1),
                                "backoff_sec": float(backoff),
                                "error": str(retry_error),
                            },
                        )
                        time.sleep(backoff)

                fetched_chunk = candles or []
                break
            except Exception as e:
                if chunk_days == min_chunk_days:
                    print(
                        "Warning: chunk fetch failed even at min chunk size",
                        {
                            "from": cursor.isoformat(),
                            "to": chunk_end.isoformat(),
                            "chunk_days": chunk_days,
                            "error": str(e),
                        },
                    )
                    fetched_chunk = []
                    break

                next_chunk_days = max(min_chunk_days, chunk_days // 2)
                print(
                    "Warning: reducing chunk size after fetch failure",
                    {
                        "from": cursor.isoformat(),
                        "to": chunk_end.isoformat(),
                        "chunk_days": chunk_days,
                        "next_chunk_days": next_chunk_days,
                        "error": str(e),
                    },
                )
                chunk_days = next_chunk_days

        all_candles.extend(fetched_chunk)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        # Advance to the day after the actual attempted chunk-end, not the base window.
        cursor = chunk_end_used + timedelta(days=1)

    return all_candles


def _build_nse_instrument_map(client):
    """Fetch instruments once and build a tradingsymbol->instrument_token map."""
    instruments = client.instruments(exchange="NSE")
    mapping = {}
    for instr in instruments:
        symbol = instr.get("tradingsymbol")
        token = instr.get("instrument_token")
        if symbol and token:
            mapping[str(symbol)] = str(token)
    return mapping


def _default_project_cache_dir():
    # Keep cache inside repository root so it is easy to find and reuse.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "data", "cache")


def _legacy_user_cache_dir():
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        return os.path.join(local_app_data, "QuantPulse", "cache")
    return os.path.join(os.path.expanduser("~"), ".quantpulse", "cache")


def _normalize_cached_data(cached_data):
    data = cached_data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["Date"] = data["Date"].apply(_strip_timezone)
    data = data.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"], keep="last")
    return data


def download_data():
    """
    Download OHLCV data from Zerodha for all tracked stocks.
    Returns DataFrame with Date, Open, High, Low, Close, Adj Close, Volume, Ticker
    
    Note: Lookback is controlled by QUANT_HISTORY_YEARS and fetched in chunks.
    """
    history_years = max(1, int(os.getenv("QUANT_HISTORY_YEARS", "5")))
    use_cache = os.getenv("QUANT_USE_DATA_CACHE", "1").strip() == "1"
    refresh_cache = os.getenv("QUANT_REFRESH_DATA", "0").strip() == "1"
    incremental_refresh = os.getenv("QUANT_INCREMENTAL_REFRESH", "1").strip() == "1"
    cache_dir = os.getenv("QUANT_DATA_CACHE_DIR", _default_project_cache_dir())
    cache_file = os.path.join(cache_dir, f"zerodha_dataset_{history_years}y.pkl")
    legacy_user_cache_file = os.path.join(_legacy_user_cache_dir(), f"zerodha_dataset_{history_years}y.pkl")

    if use_cache and (not refresh_cache) and (not os.path.exists(cache_file)) and os.path.exists(legacy_user_cache_file):
        os.makedirs(cache_dir, exist_ok=True)
        shutil.copy2(legacy_user_cache_file, cache_file)
        print(f"Migrated user cache to project cache location: {cache_file}")

    cached_data = None
    if use_cache and os.path.exists(cache_file):
        cached_data = _normalize_cached_data(pd.read_pickle(cache_file))

    if use_cache and (not refresh_cache) and (cached_data is not None):
        print(f"Loading cached market data: {cache_file}")
        print(f"Cached records: {len(cached_data)}")
        print(f"Cached date range: {cached_data['Date'].min()} to {cached_data['Date'].max()}")
        return cached_data

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
    nse_instrument_map = {}

    today = datetime.now().date()
    cached_max_date = None
    if cached_data is not None and (not cached_data.empty):
        cached_max_date = pd.Timestamp(cached_data["Date"].max()).date()

    incremental_start_date = None
    if refresh_cache and incremental_refresh and (cached_max_date is not None):
        incremental_start_date = cached_max_date + timedelta(days=1)
        print(
            "Incremental refresh enabled:",
            {
                "cached_max_date": cached_max_date.isoformat(),
                "incremental_start_date": incremental_start_date.isoformat(),
                "today": today.isoformat(),
            },
        )

    if (incremental_start_date is not None) and (incremental_start_date > today):
        print("Cache already up to date for historical candles. Only live snapshot refresh will run.")
    fallback_candidates = [history_years, 10, 7, 5, 3, 1]
    attempt_years_list = []
    for years in fallback_candidates:
        years = max(1, int(years))
        if years not in attempt_years_list:
            attempt_years_list.append(years)

    actual_years_used = history_years
    date_str_to = _format_dt(today, is_end=True)

    for attempt_years in attempt_years_list:
        if incremental_start_date is not None:
            start_date = incremental_start_date
        else:
            start_date = today - timedelta(days=365 * attempt_years)
        date_str_from = _format_dt(start_date, is_end=False)
        frames = []

        print(f"Fetching Zerodha candles ({attempt_years} years): {date_str_from} to {date_str_to}")

        if not nse_instrument_map:
            print("Loading NSE instrument map once...")
            nse_instrument_map = _build_nse_instrument_map(client)

        for yfinance_ticker, nse_symbol in ticker_map.items():
            try:
                print(f"Fetching {nse_symbol}...")

                token = nse_instrument_map.get(nse_symbol)
                if not token:
                    print(f"Warning: Instrument token missing for {nse_symbol}")
                    continue

                if start_date <= today:
                    candles = _fetch_candles_chunked(
                        client=client,
                        instrument_token=token,
                        start_date=start_date,
                        end_date=today,
                    )
                else:
                    candles = []

                if not candles:
                    print(f"Warning: No candles fetched for {nse_symbol}")
                    continue

                rows = []
                for candle in candles:
                    if len(candle) >= 6:
                        rows.append({
                            "Date": candle[0],
                            "Open": float(candle[1]),
                            "High": float(candle[2]),
                            "Low": float(candle[3]),
                            "Close": float(candle[4]),
                            "Adj Close": float(candle[4]),
                            "Volume": int(candle[5]),
                            "Ticker": yfinance_ticker,
                        })

                if rows:
                    df = pd.DataFrame(rows)
                    df["Date"] = pd.to_datetime(df["Date"])
                    frames.append(df)
                    print(f"  OK {len(rows)} candles")

            except Exception as e:
                print(f"Warning: Failed to fetch {nse_symbol}: {e}")
                continue

        if frames:
            actual_years_used = attempt_years
            if actual_years_used != history_years:
                print(
                    f"Requested {history_years} years but Zerodha only returned data for {actual_years_used} years."
                )
            break

        if incremental_start_date is not None:
            # Incremental path can legitimately have no historical candles (already up to date).
            break

        print(f"No usable candles for {attempt_years} years, trying a shorter lookback...")

    if (cached_data is None) and (not frames):
        raise RuntimeError("No stock data fetched from Zerodha. Check credentials and instrument availability.")

    stock_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    nifty_cols = ["Date", "nifty_open", "nifty_high", "nifty_low", "nifty_return", "nifty_close"]
    existing_nifty = None
    if cached_data is not None:
        existing_stock = cached_data[[c for c in stock_cols if c in cached_data.columns]].copy()
        existing_nifty_cols = [c for c in nifty_cols if c in cached_data.columns]
        if existing_nifty_cols:
            existing_nifty = (
                cached_data[["Date", *[c for c in existing_nifty_cols if c != "Date"]]]
                .drop_duplicates(subset=["Date"], keep="last")
                .copy()
            )
    else:
        existing_stock = pd.DataFrame(columns=stock_cols)

    new_stock = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=stock_cols)

    # Combine existing + newly fetched stock data.
    data = pd.concat([existing_stock, new_stock], ignore_index=True)
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
            print(f"  OK Appended {len(live_df)} live snapshot rows for {live_date.date()}")
    except Exception as e:
        print(f"Warning: Failed to fetch live Zerodha snapshot: {e}")
    
    # Fetch Nifty index data or create synthetic
    print("Fetching Nifty index...")
    nifty_fetched = False
    fetched_nifty = None

    nifty_fetch_start = incremental_start_date
    if nifty_fetch_start is None:
        nifty_fetch_start = today - timedelta(days=365 * actual_years_used)
    
    # Try different Nifty symbol variants
    for nifty_symbol in ["NIFTY50", "NIFTYNXT50", "NIFTY 50", "NIFTY"]:
        try:
            nifty_token = client.get_instrument_token(
                exchange="NSE",
                tradingsymbol=nifty_symbol,
                cache=instrument_cache
            )
            print(f"  Found Nifty: {nifty_symbol}")
            
            if nifty_fetch_start <= today:
                nifty_candles = _fetch_candles_chunked(
                    client=client,
                    instrument_token=nifty_token,
                    start_date=nifty_fetch_start,
                    end_date=today,
                )
            else:
                nifty_candles = []
            
            if nifty_candles:
                nifty_rows = []
                for candle in nifty_candles:
                    if len(candle) >= 5:
                        nifty_rows.append({
                            "Date": candle[0],
                            "nifty_open": float(candle[1]),
                            "nifty_high": float(candle[2]),
                            "nifty_low": float(candle[3]),
                            "nifty_close": float(candle[4]),
                        })
                
                if nifty_rows:
                    nifty = pd.DataFrame(nifty_rows)
                    nifty["Date"] = pd.to_datetime(nifty["Date"])
                    nifty["Date"] = nifty["Date"].apply(_strip_timezone)
                    nifty["nifty_return"] = nifty["nifty_close"].pct_change()
                    fetched_nifty = nifty[["Date", "nifty_open", "nifty_high", "nifty_low", "nifty_return", "nifty_close"]].copy()
                    print(f"  OK {len(nifty_rows)} nifty candles")
                    nifty_fetched = True
                    break
        except (ValueError, Exception):
            continue
    
    # Fallback: Create synthetic Nifty from top stocks if real Nifty not found
    if not nifty_fetched:
        print("  Nifty not found. Creating synthetic Nifty from top 3 stocks...")
        synth_start = nifty_fetch_start if nifty_fetch_start is not None else data["Date"].min()
        synth_source = data[data["Date"] >= pd.Timestamp(synth_start)].copy()
        nifty = synth_source[synth_source["Ticker"].isin(["RELIANCE.NS", "TCS.NS", "INFY.NS"])].copy()
        nifty = nifty.groupby("Date")["Close"].mean().reset_index()
        nifty.columns = ["Date", "nifty_close"]
        nifty["nifty_open"] = nifty["nifty_close"]
        nifty["nifty_high"] = nifty["nifty_close"]
        nifty["nifty_low"] = nifty["nifty_close"]
        nifty["nifty_return"] = nifty["nifty_close"].pct_change()
        fetched_nifty = nifty[["Date", "nifty_open", "nifty_high", "nifty_low", "nifty_return", "nifty_close"]].copy()
        print(f"  OK {len(nifty)} synthetic nifty rows")

    if existing_nifty is not None and fetched_nifty is not None:
        combined_nifty = pd.concat([existing_nifty, fetched_nifty], ignore_index=True)
    elif existing_nifty is not None:
        combined_nifty = existing_nifty.copy()
    elif fetched_nifty is not None:
        combined_nifty = fetched_nifty.copy()
    else:
        combined_nifty = pd.DataFrame(columns=nifty_cols)

    if live_nifty_close is not None:
        live_nifty_date = pd.Timestamp.now().normalize()
        historical_nifty_close = (
            float(combined_nifty["nifty_close"].dropna().iloc[-1])
            if ("nifty_close" in combined_nifty.columns and not combined_nifty["nifty_close"].dropna().empty)
            else live_nifty_close
        )
        live_nifty_row = pd.DataFrame([
            {
                "Date": live_nifty_date,
                "nifty_open": historical_nifty_close,
                "nifty_high": max(historical_nifty_close, live_nifty_close),
                "nifty_low": min(historical_nifty_close, live_nifty_close),
                "nifty_close": live_nifty_close,
                "nifty_return": (live_nifty_close / historical_nifty_close) - 1 if historical_nifty_close else 0.0,
            }
        ])
        combined_nifty = pd.concat([combined_nifty, live_nifty_row], ignore_index=True)

    if not combined_nifty.empty:
        combined_nifty = combined_nifty.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        if "nifty_return" in combined_nifty.columns:
            combined_nifty["nifty_return"] = combined_nifty["nifty_close"].pct_change()

    data = data.merge(combined_nifty, on="Date", how="left")

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "nifty_open",
        "nifty_high",
        "nifty_low",
        "nifty_close",
        "nifty_return",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    
    print(f"\nTotal records: {len(data)}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

    # Make sure duplicate ticker/date pairs keep the freshest snapshot row.
    data = data.sort_values(["Ticker", "Date"]).drop_duplicates(subset=["Ticker", "Date"], keep="last")

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        data.to_pickle(cache_file)
        print(f"Saved market data cache: {cache_file}")
        if actual_years_used != history_years:
            print(
                f"Cache contains {actual_years_used}-year data due to API lookback limits for requested {history_years} years."
            )
    
    return data
