import yfinance as yf
import pandas as pd


def download_data():

    # 24 stocks: 20 stable large-caps + 4 aggressive mid-caps
    tickers = [
        # === STABLE LARGE-CAP (Dividend, Liquidity, Lower Volatility) ===
        "RELIANCE.NS",      # Energy/Conglomerate
        "TCS.NS",           # IT
        "INFY.NS",          # IT
        "HDFCBANK.NS",      # Banking
        "ICICIBANK.NS",     # Banking
        "SBIN.NS",          # Banking
        "LT.NS",            # Capital Goods
        "ITC.NS",           # Consumer/Tobacco
        "BHARTIARTL.NS",    # Telecom
        "HINDUNILVR.NS",    # Consumer
        "MARUTI.NS",        # Auto
        "BAJAJFINSV.NS",    # Finance
        "POWERGRID.NS",     # Power
        "ADANIPORTS.NS",    # Ports
        "ASIANPAINT.NS",    # Paint/Chemicals
        "SUNPHARMA.NS",     # Pharma
        "AXISBANK.NS",      # Banking
        "JSWSTEEL.NS",      # Steel
        "WIPRO.NS",         # IT
        "NTPC.NS",          # Power
        # === AGGRESSIVE MID-CAP (Growth, Higher Volatility) ===
        "TATASTEEL.NS",     # Steel/Cyclicals (growth potential)
        "ULTRACEMCO.NS",    # Cement (construction exposure)
        "IDFCFIRSTB.NS",    # Banking (growth-oriented, listed as IDFCFIRSTB)
        "EICHERMOT.NS",     # Auto/Motorcycles (defensive growth)
    ]

    frames = []

    for ticker in tickers:

        print("Downloading", ticker)

        df = yf.download(ticker, start="2011-01-01")

        if df is None or df.empty:
            print("Skipping", ticker, "(download returned empty data)")
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

    # Download Nifty index
    nifty = yf.download("^NSEI", start="2011-01-01")

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