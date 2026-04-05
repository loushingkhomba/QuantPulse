import yfinance as yf
import pandas as pd


def download_data():

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

    for ticker in tickers:

        print("Downloading", ticker)

        df = yf.download(ticker, start="2015-01-01")

        # Flatten any multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df["Ticker"] = ticker

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Download Nifty index
    nifty = yf.download("^NSEI", start="2015-01-01")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    nifty = nifty.reset_index()

    nifty["nifty_return"] = nifty["Close"].pct_change()

    data = data.merge(
        nifty[["Date", "nifty_return"]],
        on="Date",
        how="left"
    )

    return data