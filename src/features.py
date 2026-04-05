import ta

def create_features(df):

    df = df.copy()
    df = df.reset_index()
    df["returns"] = df.groupby("Ticker")["Close"].pct_change()

    df["ma10"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(10).mean()
    )

    df["ma20"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(20).mean()
    )

    df["ma50"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(50).mean()
    )

    df["volatility"] = df.groupby("Ticker")["returns"].transform(
        lambda x: x.rolling(10).std()
    )

    df["rsi"] = df.groupby("Ticker")["Close"].transform(
        lambda x: ta.momentum.RSIIndicator(close=x).rsi()
    )

    df["momentum"] = df.groupby("Ticker")["Close"].transform(
        lambda x: ta.momentum.ROCIndicator(close=x).roc()
    )

    df["volume_ema"] = df.groupby("Ticker")["Volume"].transform(
        lambda x: x.ewm(span=10).mean()
    )

    # 🚀 NEW TARGET (5-DAY)
    df["future_return_5"] = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(-5) / x - 1
    )

    # rank stocks within each day
    df["rank"] = df.groupby("Date")["future_return_5"].rank(pct=True)

    # top 30% stocks = 1, rest = 0
    df["target"] = (df["rank"] >= 0.7).astype(int)

    df = df.dropna()
    df.drop(columns=["future_return_5", "rank"], inplace=True)

    # 🔍 SANITY CHECK
    print("\nTarget Distribution:")
    print(df["target"].value_counts(normalize=True))

    return df