import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_dataset(df, sequence_length=20):

    features = [
        "returns",
        "ma10",
        "ma20",
        "ma50",
        "volatility",
        "rsi",
        "momentum",
        "volume_ema",
        "nifty_return"
    ]

    sequences = []
    labels = []
    tickers = []
    dates = []

    # Keep chronological order explicitly by Date, not dataframe index.
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    for ticker in df["Ticker"].unique():

        stock_df = df[df["Ticker"] == ticker].copy()
        stock_df = stock_df.sort_values("Date").reset_index(drop=True)

        X = stock_df[features].values
        y = stock_df["target"].values
        date_index = stock_df["Date"].values

        # -----------------------------
        # SCALE PER STOCK (NO LEAKAGE FIX LATER)
        # -----------------------------
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        for i in range(len(X) - sequence_length):

            sequences.append(X[i:i + sequence_length])
            labels.append(y[i + sequence_length])
            tickers.append(ticker)
            dates.append(date_index[i + sequence_length])

    X_seq = np.array(sequences)
    y_seq = np.array(labels)
    tickers = np.array(tickers)
    dates = np.array(dates)

    # -----------------------------
    # TIME-BASED SPLIT (IMPORTANT)
    # -----------------------------
    split = int(0.8 * len(X_seq))

    X_train = X_seq[:split]
    X_test = X_seq[split:]

    y_train = y_seq[:split]
    y_test = y_seq[split:]

    tickers_test = tickers[split:]
    dates_test = dates[split:]

    return X_train, X_test, y_train, y_test, tickers_test, dates_test