import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_dataset(df, sequence_length=20, test_fraction=0.2, split_date=None):

    features = [
        "returns",
        "ma10_ma50_ratio",
        "ma20_ma50_ratio",
        "ma50_ma200_ratio",
        "volatility",
        "rsi",
        "momentum",
        "volume_ema",
        "nifty_return",
        "distance_from_52w_high",
        "rs_vs_nifty",
        "rs_vs_nifty_20d",
        "volume_spike",
        "nifty_trend",
        "market_volatility",
        "market_volatility_60d",
        "volatility_regime",
        "nifty_drawdown_63d",
        "nifty_slope_20",
        "adx_14",
        "trend_strength",
        "regime_state"
    ]

    sequences_train = []
    labels_train = []
    dates_train = []
    sequences_test = []
    labels_test = []
    tickers_test = []
    dates_test = []

    # Keep chronological order explicitly by Date.
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    all_dates = np.sort(df["Date"].unique())

    if len(all_dates) == 0:
        raise RuntimeError("No rows available after feature engineering; cannot build dataset.")

    if split_date is None:
        test_fraction = min(max(test_fraction, 0.05), 0.5)
        split_idx = int((1.0 - test_fraction) * len(all_dates))
        split_idx = min(split_idx, len(all_dates) - 1)
        split_date = all_dates[split_idx]
    else:
        split_date = np.datetime64(split_date)

    for ticker in df["Ticker"].unique():

        stock_df = df[df["Ticker"] == ticker].copy()
        stock_df = stock_df.sort_values("Date").reset_index(drop=True)

        X_raw = stock_df[features].values
        y = stock_df["target"].values
        date_index = stock_df["Date"].values

        # Fit scaler only on train period for each stock.
        train_mask = date_index < split_date
        if train_mask.sum() == 0:
            continue

        scaler = StandardScaler()
        scaler.fit(X_raw[train_mask])
        X = scaler.transform(X_raw)

        for i in range(len(X) - sequence_length):
            seq = X[i:i + sequence_length]
            label = y[i + sequence_length]
            label_date = date_index[i + sequence_length]

            if label_date < split_date:
                sequences_train.append(seq)
                labels_train.append(label)
                dates_train.append(label_date)
            else:
                sequences_test.append(seq)
                labels_test.append(label)
                tickers_test.append(ticker)
                dates_test.append(label_date)

    X_train = np.array(sequences_train)
    y_train = np.array(labels_train)
    dates_train = np.array(dates_train)
    X_test = np.array(sequences_test)
    y_test = np.array(labels_test)
    tickers_test = np.array(tickers_test)
    dates_test = np.array(dates_test)

    return X_train, X_test, y_train, y_test, dates_train, tickers_test, dates_test