import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_dataset(
    df,
    sequence_length=20,
    test_fraction=0.2,
    split_date=None,
    return_regime_state=False,
    return_train_tickers=False,
):

    features = [
        "returns",
        "ma10_ma50_ratio",
        "ma20_ma50_ratio",
        "ma50_ma200_ratio",
        "volatility",
        "xs_rsi_rank",
        "xs_momentum_rank",
        "xs_volume_ema_rank",
        "nifty_return",
        "xs_distance_from_52w_high_rank",
        "xs_rs_vs_nifty_rank",
        "xs_rs_vs_nifty_20d_rank",
        "xs_volume_spike_rank",
        "nifty_trend",
        "market_volatility",
        "market_volatility_60d",
        "volatility_regime",
        "nifty_drawdown_63d",
        "nifty_slope_20",
        "adx_14",
        "trend_strength",
        "regime_state",
        "xs_return_rank",
        "xs_overnight_gap_rank",
        "xs_volume_surprise_rank",
        "xs_return_over_atr14_rank",
        "nifty_atr_range_pct",
        "regime_safety_strict",
    ]

    sequences_train = []
    labels_train = []
    dates_train = []
    tickers_train = []
    regimes_train = []
    sequences_test = []
    labels_test = []
    tickers_test = []
    dates_test = []
    regimes_test = []

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
            regime_state = stock_df["regime_state"].values[i + sequence_length]

            if label_date < split_date:
                sequences_train.append(seq)
                labels_train.append(label)
                dates_train.append(label_date)
                tickers_train.append(ticker)
                regimes_train.append(regime_state)
            else:
                sequences_test.append(seq)
                labels_test.append(label)
                tickers_test.append(ticker)
                dates_test.append(label_date)
                regimes_test.append(regime_state)

    X_train = np.array(sequences_train)
    y_train = np.array(labels_train)
    dates_train = np.array(dates_train)
    tickers_train = np.array(tickers_train)
    regimes_train = np.array(regimes_train)
    X_test = np.array(sequences_test)
    y_test = np.array(labels_test)
    tickers_test = np.array(tickers_test)
    dates_test = np.array(dates_test)
    regimes_test = np.array(regimes_test)

    if return_regime_state and return_train_tickers:
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            dates_train,
            tickers_train,
            tickers_test,
            dates_test,
            regimes_train,
            regimes_test,
        )

    if return_regime_state:
        return X_train, X_test, y_train, y_test, dates_train, tickers_test, dates_test, regimes_train, regimes_test

    if return_train_tickers:
        return X_train, X_test, y_train, y_test, dates_train, tickers_train, tickers_test, dates_test

    return X_train, X_test, y_train, y_test, dates_train, tickers_test, dates_test