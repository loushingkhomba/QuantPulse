import os

import numpy as np
import pandas as pd
import ta


def _safe_adx(group):
    if {"High", "Low", "Close"}.issubset(group.columns):
        indicator = ta.trend.ADXIndicator(
            high=group["High"],
            low=group["Low"],
            close=group["Close"],
            window=14,
        )
        return indicator.adx()
    return pd.Series(np.nan, index=group.index)

def create_features(df):

    df = df.copy()
    df = df.reset_index()
    disable_regime = os.getenv("QUANT_DISABLE_REGIME_FEATURES", "0").strip() == "1"
    forward_inference = os.getenv("QUANT_FORWARD_INFERENCE", "0").strip() == "1"
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

    df["ma200"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(200).mean()
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

    df["distance_from_52w_high"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x / x.rolling(252).max()
    )

    df["rs_vs_nifty"] = df["returns"] - df["nifty_return"]

    df["rs_vs_nifty_20d"] = df.groupby("Ticker")["rs_vs_nifty"].transform(
        lambda x: x.rolling(20).sum()
    )

    df["volume_spike"] = df.groupby("Ticker")["Volume"].transform(
        lambda x: x / x.rolling(20).mean()
    )

    # Market regime context from Nifty.
    df["nifty_ma50"] = df["nifty_close"].rolling(50).mean()
    df["nifty_ma200"] = df["nifty_close"].rolling(200).mean()
    df["nifty_trend"] = df["nifty_ma50"] / df["nifty_ma200"]
    df["market_volatility"] = df["nifty_return"].rolling(20).std()
    df["market_volatility_60d"] = df["market_volatility"].rolling(60).median()
    df["volatility_regime"] = df["market_volatility"] / df["market_volatility_60d"]
    df["nifty_drawdown_63d"] = df["nifty_close"] / df["nifty_close"].rolling(63).max() - 1
    df["nifty_slope_20"] = np.log(df["nifty_close"]).diff(20) / 20.0

    if {"High", "Low", "Close"}.issubset(df.columns):
        df["adx_14"] = (
            df.groupby("Ticker", group_keys=False)
            .apply(_safe_adx)
            .reset_index(level=0, drop=True)
        )
    else:
        df["adx_14"] = np.nan

    df["trend_strength"] = df["nifty_slope_20"].fillna(0) + (df["adx_14"].fillna(0) / 100.0)
    df["regime_state"] = 0
    df.loc[(df["volatility_regime"] > 1.25) | (df["nifty_drawdown_63d"] < -0.08), "regime_state"] = -1
    df.loc[
        (df["volatility_regime"] < 0.90)
        & (df["nifty_drawdown_63d"] > -0.04)
        & (df["trend_strength"] > 0),
        "regime_state",
    ] = 1

    if disable_regime:
        df["nifty_trend"] = 1.0
        df["market_volatility"] = 0.0
        df["market_volatility_60d"] = 0.0
        df["volatility_regime"] = 1.0
        df["nifty_drawdown_63d"] = 0.0
        df["nifty_slope_20"] = 0.0
        df["adx_14"] = 0.0
        df["trend_strength"] = 0.0
        df["regime_state"] = 0

    # Replace raw level MAs with scale-free ratios.
    df["ma10_ma50_ratio"] = df["ma10"] / df["ma50"]
    df["ma20_ma50_ratio"] = df["ma20"] / df["ma50"]
    df["ma50_ma200_ratio"] = df["ma50"] / df["ma200"]

    # 5-day stock return target components.
    df["future_return_5"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.shift(-5) / x - 1
    )

    # 5-day benchmark return from Nifty close.
    nifty_daily = (
        df[["Date", "nifty_close"]]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .copy()
    )
    nifty_daily["nifty_future_5"] = nifty_daily["nifty_close"].shift(-5) / nifty_daily["nifty_close"] - 1
    df = df.merge(nifty_daily[["Date", "nifty_future_5"]], on="Date", how="left")

    # Excess return target: beat Nifty by 0.5% over 5 days.
    df["excess_return_5"] = df["future_return_5"] - df["nifty_future_5"]
    df["target"] = np.where(
        df["excess_return_5"].notna(),
        (df["excess_return_5"] > 0.005).astype(int),
        np.nan,
    )

    if forward_inference:
        df = df.dropna(subset=[col for col in df.columns if col not in {"future_return_5", "nifty_future_5", "excess_return_5", "target"}])
    else:
        df = df.dropna()
    df.drop(columns=["future_return_5", "nifty_future_5", "excess_return_5"], inplace=True)

    # 🔍 SANITY CHECK
    print("\nTarget Distribution:")
    print(df["target"].value_counts(normalize=True))

    return df