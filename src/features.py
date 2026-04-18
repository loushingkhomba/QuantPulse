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


def _safe_atr(group):
    if {"High", "Low", "Close"}.issubset(group.columns):
        indicator = ta.volatility.AverageTrueRange(
            high=group["High"],
            low=group["Low"],
            close=group["Close"],
            window=14,
        )
        return indicator.average_true_range()
    return pd.Series(np.nan, index=group.index)


def _cross_sectional_rank(df, columns):
    # Rank feature values by date to reduce market-wide drift/noise.
    for col in columns:
        if col in df.columns:
            df[col] = df.groupby("Date")[col].rank(pct=True)
    return df


def _append_rank_feature(df, source_col, target_col):
    if source_col in df.columns:
        df[target_col] = df.groupby("Date")[source_col].rank(pct=True)
    else:
        df[target_col] = np.nan
    return df

def create_features(df):

    df = df.copy()
    df = df.reset_index()
    disable_regime = os.getenv("QUANT_DISABLE_REGIME_FEATURES", "0").strip() == "1"
    forward_inference = os.getenv("QUANT_FORWARD_INFERENCE", "0").strip() == "1"
    target_horizon = max(1, int(os.getenv("QUANT_TARGET_HORIZON_DAYS", "5")))
    target_mode = os.getenv("QUANT_TARGET_MODE", "alpha").strip().lower()
    alpha_threshold = float(os.getenv("QUANT_TARGET_ALPHA_THRESHOLD", "0.005"))
    absolute_threshold = float(os.getenv("QUANT_TARGET_ABS_THRESHOLD", "0.0"))
    target_cost_bps = float(os.getenv("QUANT_TARGET_COST_BPS", "0.0")) / 10000.0
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

    volume_mean_20 = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20).mean())
    volume_std_20 = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20).std())
    df["volume_surprise_z20"] = (df["Volume"] - volume_mean_20) / (volume_std_20 + 1e-9)

    # Market regime context from Nifty.
    df["nifty_ma50"] = df["nifty_close"].rolling(50).mean()
    df["nifty_ma200"] = df["nifty_close"].rolling(200).mean()
    df["nifty_trend"] = df["nifty_ma50"] / df["nifty_ma200"]
    df["market_volatility"] = df["nifty_return"].rolling(20).std()
    df["market_volatility_60d"] = df["market_volatility"].rolling(60).median()
    df["volatility_regime"] = df["market_volatility"] / df["market_volatility_60d"]
    df["nifty_drawdown_63d"] = df["nifty_close"] / df["nifty_close"].rolling(63).max() - 1
    df["nifty_slope_20"] = np.log(df["nifty_close"]).diff(20) / 20.0

    safety_lookback = max(20, int(os.getenv("QUANT_REGIME_SAFETY_LOOKBACK_DAYS", "90")))
    safety_strict_pct = float(os.getenv("QUANT_REGIME_SAFETY_STRICT_PCT", "0.80"))

    if {"nifty_high", "nifty_low", "nifty_close"}.issubset(df.columns):
        nifty_daily = (
            df[["Date", "nifty_high", "nifty_low", "nifty_close"]]
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .copy()
        )
        indicator = ta.volatility.AverageTrueRange(
            high=nifty_daily["nifty_high"],
            low=nifty_daily["nifty_low"],
            close=nifty_daily["nifty_close"],
            window=14,
        )
        nifty_daily["nifty_atr_frac_14"] = indicator.average_true_range() / (nifty_daily["nifty_close"] + 1e-9)
    else:
        nifty_daily = (
            df[["Date", "market_volatility"]]
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .copy()
        )
        nifty_daily["nifty_atr_frac_14"] = nifty_daily["market_volatility"]

    rolling_min = nifty_daily["nifty_atr_frac_14"].rolling(safety_lookback, min_periods=max(10, safety_lookback // 3)).min()
    rolling_max = nifty_daily["nifty_atr_frac_14"].rolling(safety_lookback, min_periods=max(10, safety_lookback // 3)).max()
    nifty_daily["nifty_atr_range_pct"] = (
        (nifty_daily["nifty_atr_frac_14"] - rolling_min) / ((rolling_max - rolling_min) + 1e-9)
    ).clip(0.0, 1.0)
    nifty_daily["regime_safety_strict"] = (nifty_daily["nifty_atr_range_pct"] >= safety_strict_pct).astype(float)
    df = df.merge(
        nifty_daily[["Date", "nifty_atr_frac_14", "nifty_atr_range_pct", "regime_safety_strict"]],
        on="Date",
        how="left",
    )

    if {"High", "Low", "Close"}.issubset(df.columns):
        df["adx_14"] = (
            df.groupby("Ticker", group_keys=False)
            .apply(_safe_adx)
            .reset_index(level=0, drop=True)
        )
        df["atr_14"] = (
            df.groupby("Ticker", group_keys=False)
            .apply(_safe_atr)
            .reset_index(level=0, drop=True)
        )
    else:
        df["adx_14"] = np.nan
        df["atr_14"] = np.nan

    # Normalize returns by ATR fraction of price to get risk-adjusted move strength.
    df["return_over_atr14"] = df["returns"] / ((df["atr_14"] / df["Close"]) + 1e-9)

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

    # Cross-sectional return rank vs peers same day (removes market beta).
    df["xs_return_rank"] = df.groupby("Date")["returns"].rank(pct=True)

    # Overnight gap: open / prev_close - 1 (intraday sentiment proxy).
    if "Open" in df.columns:
        df["overnight_gap"] = df.groupby("Ticker").apply(
            lambda g: g["Open"] / g["Close"].shift(1) - 1
        ).reset_index(level=0, drop=True)
    else:
        df["overnight_gap"] = 0.0

    # RSI divergence: price near 52w high but RSI declining (momentum exhaustion).
    df["rsi_divergence"] = (df["distance_from_52w_high"] - 1.0) - (
        df.groupby("Ticker")["rsi"].transform(lambda x: x / x.rolling(20).mean()) - 1.0
    )

    # Replace raw level MAs with scale-free ratios.
    df["ma10_ma50_ratio"] = df["ma10"] / df["ma50"]
    df["ma20_ma50_ratio"] = df["ma20"] / df["ma50"]
    df["ma50_ma200_ratio"] = df["ma50"] / df["ma200"]

    xs_rank_pairs = [
        ("rsi", "xs_rsi_rank"),
        ("momentum", "xs_momentum_rank"),
        ("volume_ema", "xs_volume_ema_rank"),
        ("distance_from_52w_high", "xs_distance_from_52w_high_rank"),
        ("rs_vs_nifty", "xs_rs_vs_nifty_rank"),
        ("rs_vs_nifty_20d", "xs_rs_vs_nifty_20d_rank"),
        ("volume_spike", "xs_volume_spike_rank"),
        ("overnight_gap", "xs_overnight_gap_rank"),
        ("volume_surprise_z20", "xs_volume_surprise_rank"),
        ("return_over_atr14", "xs_return_over_atr14_rank"),
    ]
    for source_col, target_col in xs_rank_pairs:
        df = _append_rank_feature(df, source_col, target_col)

    rank_all_features = os.getenv("QUANT_XS_RANK_ALL_FEATURES", "0").strip() == "1"
    if rank_all_features:
        rank_cols = [
            "returns",
            "ma10_ma50_ratio",
            "ma20_ma50_ratio",
            "ma50_ma200_ratio",
            "volatility",
            "rsi",
            "momentum",
            "volume_ema",
            "distance_from_52w_high",
            "rs_vs_nifty",
            "rs_vs_nifty_20d",
            "volume_spike",
            "volume_surprise_z20",
            "xs_return_rank",
            "overnight_gap",
            "return_over_atr14",
        ]
        df = _cross_sectional_rank(df, rank_cols)

    # Horizon return target components (configurable for next-day or multi-day setups).
    future_stock_col = f"future_return_{target_horizon}"
    future_nifty_col = f"nifty_future_{target_horizon}"
    excess_col = f"excess_return_{target_horizon}"

    df[future_stock_col] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.shift(-target_horizon) / x - 1
    )

    # Horizon benchmark return from Nifty close.
    nifty_daily = (
        df[["Date", "nifty_close"]]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .copy()
    )
    nifty_daily[future_nifty_col] = (
        nifty_daily["nifty_close"].shift(-target_horizon) / nifty_daily["nifty_close"] - 1
    )
    df = df.merge(nifty_daily[["Date", future_nifty_col]], on="Date", how="left")

    # Target mode:
    # - alpha: beat benchmark by threshold
    # - absolute: absolute post-cost return above threshold
    if target_mode == "absolute":
        net_future_return = df[future_stock_col] - target_cost_bps
        df["target"] = np.where(
            net_future_return.notna(),
            (net_future_return > absolute_threshold).astype(int),
            np.nan,
        )
    else:
        df[excess_col] = df[future_stock_col] - df[future_nifty_col]
        df["target"] = np.where(
            df[excess_col].notna(),
            (df[excess_col] > alpha_threshold).astype(int),
            np.nan,
        )

    if forward_inference:
        skip_cols = {future_stock_col, future_nifty_col, excess_col, "target"}
        df = df.dropna(subset=[col for col in df.columns if col not in skip_cols])
    else:
        df = df.dropna()
    drop_cols = [future_stock_col, future_nifty_col]
    if excess_col in df.columns:
        drop_cols.append(excess_col)
    df.drop(columns=drop_cols, inplace=True)

    # 🔍 SANITY CHECK
    print("\nTarget Distribution:")
    print(df["target"].value_counts(normalize=True))

    return df