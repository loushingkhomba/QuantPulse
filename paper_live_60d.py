import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.zerodha_ohlc_loader import download_data
from src.features import create_features
from src.model import QuantPulse, QuantPulseMLP, QuantPulseSimple


FEATURE_COLUMNS = [
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
    "regime_state",
]


@dataclass
class RunConfig:
    top_k: int
    sequence_length: int
    holding_days: int
    campaign_days: int
    logs_dir: Path
    models_dir: Path
    model_type: str
    ensemble_seeds: list[int]
    random_seed: int
    campaign_name: str
    campaign_start_date: str
    max_data_lag_days: int
    allow_stale_data: bool
    enable_train_parity_gates: bool


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ensemble_seeds() -> list[int]:
    raw = os.getenv("QUANT_ENSEMBLE_SEEDS", "42,123,777")
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return seeds or [42, 123, 777]


def sanitize_campaign_name(name: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in ["-", "_"]) else "_" for ch in name.strip().lower())
    return safe or "default"


def artifact_paths(cfg: RunConfig) -> tuple[Path, Path, Path]:
    if cfg.campaign_name == "default":
        return (
            cfg.logs_dir / "paper_live_signals.csv",
            cfg.logs_dir / "paper_live_summary.json",
            cfg.logs_dir / "paper_live_state.json",
        )

    prefix = f"paper_live_{cfg.campaign_name}"
    return (
        cfg.logs_dir / f"{prefix}_signals.csv",
        cfg.logs_dir / f"{prefix}_summary.json",
        cfg.logs_dir / f"{prefix}_state.json",
    )


def normalize_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value).tz_localize(None).normalize()


def assert_data_freshness(available_dates: list[pd.Timestamp], cfg: RunConfig) -> None:
    # Convert to timezone-naive for comparison
    latest_market_date = pd.Timestamp(max(available_dates)).tz_localize(None).normalize()
    today = pd.Timestamp(datetime.utcnow().date())
    lag_days = int((today - latest_market_date).days)

    print(f"Latest market date in data: {latest_market_date.strftime('%Y-%m-%d')}")
    print(f"Data lag days vs today: {lag_days}")

    if lag_days > cfg.max_data_lag_days and not cfg.allow_stale_data:
        raise RuntimeError(
            "Market data is stale for forward paper mode. "
            f"Latest date={latest_market_date.strftime('%Y-%m-%d')}, lag_days={lag_days}, "
            f"max_allowed={cfg.max_data_lag_days}. "
            "Use fresher data feed or pass --allow-stale-data to bypass intentionally."
        )


def build_model(model_type: str, input_size: int) -> torch.nn.Module:
    if model_type == "simple":
        return QuantPulseSimple(input_size=input_size)
    if model_type == "mlp":
        return QuantPulseMLP(input_size=input_size)
    if model_type == "lstm":
        return QuantPulse(input_size=input_size)
    raise ValueError(f"Unsupported model type: {model_type}")


def load_model_confidence(
    df_features: pd.DataFrame,
    run_date: pd.Timestamp,
    cfg: RunConfig,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for ticker, ticker_df in df_features.groupby("Ticker"):
        ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
        ticker_df = ticker_df[ticker_df["Date"] <= run_date]
        if len(ticker_df) < cfg.sequence_length + 30:
            continue

        x_raw = ticker_df[FEATURE_COLUMNS].values
        scaler = StandardScaler()
        scaler.fit(x_raw)
        x_scaled = scaler.transform(x_raw)

        seq = x_scaled[-cfg.sequence_length :]
        latest_price = float(ticker_df["Close"].iloc[-1])
        latest_row = ticker_df.iloc[-1]
        rows.append(
            (
                ticker,
                seq,
                latest_price,
                float(latest_row.get("regime_state", 0.0)),
                float(latest_row.get("volatility_regime", 1.0)),
                float(latest_row.get("nifty_drawdown_63d", 0.0)),
                float(latest_row.get("trend_strength", 0.0)),
            )
        )

    if not rows:
        return pd.DataFrame(columns=["signal_date", "ticker", "confidence", "price_signal"])

    tickers = [r[0] for r in rows]
    prices = [r[2] for r in rows]
    regime_state = [r[3] for r in rows]
    volatility_regime = [r[4] for r in rows]
    nifty_drawdown_63d = [r[5] for r in rows]
    trend_strength = [r[6] for r in rows]
    x_batch = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32).to(device)

    all_probs = []
    for seed in cfg.ensemble_seeds:
        model = build_model(cfg.model_type, input_size=len(FEATURE_COLUMNS)).to(device)
        model_path = cfg.models_dir / f"quantpulse_model_seed{seed}.pth"
        if not model_path.exists() and len(cfg.ensemble_seeds) == 1:
            model_path = cfg.models_dir / "quantpulse_model.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    confidence = np.mean(np.stack(all_probs, axis=0), axis=0)
    out = pd.DataFrame(
        {
            "signal_date": run_date,
            "ticker": tickers,
            "confidence": confidence,
            "price_signal": prices,
            "regime_state": regime_state,
            "volatility_regime": volatility_regime,
            "nifty_drawdown_63d": nifty_drawdown_63d,
            "trend_strength": trend_strength,
        }
    )
    return out


def make_daily_signals(df_conf: pd.DataFrame, cfg: RunConfig, run_date: pd.Timestamp) -> pd.DataFrame:
    if df_conf.empty:
        return df_conf

    day_df = df_conf.sort_values("confidence", ascending=False).reset_index(drop=True)
    day_df["source"] = "model"

    selected_tickers: set[str] = set()
    gate_reason = "top_k"
    exposure_scale = 1.0
    signal_spread = np.nan
    is_bad_regime = False

    if cfg.enable_train_parity_gates:
        bad_volatility_cutoff = float(os.getenv("QUANT_BAD_VOLATILITY_CUTOFF", "1.25"))
        bad_drawdown_cutoff = float(os.getenv("QUANT_BAD_DRAWDOWN_CUTOFF", "-0.08"))
        trend_strength_cutoff = float(os.getenv("QUANT_TREND_STRENGTH_CUTOFF", "0.00"))

        rank_threshold_bad = float(os.getenv("QUANT_RANK_THRESHOLD_BAD", "0.68"))
        rank_threshold_neutral = float(os.getenv("QUANT_RANK_THRESHOLD_NEUTRAL", "0.60"))
        rank_threshold_trending = float(os.getenv("QUANT_RANK_THRESHOLD_TRENDING", "0.58"))

        spread_threshold_bad = float(os.getenv("QUANT_SIGNAL_SPREAD_BAD", "0.015"))
        spread_threshold_neutral = float(os.getenv("QUANT_SIGNAL_SPREAD_NEUTRAL", "0.010"))
        spread_threshold_trending = float(os.getenv("QUANT_SIGNAL_SPREAD_TRENDING", "0.008"))

        fallback_conf_threshold_bad = float(os.getenv("QUANT_FALLBACK_CONF_THRESHOLD_BAD", "0.52"))
        fallback_reduce_factor_bad = float(os.getenv("QUANT_FALLBACK_REDUCE_FACTOR_BAD", "0.50"))
        regime_exposure_scale_bad = float(os.getenv("QUANT_REGIME_EXPOSURE_SCALE_BAD", "0.60"))

        day_regime_state = int(np.nan_to_num(day_df["regime_state"].median(), nan=0.0))
        day_volatility_regime = float(np.nan_to_num(day_df["volatility_regime"].median(), nan=1.0))
        day_drawdown_state = float(np.nan_to_num(day_df["nifty_drawdown_63d"].median(), nan=0.0))
        day_trend_strength = float(np.nan_to_num(day_df["trend_strength"].median(), nan=0.0))

        is_bad_regime = (
            (day_regime_state < 0)
            or (day_volatility_regime > bad_volatility_cutoff)
            or (day_drawdown_state < bad_drawdown_cutoff)
        )
        is_trending_regime = (not is_bad_regime) and (day_trend_strength > trend_strength_cutoff)

        if is_bad_regime:
            day_min_rank_threshold = rank_threshold_bad
            day_min_signal_spread = spread_threshold_bad
        elif is_trending_regime:
            day_min_rank_threshold = rank_threshold_trending
            day_min_signal_spread = spread_threshold_trending
        else:
            day_min_rank_threshold = rank_threshold_neutral
            day_min_signal_spread = spread_threshold_neutral

        day_df["rank_pct"] = day_df["confidence"].rank(pct=True, method="first")
        candidates = day_df[day_df["rank_pct"] >= day_min_rank_threshold]
        top = candidates.sort_values("confidence", ascending=False).head(cfg.top_k)

        if len(top) == 0:
            gate_reason = "no_candidates"
        else:
            signal_spread = float(top["confidence"].iloc[0] - top["confidence"].iloc[-1])
            if signal_spread < day_min_signal_spread:
                gate_reason = "low_spread"
            else:
                day_confidence = float(top["confidence"].max())
                fallback_scale = 1.0
                regime_scale = 1.0
                if is_bad_regime and day_confidence < fallback_conf_threshold_bad:
                    fallback_scale = fallback_reduce_factor_bad
                if is_bad_regime:
                    regime_scale = regime_exposure_scale_bad
                exposure_scale = max(0.0, float(fallback_scale * regime_scale))
                effective_top_k = int(np.floor(cfg.top_k * exposure_scale + 1e-9))
                if effective_top_k > 0:
                    selected_tickers = set(top.head(effective_top_k)["ticker"].tolist())
                    gate_reason = "selected"
                else:
                    gate_reason = "zero_exposure"
    else:
        selected_tickers = set(day_df.head(cfg.top_k)["ticker"].tolist())

    day_df["rank"] = np.arange(1, len(day_df) + 1)
    day_df["is_selected"] = day_df["ticker"].isin(selected_tickers)
    day_df["gate_reason"] = gate_reason
    day_df["exposure_scale"] = exposure_scale
    day_df["signal_spread"] = signal_spread
    day_df["is_bad_regime"] = bool(is_bad_regime)

    rng = np.random.RandomState(cfg.random_seed + int(run_date.strftime("%Y%m%d")))
    random_df = day_df[["signal_date", "ticker", "price_signal"]].copy()
    random_df["confidence"] = rng.permutation(day_df["confidence"].values)
    random_df = random_df.sort_values("confidence", ascending=False).reset_index(drop=True)
    random_df["rank"] = np.arange(1, len(random_df) + 1)
    random_df["source"] = "random"
    random_df["is_selected"] = random_df["rank"] <= cfg.top_k

    random_df["gate_reason"] = "random_top_k"
    random_df["exposure_scale"] = 1.0
    random_df["signal_spread"] = np.nan
    random_df["is_bad_regime"] = False

    export_cols = [
        "signal_date",
        "ticker",
        "source",
        "confidence",
        "rank",
        "is_selected",
        "price_signal",
        "gate_reason",
        "exposure_scale",
        "signal_spread",
        "is_bad_regime",
    ]
    both = pd.concat(
        [
            day_df[export_cols],
            random_df[export_cols],
        ],
        ignore_index=True,
    )
    return both


def get_exit_price_map(df_features: pd.DataFrame, holding_days: int) -> dict[tuple[pd.Timestamp, str], tuple[pd.Timestamp, float]]:
    out = {}
    for ticker, ticker_df in df_features.groupby("Ticker"):
        ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
        dates = pd.to_datetime(ticker_df["Date"]).tolist()
        prices = ticker_df["Close"].astype(float).tolist()
        for i in range(len(dates) - holding_days):
            out[(pd.Timestamp(dates[i]), ticker)] = (pd.Timestamp(dates[i + holding_days]), float(prices[i + holding_days]))
    return out


def update_realized_returns(signal_df: pd.DataFrame, exit_map: dict, holding_days: int) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df

    df = signal_df.copy()
    if "exit_date" not in df.columns:
        df["exit_date"] = pd.NaT
    if "price_exit" not in df.columns:
        df["price_exit"] = np.nan
    if "realized_return" not in df.columns:
        df["realized_return"] = np.nan

    for idx, row in df.iterrows():
        if pd.notna(row["realized_return"]):
            continue
        key = (pd.Timestamp(row["signal_date"]), row["ticker"])
        if key not in exit_map:
            continue
        exit_date, exit_price = exit_map[key]
        entry_price = float(row["price_signal"])
        if entry_price <= 0:
            continue
        df.at[idx, "exit_date"] = exit_date
        df.at[idx, "price_exit"] = exit_price
        df.at[idx, "realized_return"] = (exit_price - entry_price) / entry_price

    return df


def load_existing_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "signal_date",
            "ticker",
            "source",
            "confidence",
            "rank",
            "is_selected",
            "price_signal",
            "exit_date",
            "price_exit",
            "realized_return",
        ])
    df = pd.read_csv(path)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    if "exit_date" in df.columns:
        df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    return df


def resolve_target_dates(
    available_dates: list[pd.Timestamp],
    replay_last_n_days: int,
    forward_date: str,
    backfill_start: str,
    backfill_end: str,
) -> tuple[list[pd.Timestamp], str]:
    if backfill_start or backfill_end:
        start_ts = normalize_timestamp(backfill_start) or pd.Timestamp(available_dates[0]).normalize()
        end_ts = normalize_timestamp(backfill_end) or pd.Timestamp(available_dates[-1]).normalize()
        target_dates = [pd.Timestamp(value).normalize() for value in available_dates if start_ts <= pd.Timestamp(value).normalize() <= end_ts]
        if not target_dates:
            raise RuntimeError(f"No available market dates found for backfill window {start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d}.")
        return target_dates, f"backfill {start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d}"

    if forward_date:
        target_ts = normalize_timestamp(forward_date)
        if target_ts is None:
            raise ValueError("--date requires a YYYY-MM-DD value.")
        target_dates = [pd.Timestamp(value).normalize() for value in available_dates if pd.Timestamp(value).normalize() == target_ts]
        if not target_dates:
            raise RuntimeError(f"Requested forward date {target_ts:%Y-%m-%d} is not present in the available market data.")
        return target_dates, f"forward {target_ts:%Y-%m-%d}"

    if replay_last_n_days > 0:
        target_dates = [pd.Timestamp(value).normalize() for value in available_dates[-replay_last_n_days:]]
        return target_dates, f"replay last {replay_last_n_days} days"

    target_dates = [pd.Timestamp(available_dates[-1]).normalize()]
    return target_dates, f"forward {target_dates[0]:%Y-%m-%d}"


def merge_day(existing: pd.DataFrame, today: pd.DataFrame, run_date: pd.Timestamp) -> pd.DataFrame:
    if today.empty:
        return existing
    keep = ~((existing["signal_date"] == run_date) & (existing["source"].isin(["model", "random"])))
    merged = pd.concat([existing[keep], today], ignore_index=True)
    merged = merged.sort_values(["signal_date", "source", "rank", "ticker"]).reset_index(drop=True)
    return merged


def build_summary(df: pd.DataFrame, cfg: RunConfig, run_dates: list[pd.Timestamp]) -> dict:
    selected = df[df["is_selected"]].copy()
    matured = selected.dropna(subset=["realized_return"]).copy()

    summary = {
        "campaign_days_target": cfg.campaign_days,
        "signal_days_recorded": int(len(run_dates)),
        "matured_selected_trades": int(len(matured)),
        "holding_days": cfg.holding_days,
        "top_k": cfg.top_k,
        "by_source": {},
    }

    for source in ["model", "random"]:
        part = matured[matured["source"] == source]
        if len(part) == 0:
            summary["by_source"][source] = {
                "trades": 0,
                "avg_return": 0.0,
                "hit_rate": 0.0,
                "compounded_equity": 1.0,
            }
            continue

        grouped = part.groupby("signal_date")["realized_return"].mean().sort_index()
        equity = float(np.prod(1.0 + grouped.values))
        summary["by_source"][source] = {
            "trades": int(len(part)),
            "avg_return": float(part["realized_return"].mean()),
            "hit_rate": float((part["realized_return"] > 0).mean()),
            "compounded_equity": equity,
            "days_with_matured_signals": int(len(grouped)),
        }

    model_equity = summary["by_source"]["model"]["compounded_equity"]
    random_equity = summary["by_source"]["random"]["compounded_equity"]
    summary["equity_edge_model_minus_random"] = float(model_equity - random_equity)
    summary["campaign_complete"] = len(run_dates) >= cfg.campaign_days
    return summary


def save_state(state_path: Path, run_dates: list[pd.Timestamp], cfg: RunConfig) -> None:
    state = {
        "campaign_days": cfg.campaign_days,
        "holding_days": cfg.holding_days,
        "top_k": cfg.top_k,
        "signal_days_recorded": len(run_dates),
        "run_dates": [d.strftime("%Y-%m-%d") for d in sorted(run_dates)],
        "campaign_complete": len(run_dates) >= cfg.campaign_days,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_campaign(
    cfg: RunConfig,
    replay_last_n_days: int = 0,
    forward_date: str = "",
    backfill_start: str = "",
    backfill_end: str = "",
) -> None:
    set_global_seed(int(os.getenv("QUANT_SEED", "42")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Downloading data...")
    raw = download_data()
    print("Creating features...")
    os.environ["QUANT_FORWARD_INFERENCE"] = "1"
    df = create_features(raw)
    df["Date"] = pd.to_datetime(df["Date"])

    available_dates = sorted(pd.to_datetime(df["Date"].unique()))
    if not available_dates:
        raise RuntimeError("No market dates available after feature engineering.")

    signals_path, summary_path, state_path = artifact_paths(cfg)

    if replay_last_n_days <= 0:
        assert_data_freshness(available_dates, cfg)

    existing = load_existing_signals(signals_path)
    if cfg.campaign_start_date:
        campaign_start_ts = normalize_timestamp(cfg.campaign_start_date)
        existing["signal_date"] = pd.to_datetime(existing["signal_date"]).apply(
            lambda value: value.tz_localize(None).normalize() if getattr(value, "tzinfo", None) is not None else pd.Timestamp(value).normalize()
        )
        existing = existing[existing["signal_date"] >= campaign_start_ts].reset_index(drop=True)

    if backfill_start or backfill_end:
        backfill_start_ts = normalize_timestamp(backfill_start) or pd.Timestamp(available_dates[0]).normalize()
        backfill_end_ts = normalize_timestamp(backfill_end) or pd.Timestamp(available_dates[-1]).normalize()
        existing["signal_date"] = pd.to_datetime(existing["signal_date"]).apply(
            lambda value: value.tz_localize(None).normalize() if getattr(value, "tzinfo", None) is not None else pd.Timestamp(value).normalize()
        )
        existing = existing[existing["signal_date"] < backfill_start_ts].reset_index(drop=True)

    run_dates_set = set(pd.to_datetime(existing["signal_date"]).unique())

    target_dates, mode_label = resolve_target_dates(
        available_dates,
        replay_last_n_days=replay_last_n_days,
        forward_date=forward_date,
        backfill_start=backfill_start,
        backfill_end=backfill_end,
    )

    if mode_label.startswith("backfill"):
        print(f"Backfilling signals: {mode_label.split(' ', 1)[1]}")
        print(f"Signal days backfilled: {len(target_dates)}")
    elif mode_label.startswith("forward"):
        print(f"Forward signal date: {mode_label.split(' ', 1)[1]}")

    for run_date in target_dates:
        conf_df = load_model_confidence(df, run_date, cfg, device)
        daily = make_daily_signals(conf_df, cfg, run_date)
        existing = merge_day(existing, daily, run_date)
        run_dates_set.add(pd.Timestamp(run_date))

    exit_map = get_exit_price_map(df, cfg.holding_days)
    existing = update_realized_returns(existing, exit_map, cfg.holding_days)

    existing = existing.sort_values(["signal_date", "source", "rank", "ticker"]).reset_index(drop=True)
    existing.to_csv(signals_path, index=False)

    run_dates = sorted(run_dates_set)
    summary = build_summary(existing, cfg, run_dates)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_state(state_path, run_dates, cfg)

    latest_day = max(target_dates).strftime("%Y-%m-%d") if target_dates else "n/a"
    print("\nPaper campaign updated")
    print(f"Latest processed signal date: {latest_day}")
    print(f"Signal days recorded: {summary['signal_days_recorded']}/{cfg.campaign_days}")
    print(f"Matured selected trades: {summary['matured_selected_trades']}")
    print(f"Model compounded equity: {summary['by_source']['model']['compounded_equity']:.4f}")
    print(f"Random compounded equity: {summary['by_source']['random']['compounded_equity']:.4f}")
    print(f"Model equity edge: {summary['equity_edge_model_minus_random']:.4f}")
    print(f"Campaign complete: {summary['campaign_complete']}")
    print(f"Signals log: {signals_path}")
    print(f"Summary: {summary_path}")
    print(f"State: {state_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and track a 60-day live-paper campaign.")
    parser.add_argument("--campaign-days", type=int, default=int(os.getenv("QUANT_PAPER_CAMPAIGN_DAYS", "60")))
    parser.add_argument("--holding-days", type=int, default=int(os.getenv("QUANT_HOLDING_DAYS", "5")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("QUANT_TOP_K", "3")))
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--forward", action="store_true", help="Run a single forward signal for the requested date or latest available date.")
    parser.add_argument("--backfill", action="store_true", help="Replay a date range and backfill signals into the campaign log.")
    parser.add_argument("--date", type=str, default="", help="Forward signal date (YYYY-MM-DD).")
    parser.add_argument("--start", type=str, default="", help="Backfill start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="", help="Backfill end date (YYYY-MM-DD).")
    parser.add_argument("--replay-last-n-days", type=int, default=0)
    parser.add_argument("--campaign-name", type=str, default=os.getenv("QUANT_PAPER_CAMPAIGN_NAME", "default"))
    parser.add_argument("--campaign-start-date", type=str, default=os.getenv("QUANT_PAPER_CAMPAIGN_START_DATE", ""))
    parser.add_argument("--max-data-lag-days", type=int, default=int(os.getenv("QUANT_MAX_DATA_LAG_DAYS", "3")))
    parser.add_argument("--allow-stale-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forward and args.backfill:
        raise ValueError("Choose either --forward or --backfill, not both.")
    cfg = RunConfig(
        top_k=max(1, args.top_k),
        sequence_length=max(10, args.sequence_length),
        holding_days=max(1, args.holding_days),
        campaign_days=max(10, args.campaign_days),
        logs_dir=Path("logs"),
        models_dir=Path("models"),
        model_type=os.getenv("QUANT_MODEL_TYPE", "simple").strip().lower(),
        ensemble_seeds=parse_ensemble_seeds(),
        random_seed=int(os.getenv("QUANT_RANDOM_BASELINE_SEED", "42")),
        campaign_name=sanitize_campaign_name(args.campaign_name),
        campaign_start_date=args.campaign_start_date.strip(),
        max_data_lag_days=max(0, args.max_data_lag_days),
        allow_stale_data=bool(args.allow_stale_data),
        enable_train_parity_gates=os.getenv("QUANT_PAPER_ENABLE_TRAIN_GATES", "0").strip() == "1",
    )

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    forward_date = args.date.strip() if (args.forward or args.date.strip()) else ""
    backfill_start = args.start.strip() if args.backfill else ""
    backfill_end = args.end.strip() if args.backfill else ""

    if args.backfill and (not backfill_start or not backfill_end):
        raise ValueError("--backfill requires both --start and --end.")

    run_campaign(
        cfg,
        replay_last_n_days=max(0, args.replay_last_n_days),
        forward_date=forward_date,
        backfill_start=backfill_start,
        backfill_end=backfill_end,
    )


if __name__ == "__main__":
    main()
