import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
POLICY_PATH = ROOT / "go_live_policy.json"
INCIDENTS_PATH = LOGS / "operational_incidents.csv"
FILLS_PATH = LOGS / "live_fills.csv"


def _campaign_paths(campaign_name: str) -> dict:
    if campaign_name == "default":
        return {
            "signals": LOGS / "paper_live_signals.csv",
            "state": LOGS / "go_live_gate_state.json",
            "report": LOGS / "go_live_weekly_report.json",
        }

    prefix = f"paper_live_{campaign_name}"
    return {
        "signals": LOGS / f"{prefix}_signals.csv",
        "state": LOGS / f"go_live_gate_state_{campaign_name}.json",
        "report": LOGS / f"go_live_weekly_report_{campaign_name}.json",
    }


def _load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _load_signals(signals_path: Path) -> pd.DataFrame:
    if not signals_path.exists():
        raise FileNotFoundError(f"Missing signals log: {signals_path}")

    df = pd.read_csv(signals_path)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    if "realized_return" not in df.columns:
        raise RuntimeError("signals log missing realized_return column")
    return df


def _weekly_date_set(model_daily: pd.Series) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(model_daily.index))
    return dates[-5:] if len(dates) >= 5 else dates


def _drawdown_from_returns(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity / np.maximum(peak, 1e-12)) - 1.0
    return float(abs(np.min(dd)))


def _max_consecutive_loss_days(daily_returns: np.ndarray) -> int:
    best = 0
    cur = 0
    for r in daily_returns:
        if r < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _compute_operational_incidents(weekly_dates: list[pd.Timestamp]) -> dict:
    if not INCIDENTS_PATH.exists() or len(weekly_dates) == 0:
        return {"count": 0, "details": []}

    incidents = pd.read_csv(INCIDENTS_PATH)
    if "date" not in incidents.columns:
        return {"count": 0, "details": []}

    incidents["date"] = pd.to_datetime(incidents["date"]).dt.normalize()
    week_set = set(pd.to_datetime(weekly_dates).normalize())
    week_df = incidents[incidents["date"].isin(week_set)].copy()

    details = week_df.to_dict(orient="records")
    return {"count": int(len(week_df)), "details": details}


def _compute_slippage_check(policy: dict, weekly_dates: list[pd.Timestamp]) -> dict:
    assumed = float(policy["slippage_monitor"]["assumed_slippage"])
    max_excess = float(policy["slippage_monitor"]["max_excess_slippage_before_alert"])

    out = {
        "assumed_slippage": assumed,
        "max_excess_slippage_before_alert": max_excess,
        "has_fill_data": False,
        "actual_slippage": None,
        "excess_slippage": None,
        "pass": True,
    }

    if not FILLS_PATH.exists() or len(weekly_dates) == 0:
        return out

    fills = pd.read_csv(FILLS_PATH)
    required_cols = {"signal_date", "expected_entry_price", "actual_entry_price"}
    if not required_cols.issubset(set(fills.columns)):
        return out

    fills["signal_date"] = pd.to_datetime(fills["signal_date"]).dt.normalize()
    week_set = set(pd.to_datetime(weekly_dates).normalize())
    week_fills = fills[fills["signal_date"].isin(week_set)].copy()
    if week_fills.empty:
        return out

    week_fills = week_fills[week_fills["expected_entry_price"] > 0]
    if week_fills.empty:
        return out

    week_fills["slip"] = (
        (week_fills["actual_entry_price"] - week_fills["expected_entry_price"])
        / week_fills["expected_entry_price"]
    )

    actual = float(week_fills["slip"].mean())
    excess = float(actual - assumed)

    out["has_fill_data"] = True
    out["actual_slippage"] = actual
    out["excess_slippage"] = excess
    out["pass"] = excess <= max_excess
    return out


def _compute_model_decay(policy: dict, model_daily: pd.Series, random_daily: pd.Series) -> dict:
    window = int(policy["model_decay"]["rolling_sharpe_window_days"])
    min_edge = float(policy["model_decay"]["min_sharpe_edge"])

    common_idx = model_daily.index.intersection(random_daily.index)
    if len(common_idx) < 3:
        return {
            "rolling_window_days": window,
            "rolling_sharpe_edge": None,
            "pass": True,
        }

    model_vals = model_daily.loc[common_idx].sort_index().values
    random_vals = random_daily.loc[common_idx].sort_index().values

    model_recent = model_vals[-window:]
    random_recent = random_vals[-window:]

    def _sharpe(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        return float(np.mean(x) / (np.std(x) + 1e-9) * np.sqrt(252))

    edge = _sharpe(model_recent) - _sharpe(random_recent)
    return {
        "rolling_window_days": window,
        "rolling_sharpe_edge": float(edge),
        "pass": edge >= min_edge,
    }


def _load_gate_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"consecutive_weekly_passes": 0, "last_report_date": None}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_gate_state(state_path: Path, state: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_weekly_report(campaign_name: str = "default") -> dict:
    campaign_name = campaign_name.strip().lower() or "default"
    paths = _campaign_paths(campaign_name)

    policy = _load_policy()
    phase_a = policy["phases"]["A_forward_paper"]
    phase_b = policy["phases"]["B_micro_live"]

    df = _load_signals(paths["signals"])
    selected = df[df["is_selected"] == True].copy()
    selected = selected.dropna(subset=["realized_return"])

    model_trades = selected[selected["source"] == "model"].copy()
    random_trades = selected[selected["source"] == "random"].copy()

    model_daily = model_trades.groupby("signal_date")["realized_return"].mean().sort_index()
    random_daily = random_trades.groupby("signal_date")["realized_return"].mean().sort_index()

    weekly_dates = _weekly_date_set(model_daily)
    if len(weekly_dates) == 0:
        raise RuntimeError("No matured model signal days available for weekly report.")

    model_week = model_daily[model_daily.index.isin(weekly_dates)]
    random_week = random_daily[random_daily.index.isin(weekly_dates)]

    model_weekly_equity = float(np.prod(1.0 + model_week.values)) if len(model_week) else 1.0
    random_weekly_equity = float(np.prod(1.0 + random_week.values)) if len(random_week) else 1.0
    model_vs_random_edge = float(model_weekly_equity - random_weekly_equity)

    model_week_trades = model_trades[model_trades["signal_date"].isin(weekly_dates)]
    model_hit_rate = float((model_week_trades["realized_return"] > 0).mean()) if len(model_week_trades) else 0.0

    max_weekly_drawdown = _drawdown_from_returns(model_week.values)
    max_consec_loss_days = _max_consecutive_loss_days(model_week.values)

    incidents = _compute_operational_incidents(weekly_dates)
    slippage_check = _compute_slippage_check(policy, weekly_dates)
    decay_check = _compute_model_decay(policy, model_daily, random_daily)

    expected_week_days = len(weekly_dates)
    actual_week_days = int(len(model_week))
    no_missing_run_days = (actual_week_days == expected_week_days)

    gates = {
        "min_model_weekly_equity": model_weekly_equity >= phase_a["weekly_gates"]["min_model_weekly_equity"],
        "min_model_vs_random_edge": model_vs_random_edge >= phase_a["weekly_gates"]["min_model_vs_random_edge"],
        "min_model_hit_rate": model_hit_rate >= phase_a["weekly_gates"]["min_model_hit_rate"],
        "max_weekly_drawdown": max_weekly_drawdown <= phase_a["weekly_gates"]["max_weekly_drawdown"],
        "no_missing_run_days": no_missing_run_days,
        "max_consecutive_loss_days": max_consec_loss_days <= phase_b["risk_limits"]["max_consecutive_loss_days"],
        "operational_incidents": incidents["count"] <= phase_b["required_operational_incidents"],
        "slippage_monitor": slippage_check["pass"],
        "model_decay": decay_check["pass"],
    }

    weekly_pass = all(gates.values())

    state = _load_gate_state(paths["state"])
    consecutive = int(state.get("consecutive_weekly_passes", 0))
    consecutive = consecutive + 1 if weekly_pass else 0

    state["consecutive_weekly_passes"] = consecutive
    state["last_report_date"] = datetime.utcnow().strftime("%Y-%m-%d")
    _save_gate_state(paths["state"], state)

    required = int(phase_a["required_consecutive_weekly_passes"])
    ready_for_phase_b = consecutive >= required

    report = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "campaign_name": campaign_name,
        "signals_path": str(paths["signals"]),
        "window_dates": [d.strftime("%Y-%m-%d") for d in weekly_dates],
        "metrics": {
            "model_weekly_equity": model_weekly_equity,
            "random_weekly_equity": random_weekly_equity,
            "model_vs_random_edge": model_vs_random_edge,
            "model_hit_rate": model_hit_rate,
            "max_weekly_drawdown": max_weekly_drawdown,
            "max_consecutive_loss_days": max_consec_loss_days,
            "matured_model_trades_in_week": int(len(model_week_trades)),
            "matured_model_days_in_week": int(len(model_week)),
        },
        "checks": {
            "gates": gates,
            "weekly_pass": weekly_pass,
            "consecutive_weekly_passes": consecutive,
            "required_for_phase_b": required,
            "ready_for_phase_b": ready_for_phase_b,
            "operational_incidents": incidents,
            "slippage": slippage_check,
            "model_decay": decay_check,
        },
        "actions": {
            "if_fail": "rollback_one_size_step_and_freeze_one_week",
            "if_model_decay_fail": policy["model_decay"]["action_on_fail"],
        },
    }

    LOGS.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    campaign_name = os.getenv("QUANT_PAPER_CAMPAIGN_NAME", "default").strip().lower() or "default"
    report = build_weekly_report(campaign_name=campaign_name)
    paths = _campaign_paths(campaign_name)
    print("Weekly go-live report generated")
    print(f"Campaign: {campaign_name}")
    print(f"Path: {paths['report']}")
    print(f"Weekly pass: {report['checks']['weekly_pass']}")
    print(
        "Consecutive weekly passes: "
        f"{report['checks']['consecutive_weekly_passes']}/"
        f"{report['checks']['required_for_phase_b']}"
    )
    print(f"Ready for Phase B: {report['checks']['ready_for_phase_b']}")


if __name__ == "__main__":
    main()
