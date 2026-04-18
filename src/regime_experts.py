"""Phase-2 regime-aware expert gate module.

Trains a separate trade-acceptance MLP for each market regime
(BULLISH / BEARISH / NEUTRAL). Each expert is calibrated using only
rows whose ``regime_state`` matches that regime.

Regime mapping (matches src/features.py):
    BULLISH  -> regime_state == 1
    NEUTRAL  -> regime_state == 0
    BEARISH  -> regime_state == -1
"""
import numpy as np
import pandas as pd

from src.trade_acceptance import prepare_acceptance_frame, fit_trade_acceptance_model

# Maps regime name to numeric regime_state value used in features.py
_REGIME_VALUE_MAP: dict[str, int] = {
    "BULLISH": 1,
    "NEUTRAL": 0,
    "BEARISH": -1,
}


def fit_regime_experts(
    gate_fit_df: pd.DataFrame,
    cutoff_date,
    epochs: int = 25,
    min_rows_per_regime: int = 300,
    seed: int = 42,
) -> dict:
    """Train a separate acceptance MLP per market regime.

    Parameters
    ----------
    gate_fit_df : pd.DataFrame
        Full training frame (val-history + test rows) with ``regime_state``,
        ``future_alpha_*`` forward labels and all acceptance feature columns.
    cutoff_date :
        Date boundary — rows *before* this date are used for training,
        rows *on or after* are used only for inference.
    epochs : int
        Training epochs for each regime expert.
    min_rows_per_regime : int
        Minimum pre-cutoff rows required to train an expert. Regimes with fewer
        rows are skipped and fall back to the global Phase-1 threshold.
    seed : int
        Base RNG seed. Each regime gets ``seed + i`` to avoid correlation.

    Returns
    -------
    dict
        Keyed by regime name (``"BULLISH"``, ``"NEUTRAL"``, ``"BEARISH"``).
        Each value is a dict with:
        - ``trained``           (bool)
        - ``threshold``         (float - auto-calibrated acceptance cut-off)
        - ``probs_df``          (pd.DataFrame with date / ticker / regime_accept_prob)
        - ``train_rows``        (int)
        - ``val_rows``          (int)
        - ``val_loss``          (float or None)
        - ``reason``            (str)
    """
    results = {}

    for expert_idx, (regime_name, regime_val) in enumerate(sorted(_REGIME_VALUE_MAP.items())):
        regime_df = gate_fit_df[gate_fit_df["regime_state"] == regime_val].copy()

        if regime_df.empty:
            results[regime_name] = {
                "trained": False,
                "reason": "no_regime_rows",
                "threshold": 0.5,
                "probs_df": pd.DataFrame(columns=["date", "ticker", "regime_accept_prob"]),
                "train_rows": 0,
                "val_rows": 0,
                "val_loss": None,
            }
            continue

        result = fit_trade_acceptance_model(
            regime_df,
            cutoff_date=cutoff_date,
            epochs=epochs,
            min_rows=min_rows_per_regime,
            seed=seed + expert_idx,
        )

        if result.get("trained"):
            # Reconstruct the sorted frame the model used to align probs
            prepared = prepare_acceptance_frame(regime_df)
            prepared = prepared.sort_values("date").reset_index(drop=True)
            probs_df = prepared[["date", "ticker"]].copy()
            probs_df["regime_accept_prob"] = result["probs"].astype(np.float32)
        else:
            probs_df = pd.DataFrame(columns=["date", "ticker", "regime_accept_prob"])

        results[regime_name] = {
            "trained": result.get("trained", False),
            "reason": result.get("reason", ""),
            "threshold": result.get("threshold", 0.5),
            "probs_df": probs_df,
            "train_rows": result.get("train_rows", 0),
            "val_rows": result.get("val_rows", 0),
            "val_loss": result.get("val_loss"),
        }

    return results


def apply_regime_expert_probs(
    regime_experts: dict,
    pred_df: pd.DataFrame,
    fallback_threshold: float = 0.5,
) -> pd.DataFrame:
    """Add per-regime acceptance columns to ``pred_df``.

    For each row in ``pred_df``, the expert whose regime matches that row's
    ``regime_state`` is used to assign:

    - ``regime_accept_prob``       — acceptance probability from the matching expert
    - ``regime_expert_threshold``  — the matching expert's calibrated threshold

    Rows whose regime expert was not trained get ``regime_accept_prob = 1.0``
    (pass-through) and the fallback threshold.

    Parameters
    ----------
    regime_experts : dict
        Output of :func:`fit_regime_experts`.
    pred_df : pd.DataFrame
        Test-window prediction frame. Must contain ``regime_state``, ``date``,
        and ``ticker`` columns.
    fallback_threshold : float
        Threshold applied to rows whose regime expert was not trained.

    Returns
    -------
    pd.DataFrame
        Copy of ``pred_df`` with two extra columns.
    """
    out = pred_df.copy()
    out["regime_accept_prob"] = np.float32(1.0)
    out["regime_expert_threshold"] = np.float32(fallback_threshold)

    for regime_name, regime_val in _REGIME_VALUE_MAP.items():
        expert = regime_experts.get(regime_name, {"trained": False})
        regime_mask = out["regime_state"] == regime_val

        if not regime_mask.any():
            continue

        # Set per-regime threshold for all rows of this regime
        out.loc[regime_mask, "regime_expert_threshold"] = np.float32(
            expert.get("threshold", fallback_threshold)
        )

        probs_df: pd.DataFrame = expert.get(
            "probs_df", pd.DataFrame(columns=["date", "ticker", "regime_accept_prob"])
        )

        if expert.get("trained") and not probs_df.empty:
            probs_df = probs_df.copy()
            probs_df["date"] = pd.to_datetime(probs_df["date"])

            # Merge probs onto the regime rows of pred_df
            regime_rows = out.loc[regime_mask, ["date", "ticker"]].copy()
            regime_rows["date"] = pd.to_datetime(regime_rows["date"])
            merged = regime_rows.merge(probs_df, on=["date", "ticker"], how="left")
            out.loc[regime_mask, "regime_accept_prob"] = (
                merged["regime_accept_prob"].fillna(np.float32(1.0)).values
            )

    return out
