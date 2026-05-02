import hashlib
import json
import os
from typing import Dict, Iterable, Set

# Step-1 objective freeze. Any drift should fail fast.
FROZEN_OBJECTIVE_V1_PARAMS: Dict[str, str] = {
    "QUANT_OBJECTIVE_MODE": "ranking",
    "QUANT_TARGET_MODE": "absolute",
    "QUANT_TARGET_HORIZON_DAYS": "1",
    "QUANT_HOLDING_DAYS": "1",
    "QUANT_TARGET_COST_BPS": "7",
    "QUANT_TARGET_ABS_THRESHOLD": "0.001",
    "QUANT_RANK_LOSS_WEIGHT": "1.0",
    "QUANT_CLASSIFICATION_LOSS_WEIGHT": "0.25",
    "QUANT_FP_COST_MULTIPLIER": "0.0",
    "QUANT_SIMPLE_HIDDEN_SIZE": "64",
    "QUANT_ENSEMBLE_SEEDS": "42",
    "QUANT_REGIME_SAFETY_STRICT_PCT": "0.85",
    "QUANT_STRICT_SIGNAL_SPREAD_BAD": "0.012",
    "QUANT_STRICT_SIGNAL_SPREAD_NEUTRAL": "0.010",
    "QUANT_STRICT_SIGNAL_SPREAD_TRENDING": "0.008",
    "QUANT_KELLY_BREAKEVEN_STRICT": "0.53",
}

# V2-Absolute blueprint freeze (Mon-entry / Wed-exit, ATR filter, market gate).
FROZEN_OBJECTIVE_V2_PARAMS: Dict[str, str] = {
    "QUANT_OBJECTIVE_MODE": "ranking",
    "QUANT_TARGET_MODE": "absolute",
    "QUANT_TARGET_HORIZON_DAYS": "3",
    "QUANT_HOLDING_DAYS": "3",
    "QUANT_TARGET_ABS_THRESHOLD": "0.003",
    "QUANT_SIMPLE_HIDDEN_SIZE": "64",
    "QUANT_TOP_K": "3",
    "QUANT_ENSEMBLE_SEEDS": "42",
    "QUANT_ATR_RATIO_MAX": "0.025",
    "QUANT_PAPER_ENABLE_MARKET_GATE": "1",
}

TARGET_GRID_OVERRIDE_ENV = "QUANT_OBJECTIVE_FREEZE_ALLOW_TARGET_GRID"
TARGET_GRID_OVERRIDE_KEYS: Set[str] = {
    "QUANT_TARGET_HORIZON_DAYS",
    "QUANT_HOLDING_DAYS",
    "QUANT_TARGET_COST_BPS",
    "QUANT_TARGET_ABS_THRESHOLD",
}
FP_COST_OVERRIDE_ENV = "QUANT_OBJECTIVE_FREEZE_ALLOW_FP_COST"
FP_COST_OVERRIDE_KEYS: Set[str] = {
    "QUANT_FP_COST_MULTIPLIER",
}

# V3 Robust Nifty50 freeze (institutional edge monitor + circuit breaker stack).
FROZEN_OBJECTIVE_V3ROBUSTNIFTY50_PARAMS: Dict[str, str] = {
    "QUANT_OBJECTIVE_MODE": "ranking",
    "QUANT_TARGET_MODE": "absolute",
    "QUANT_TARGET_HORIZON_DAYS": "3",
    "QUANT_HOLDING_DAYS": "3",
    "QUANT_TARGET_COST_BPS": "7",
    "QUANT_TARGET_ABS_THRESHOLD": "0.003",
    "QUANT_SIMPLE_HIDDEN_SIZE": "64",
    "QUANT_ENSEMBLE_SEEDS": "42",
    "QUANT_TOP_K": "1",
    "QUANT_SIGNAL_INVERSION_MODE": "auto",
    "QUANT_AUTO_INVERT_SIGNAL": "1",
    "QUANT_FP_COST_MULTIPLIER": "0.0",
    "QUANT_FP_COST_RANDOMIZATION_ENABLED": "0",
    "QUANT_CASH_MODE_ENABLED": "1",
    "QUANT_CASH_MODE_MIN_IC": "0.0",
    "QUANT_CASH_MODE_MIN_SHARPE": "-0.5",
    "QUANT_CIRCUIT_BREAKER_ENABLED": "1",
    "QUANT_CIRCUIT_SHARPE_TIER1": "-1.0",
    "QUANT_CIRCUIT_SHARPE_TIER2": "-2.5",
    "QUANT_CIRCUIT_RISK_TIER1": "0.35",
    "QUANT_CIRCUIT_RISK_TIER2": "0.0",
    "QUANT_ROLLING_STATE_WINDOW": "20",
    "QUANT_TRAIN_EPOCHS": "15",
    "QUANT_TRAIN_PATIENCE": "6",
}

_FREEZE_PRESETS: Dict[str, Dict[str, str]] = {
    "v1": FROZEN_OBJECTIVE_V1_PARAMS,
    "v2_absolute": FROZEN_OBJECTIVE_V2_PARAMS,
    "v3robustnifty50": FROZEN_OBJECTIVE_V3ROBUSTNIFTY50_PARAMS,
}


def _normalized_payload(params: Dict[str, str]) -> str:
    return json.dumps(dict(sorted(params.items())), separators=(",", ":"), sort_keys=True)


def compute_frozen_objective_hash(params: Dict[str, str]) -> str:
    payload = _normalized_payload(params)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


FROZEN_OBJECTIVE_V1_HASH = compute_frozen_objective_hash(FROZEN_OBJECTIVE_V1_PARAMS)
FROZEN_OBJECTIVE_V2_HASH = compute_frozen_objective_hash(FROZEN_OBJECTIVE_V2_PARAMS)
FROZEN_OBJECTIVE_V3ROBUSTNIFTY50_HASH = compute_frozen_objective_hash(FROZEN_OBJECTIVE_V3ROBUSTNIFTY50_PARAMS)


def _filter_keys(params: Dict[str, str], excluded_keys: Iterable[str]) -> Dict[str, str]:
    excluded = set(excluded_keys)
    return {k: v for k, v in params.items() if k not in excluded}


def get_frozen_objective_params(preset: str = "v1") -> Dict[str, str]:
    normalized = str(preset or "v1").strip().lower()
    if normalized not in _FREEZE_PRESETS:
        raise ValueError(
            f"Unknown objective freeze preset '{preset}'. "
            f"Allowed: {sorted(_FREEZE_PRESETS.keys())}"
        )
    return dict(_FREEZE_PRESETS[normalized])


def get_current_freeze_values(
    expected_params: Dict[str, str],
    excluded_keys: Iterable[str] = (),
) -> Dict[str, str]:
    excluded = set(excluded_keys)
    return {k: os.getenv(k, "").strip() for k in expected_params if k not in excluded}


def enforce_objective_freeze(strict: bool = True, preset: str = "v1") -> Dict[str, object]:
    normalized_preset = str(preset or "v1").strip().lower()
    expected_preset_values = get_frozen_objective_params(normalized_preset)
    target_grid_override = os.getenv(TARGET_GRID_OVERRIDE_ENV, "0").strip() == "1"
    fp_cost_override = os.getenv(FP_COST_OVERRIDE_ENV, "0").strip() == "1"
    allowed_override_keys = set()
    if target_grid_override:
        allowed_override_keys.update(TARGET_GRID_OVERRIDE_KEYS)
    if fp_cost_override:
        allowed_override_keys.update(FP_COST_OVERRIDE_KEYS)
    expected_values = _filter_keys(expected_preset_values, allowed_override_keys)
    current_values = get_current_freeze_values(expected_values, excluded_keys=allowed_override_keys)
    expected_hash = compute_frozen_objective_hash(expected_values)
    current_hash = compute_frozen_objective_hash(current_values)
    violations = {
        key: {"expected": expected, "current": current_values.get(key, "")}
        for key, expected in expected_values.items()
        if current_values.get(key, "") != expected
    }
    if strict and current_hash != expected_hash:
        raise ValueError(
            "FREEZE VIOLATION: Quantitative objective drift detected. "
            f"expected_hash={expected_hash}, current_hash={current_hash}, violations={violations}"
        )
    return {
        "freeze_name": f"objective_{normalized_preset}",
        "expected_hash": expected_hash,
        "current_hash": current_hash,
        "full_expected_hash": compute_frozen_objective_hash(expected_preset_values),
        "ok": current_hash == expected_hash,
        "violations": violations,
        "values": current_values,
        "preset": normalized_preset,
        "target_grid_override": target_grid_override,
        "fp_cost_override": fp_cost_override,
        "allowed_override_keys": sorted(allowed_override_keys),
    }


def enforce_objective_v2_freeze(strict: bool = True) -> Dict[str, object]:
    """Enforce V2-Absolute blueprint parameter freeze."""
    current_values = {k: os.getenv(k, "").strip() for k in FROZEN_OBJECTIVE_V2_PARAMS}
    expected_hash = FROZEN_OBJECTIVE_V2_HASH
    current_hash = compute_frozen_objective_hash(current_values)
    violations = {
        key: {"expected": expected, "current": current_values.get(key, "")}
        for key, expected in FROZEN_OBJECTIVE_V2_PARAMS.items()
        if current_values.get(key, "") != expected
    }
    if strict and violations:
        raise ValueError(
            "FREEZE VIOLATION: V2-Absolute parameter drift detected. "
            f"expected_hash={expected_hash}, violations={violations}"
        )
    return {
        "freeze_name": "objective_v2_absolute",
        "expected_hash": expected_hash,
        "current_hash": current_hash,
        "ok": not violations,
        "violations": violations,
        "values": current_values,
    }
