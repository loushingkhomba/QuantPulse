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

TARGET_GRID_OVERRIDE_ENV = "QUANT_OBJECTIVE_FREEZE_ALLOW_TARGET_GRID"
TARGET_GRID_OVERRIDE_KEYS: Set[str] = {
    "QUANT_TARGET_HORIZON_DAYS",
    "QUANT_HOLDING_DAYS",
    "QUANT_TARGET_COST_BPS",
    "QUANT_TARGET_ABS_THRESHOLD",
}


def _normalized_payload(params: Dict[str, str]) -> str:
    return json.dumps(dict(sorted(params.items())), separators=(",", ":"), sort_keys=True)


def compute_frozen_objective_hash(params: Dict[str, str]) -> str:
    payload = _normalized_payload(params)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


FROZEN_OBJECTIVE_V1_HASH = compute_frozen_objective_hash(FROZEN_OBJECTIVE_V1_PARAMS)


def _filter_keys(params: Dict[str, str], excluded_keys: Iterable[str]) -> Dict[str, str]:
    excluded = set(excluded_keys)
    return {k: v for k, v in params.items() if k not in excluded}


def get_current_freeze_values(excluded_keys: Iterable[str] = ()) -> Dict[str, str]:
    excluded = set(excluded_keys)
    return {k: os.getenv(k, "").strip() for k in FROZEN_OBJECTIVE_V1_PARAMS if k not in excluded}


def enforce_objective_freeze(strict: bool = True) -> Dict[str, object]:
    target_grid_override = os.getenv(TARGET_GRID_OVERRIDE_ENV, "0").strip() == "1"
    allowed_override_keys = TARGET_GRID_OVERRIDE_KEYS if target_grid_override else set()
    expected_values = _filter_keys(FROZEN_OBJECTIVE_V1_PARAMS, allowed_override_keys)
    current_values = get_current_freeze_values(excluded_keys=allowed_override_keys)
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
        "freeze_name": "objective_v1",
        "expected_hash": expected_hash,
        "current_hash": current_hash,
        "full_expected_hash": FROZEN_OBJECTIVE_V1_HASH,
        "ok": current_hash == expected_hash,
        "violations": violations,
        "values": current_values,
        "target_grid_override": target_grid_override,
        "allowed_override_keys": sorted(allowed_override_keys),
    }
