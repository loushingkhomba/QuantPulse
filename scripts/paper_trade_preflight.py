import os
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import FROZEN_OBJECTIVE_V1_PARAMS, enforce_objective_freeze


def run_preflight():
    for key, value in FROZEN_OBJECTIVE_V1_PARAMS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("QUANT_OBJECTIVE_FREEZE_ALLOW_TARGET_GRID", "1")

    freeze_meta = enforce_objective_freeze(strict=True)

    model_path = Path("models/quantpulse_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "timestamp": datetime.now().isoformat(),
        "freeze": {
            "name": freeze_meta["freeze_name"],
            "hash": freeze_meta["current_hash"],
            "ok": freeze_meta["ok"],
        },
        "model": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "last_modified": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        },
        "paper_env": {
            "mode": os.getenv("QUANT_PAPER_TRADING_MODE", "simulation"),
            "data_source": os.getenv("QUANT_REALTIME_DATA_SOURCE", "mock"),
            "max_cycles": int(os.getenv("QUANT_PAPER_MAX_CYCLES", "3")),
            "sleep_seconds": int(os.getenv("QUANT_PAPER_SLEEP_SECONDS", "5")),
        },
    }

    out_path = Path("logs/paper_trade_preflight.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("PRECHECK OK")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_preflight()
