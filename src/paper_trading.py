"""
Step 7: Paper Trading Path Scaffold.
Real-time data ingestion, signal generation, live order placement skeleton.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import FROZEN_OBJECTIVE_V1_PARAMS, enforce_objective_freeze

# Configure logging. 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG & KNOBS
# ============================================================================

PAPER_TRADING_MODE = os.getenv("QUANT_PAPER_TRADING_MODE", "simulation").lower()  # "simulation" or "live"
REAL_TIME_DATA_SOURCE = os.getenv("QUANT_REALTIME_DATA_SOURCE", "zerodha").lower()  # "zerodha", "redis", "mock"
INITIAL_CAPITAL = float(os.getenv("QUANT_PAPER_INITIAL_CAPITAL", "100000"))
MAX_POSITION_PCT = float(os.getenv("QUANT_PAPER_MAX_POSITION_PCT", "5"))  # Max % of capital per position
MAX_DAILY_LOSS_PCT = float(os.getenv("QUANT_PAPER_MAX_DAILY_LOSS_PCT", "5"))  # Daily loss circuit breaker
REBALANCE_FREQ_MINUTES = int(os.getenv("QUANT_PAPER_REBALANCE_FREQ_MIN", "1"))  # How often to check signals
TRADING_HOURS_START = os.getenv("QUANT_PAPER_TRADING_HOURS_START", "09:15")
TRADING_HOURS_END = os.getenv("QUANT_PAPER_TRADING_HOURS_END", "15:30")
PAPER_MAX_CYCLES = max(0, int(os.getenv("QUANT_PAPER_MAX_CYCLES", "0")))
PAPER_SLEEP_SECONDS = max(0, int(os.getenv("QUANT_PAPER_SLEEP_SECONDS", "0")))
PAPER_FORCE_RUN_OUTSIDE_HOURS = os.getenv("QUANT_PAPER_FORCE_RUN_OUTSIDE_HOURS", "0").strip() == "1"

# Load frozen config from backtest.
FROZEN_CONFIG = {
    "horizon_days": 3,
    "holding_days": 3,
    "abs_threshold": 0.0025,
    "transaction_cost_bps": 7,
    "regime_safety_enabled": 1,
}

logger.info(f"Paper Trading mode: {PAPER_TRADING_MODE}")
logger.info(f"Real-time data source: {REAL_TIME_DATA_SOURCE}")
logger.info(f"Initial capital: {INITIAL_CAPITAL}")


def verify_objective_freeze():
    for key, value in FROZEN_OBJECTIVE_V1_PARAMS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("QUANT_OBJECTIVE_FREEZE_ALLOW_TARGET_GRID", "1")
    freeze_meta = enforce_objective_freeze(strict=True)
    logger.info(
        "Objective freeze verified: %s",
        {
            "name": freeze_meta["freeze_name"],
            "hash": freeze_meta["current_hash"],
            "ok": freeze_meta["ok"],
        },
    )
    return freeze_meta


def run_preflight_checks(model_path):
    model_file = Path(model_path)
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    logger.info(
        "Preflight OK: %s",
        {
            "model": str(model_file),
            "size_bytes": model_file.stat().st_size,
            "logs_dir": str(logs_dir.resolve()),
        },
    )

# ============================================================================
# REAL-TIME DATA STREAM (STUB)
# ============================================================================

class RealtimeDataStream:
    """
    Fetch live price ticks and feature data.
    TODO: Implement zerodha websocket integration.
    """
    
    def __init__(self, source="mock"):
        self.source = source
        self.latest_ticks = {}  # {ticker: {'price': float, 'timestamp': str}}
    
    def connect(self):
        """Connect to data source."""
        logger.info(f"Connecting to {self.source} real-time data stream...")
        if self.source == "zerodha":
            # TODO: Initialize KiteConnect websocket.
            pass
        elif self.source == "redis":
            # TODO: Initialize Redis subscriber.
            pass
        elif self.source == "mock":
            logger.info("Using mock data stream (for testing)")
            now = datetime.now().isoformat()
            for ticker in self.get_universe():
                self.latest_ticks[ticker] = {"price": float(np.random.uniform(90, 510)), "timestamp": now}
        logger.info("Data stream ready.")
    
    def get_latest_tick(self, ticker):
        """Fetch latest price + features for ticker."""
        return self.latest_ticks.get(ticker, {})
    
    def get_universe(self):
        """Fetch current trading universe (tickers)."""
        # TODO: Load from config or fetch from market api.
        return ["INFY", "TCS", "HDFCBANK", "MARUTI", "BAJAJ-AUTO"]  # Mock.
    
    def disconnect(self):
        """Clean shutdown."""
        logger.info("Disconnecting from data stream.")

# ============================================================================
# SIGNAL GENERATOR (STUB)
# ============================================================================

class SignalGenerator:
    """
    Load trained model, compute confidence scores, apply regime filters.
    Uses same model + config from Step 5 backtest.
    """
    
    def __init__(self, model_path="models/quantpulse_model.pth"):
        self.model_path = model_path
        self.model = None
        self.regime_state = 0
        self.volatility_regime = 1.0
    
    def load_model(self):
        """Load trained ensemble model."""
        logger.info(f"Loading model from {self.model_path}...")
        # TODO: Load PyTorch model.
        logger.info("Model loaded.")
    
    def compute_signals(self, tickers, latest_features):
        """
        Compute confidence scores for each ticker.
        
        Args:
            tickers: List of ticker symbols.
            latest_features: Dict {ticker: feature_dict}.
        
        Returns:
            Dict {ticker: confidence_score}.
        """
        signals = {}
        for ticker in tickers:
            if ticker not in latest_features:
                continue
            
            features = latest_features[ticker]
            
            # TODO: Forward pass through model.
            # For now, use mock.
            confidence = np.random.uniform(0.4, 0.8)
            signals[ticker] = {
                "confidence": confidence,
                "regime_label": "BULLISH",  # TODO: Compute real regime.
            }
        
        return signals
    
    def apply_regime_filters(self, signals, regime_state, volatility_regime):
        """
        Filter signals based on regime (bad_regime → lower threshold).
        """
        filtered_signals = {}
        for ticker, sig in signals.items():
            regime_label = sig["regime_label"]
            
            # TODO: Apply regime-based rank thresholds (same as backtest).
            if regime_label == "BEARISH":
                min_conf = 0.65
            else:
                min_conf = 0.55
            
            if sig["confidence"] >= min_conf:
                filtered_signals[ticker] = sig
        
        return filtered_signals

# ============================================================================
# POSITION MANAGER (STUB)
# ============================================================================

class PositionManager:
    """
    Track open positions, entry prices, P&L.
    """
    
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = {}  # {ticker: {"shares": int, "entry_price": float, "entry_date": str}}
        self.trades_log = []
    
    def open_position(self, ticker, price, qty, reason="signal"):
        """Record entry into position."""
        if ticker in self.positions:
            logger.warning(f"Position {ticker} already open. Ignoring new entry.")
            return
        
        self.positions[ticker] = {
            "shares": qty,
            "entry_price": price,
            "entry_date": datetime.now().isoformat(),
            "reason": reason,
        }
        logger.info(f"ENTRY {ticker}: {qty} shares @ {price}")
    
    def close_position(self, ticker, price):
        """Record exit from position."""
        if ticker not in self.positions:
            logger.warning(f"Position {ticker} not open. Ignoring close.")
            return
        
        pos = self.positions.pop(ticker)
        pnl = (price - pos["entry_price"]) * pos["shares"]
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        
        self.trades_log.append({
            "ticker": ticker,
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "shares": pos["shares"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_date": pos["entry_date"],
            "exit_date": datetime.now().isoformat(),
        })
        
        logger.info(f"EXIT {ticker}: P&L {pnl:+.2f} ({pnl_pct:+.2f}%)")
        return pnl, pnl_pct
    
    def get_portfolio_value(self, latest_prices):
        """Compute current portfolio value (cash + positions)."""
        cash = self.capital
        for ticker, pos in self.positions.items():
            if ticker in latest_prices:
                cash += pos["shares"] * latest_prices[ticker]
        return cash
    
    def check_daily_loss_circuit_breaker(self, daily_pnl_pct, max_daily_loss_pct):
        """Halt trading if daily loss exceeds threshold."""
        if daily_pnl_pct < -max_daily_loss_pct:
            logger.warning(f"Daily loss {daily_pnl_pct:.2f}% exceeds {max_daily_loss_pct}%. CIRCUIT BREAKER TRIGGERED.")
            return True
        return False
    
    def export_positions(self):
        """Export positions snapshot to JSON."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.capital,
            "positions": self.positions,
            "trades_log_count": len(self.trades_log),
        }
        with open("logs/paper_positions_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2)
        logger.info("Positions snapshot exported.")

# ============================================================================
# ORDER EXECUTION (STUB)
# ============================================================================

class OrderExecutor:
    """
    Place live orders via Zerodha API or mock orders.
    """
    
    def __init__(self, mode="simulation"):
        self.mode = mode  # "simulation" or "live"
        self.kite = None  # KiteConnect instance (if live)
    
    def connect(self):
        """Initialize broker connection."""
        if self.mode == "live":
            logger.warning("LIVE MODE: Orders will execute via Zerodha API.")
            # TODO: KiteConnect initialization (requires API key).
        else:
            logger.info(f"SIMULATION MODE: Orders will be logged but not executed.")
    
    def place_order(self, ticker, qty, price, direction="BUY"):
        """
        Place market order (BUY/SELL).
        
        Args:
            ticker: Stock symbol.
            qty: Number of shares.
            price: Current price (for slippage calculation).
            direction: "BUY" or "SELL".
        
        Returns:
            order_id (str) or None if failed.
        """
        if self.mode == "live":
            # TODO: Actual Zerodha API call.
            # order_id = self.kite.place_order(
            #     variety=self.kite.VARIETY_REGULAR,
            #     exchange=self.kite.EXCHANGE_NSE,
            #     tradingsymbol=ticker,
            #     transaction_type=self.kite.TRANSACTION_TYPE_BUY,
            #     quantity=qty,
            #     order_type=self.kite.ORDER_TYPE_MARKET,
            # )
            # logger.info(f"Order {order_id} placed: {direction} {qty} {ticker}")
            pass
        else:
            order_id = f"MOCK_{datetime.now().timestamp()}"
            logger.info(f"[SIMULATION] {direction} {qty} {ticker} @ {price} (Order {order_id})")
            return order_id
        
        return None
    
    def cancel_order(self, order_id):
        """Cancel open order."""
        logger.info(f"Cancelling order {order_id}...")
        # TODO: Implement cancel.

# ============================================================================
# MAIN TRADING LOOP
# ============================================================================

def main():
    """Paper trading main loop."""
    
    logger.info("=" * 80)
    logger.info("PAPER TRADING SESSION STARTED")
    logger.info("=" * 80)
    
    # Initialize components.
    data_stream = RealtimeDataStream(source=REAL_TIME_DATA_SOURCE)
    signal_gen = SignalGenerator()
    pos_mgr = PositionManager(initial_capital=INITIAL_CAPITAL)
    order_exec = OrderExecutor(mode=PAPER_TRADING_MODE)

    verify_objective_freeze()
    run_preflight_checks(signal_gen.model_path)
    
    data_stream.connect()
    signal_gen.load_model()
    order_exec.connect()
    
    # Main loop.
    cycle_count = 0
    try:
        while True:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            
            # Check trading hours.
            if (not PAPER_FORCE_RUN_OUTSIDE_HOURS) and not (TRADING_HOURS_START <= current_time <= TRADING_HOURS_END):
                logger.debug(f"Outside trading hours ({current_time}). Sleeping...")
                import time
                time.sleep(max(1, PAPER_SLEEP_SECONDS or 60))
                continue
            
            # 1. Fetch latest ticks.
            universe = data_stream.get_universe()
            latest_features = {}
            for ticker in universe:
                tick = data_stream.get_latest_tick(ticker)
                if tick:
                    latest_features[ticker] = tick
            
            if not latest_features:
                logger.debug("No tick data. Retrying...")
                import time
                time.sleep(max(1, PAPER_SLEEP_SECONDS or REBALANCE_FREQ_MINUTES * 60))
                continue
            
            # 2. Generate signals.
            signals = signal_gen.compute_signals(universe, latest_features)
            signals = signal_gen.apply_regime_filters(signals, 0, 1.0)  # TODO: Pass real regime.
            
            logger.info(f"[{current_time}] Signals: {len(signals)} candidates")
            
            # 3. Allocate positions.
            # TODO: Implement Kelly-based allocation.
            # For now, equal-weight top 5.
            top_signals = sorted(signals.items(), key=lambda x: x[1]["confidence"], reverse=True)[:5]
            
            for ticker, sig in top_signals:
                if ticker not in pos_mgr.positions:
                    # Get current price.
                    price = latest_features[ticker].get("price", 100)
                    qty = int((INITIAL_CAPITAL * MAX_POSITION_PCT / 100) / price)
                    
                    if qty > 0:
                        order_exec.place_order(ticker, qty, price, direction="BUY")
                        pos_mgr.open_position(ticker, price, qty, reason="signal")
            
            # 4. Exit on-hold positions after holding_days.
            for ticker in list(pos_mgr.positions.keys()):
                pos = pos_mgr.positions[ticker]
                entry_datetime = datetime.fromisoformat(pos["entry_date"])
                hold_days = (now - entry_datetime).days
                
                if hold_days >= FROZEN_CONFIG["holding_days"]:
                    price = latest_features.get(ticker, {}).get("price", 100)
                    order_exec.place_order(ticker, pos["shares"], price, direction="SELL")
                    pos_mgr.close_position(ticker, price)
            
            # Export snapshot.
            pos_mgr.export_positions()

            cycle_count += 1
            if PAPER_MAX_CYCLES > 0 and cycle_count >= PAPER_MAX_CYCLES:
                logger.info("Reached QUANT_PAPER_MAX_CYCLES=%s. Exiting.", PAPER_MAX_CYCLES)
                break
            
            # Sleep until next rebalance.
            import time
            time.sleep(max(1, PAPER_SLEEP_SECONDS or REBALANCE_FREQ_MINUTES * 60))
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        data_stream.disconnect()
        pos_mgr.export_positions()
        logger.info("Session closed.")

if __name__ == "__main__":
    main()
