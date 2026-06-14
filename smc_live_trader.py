"""
SMC Live Trader - Version 3
Uses SMC (Smart Money Concepts) with FVG (Fair Value Gap) + candlestick patterns
Includes all bug fixes: data chronology, order precision, connectivity
"""
import os
import time
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smc_trades.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET')

# Initialize Binance client (Testnet)
client = Client(API_KEY, API_SECRET, testnet=True)
client.API_URL = 'https://testnet.binance.vision/api'
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
client.WEBSITE_URL = 'https://testnet.binance.vision'

# Configuration
SYMBOLS = ["BTCUSDT", "BNBUSDT", "ETHUSDT"]
INTERVAL = Client.KLINE_INTERVAL_4HOUR  # SMC works better on higher timeframes
LEVERAGE = 20
RISK_PER_TRADE = 0.02  # 2% risk per trade (since $100 is small)
CHECK_INTERVAL = 300  # Check every 5 minutes
TOTAL_CAPITAL = 100.0  # Fixed $100 total capital

# Global state to track open positions
open_positions = {}


def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average"""
    n = len(data)
    ema = np.zeros(n)
    if n < period:
        return ema
    ema[period - 1] = np.mean(data[:period])
    alpha = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate Average True Range"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _is_hammer(o, h, l, c, i):
    """Check if current candle is a hammer"""
    rng = h[i] - l[i]
    if rng <= 0:
        return False
    body = abs(c[i] - o[i])
    lower_wick = min(o[i], c[i]) - l[i]
    upper_wick = h[i] - max(o[i], c[i])
    return body <= 0.3 * rng and lower_wick >= 2 * body and upper_wick <= 0.15 * rng


def _is_shooting_star(o, h, l, c, i):
    """Check if current candle is a shooting star"""
    rng = h[i] - l[i]
    if rng <= 0:
        return False
    body = abs(c[i] - o[i])
    upper_wick = h[i] - max(o[i], c[i])
    lower_wick = min(o[i], c[i]) - l[i]
    return body <= 0.3 * rng and upper_wick >= 2 * body and lower_wick <= 0.15 * rng


def _is_bullish_engulfing(o, h, l, c, i):
    """Check if current candle is bullish engulfing"""
    return c[i-1] < o[i-1] and c[i] > o[i] and c[i] >= o[i-1] and o[i] <= c[i-1]


def _is_bearish_engulfing(o, h, l, c, i):
    """Check if current candle is bearish engulfing"""
    return c[i-1] > o[i-1] and c[i] < o[i] and c[i] <= o[i-1] and o[i] >= c[i-1]


def _is_three_white_soldiers(o, h, l, c, i):
    """Check for three white soldiers pattern"""
    return (c[i-2] > o[i-2] and c[i-1] > o[i-1] and c[i] > o[i] and
            c[i] > c[i-1] > c[i-2] and
            o[i-1] > o[i-2] and o[i-1] < c[i-2] and
            o[i] > o[i-1] and o[i] < c[i-1])


def _is_three_black_crows(o, h, l, c, i):
    """Check for three black crows pattern"""
    return (c[i-2] < o[i-2] and c[i-1] < o[i-1] and c[i] < o[i] and
            c[i] < c[i-1] < c[i-2] and
            o[i-1] < o[i-2] and o[i-1] > c[i-2] and
            o[i] < o[i-1] and o[i] > c[i-1])


def get_historical_data(symbol: str, interval: str, lookback: int = 1000) -> pd.DataFrame:
    """Get historical data with proper chronological order (oldest first)"""
    all_data = []
    end_time = None
    remaining_bars = lookback
    max_limit_per_request = 1000

    while remaining_bars > 0:
        limit = min(remaining_bars, max_limit_per_request)
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if end_time is not None:
            params["endTime"] = end_time

        response = client.futures_klines(**params)
        if not response:
            break

        all_data.extend(response)
        remaining_bars -= len(response)
        end_time = response[0][0] - 1

    # SORT by timestamp instead of reverse() to ensure oldest first
    all_data.sort(key=lambda k: k[0])

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df


def generate_smc_signal(df: pd.DataFrame) -> dict:
    """Generate trading signal using SMC/FVG/Candlestick strategy"""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    ema_fast = calculate_ema(c, 50)
    ema_slow = calculate_ema(c, 200)
    atr = calculate_atr(df, 14)

    bull_zones = []  # (bottom, top, created_idx)
    bear_zones = []
    fvg_lookback = 30
    fvg_min_atr = 0.0
    sweep_lookback = 0

    start = max(200, 14, sweep_lookback) + 3
    if n < start + 1:
        return {"signal": "hold", "stop_loss": None, "take_profit": None}

    i = n - 1  # Current candle (latest)

    # Update FVG zones
    if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= fvg_min_atr * atr[i]:
        bull_zones.append((h[i - 2], l[i], i))
    if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= fvg_min_atr * atr[i]:
        bear_zones.append((h[i], l[i - 2], i))
    bull_zones = [z for z in bull_zones if i - z[2] <= fvg_lookback]
    bear_zones = [z for z in bear_zones if i - z[2] <= fvg_lookback]

    bull_bias = ema_fast[i] > ema_slow[i]
    bear_bias = ema_fast[i] < ema_slow[i]

    in_bull_zone = any(z[0] <= c[i] <= z[1] for z in bull_zones)
    in_bear_zone = any(z[0] <= c[i] <= z[1] for z in bear_zones)

    bull_pattern = (_is_hammer(o, h, l, c, i) or
                    _is_bullish_engulfing(o, h, l, c, i) or
                    _is_three_white_soldiers(o, h, l, c, i))
    bear_pattern = (_is_shooting_star(o, h, l, c, i) or
                    _is_bearish_engulfing(o, h, l, c, i) or
                    _is_three_black_crows(o, h, l, c, i))

    swept_low = swept_high = True
    if sweep_lookback > 0:
        swept_low = l[i] < np.min(l[i - sweep_lookback:i])
        swept_high = h[i] > np.max(h[i - sweep_lookback:i])

    signal = {"signal": "hold", "stop_loss": None, "take_profit": None}
    sl_atr = 1.0
    tp_atr = 2.5

    if bull_bias and in_bull_zone and bull_pattern and swept_low:
        signal["signal"] = "long"
        entry = c[i]
        stop_loss = min(l[i], l[i - 1]) - 0.1 * atr[i]
        if stop_loss >= entry:
            stop_loss = entry - sl_atr * atr[i]
        signal["stop_loss"] = stop_loss
        signal["take_profit"] = entry + tp_atr * atr[i]
    elif bear_bias and in_bear_zone and bear_pattern and swept_high:
        signal["signal"] = "short"
        entry = c[i]
        stop_loss = max(h[i], h[i - 1]) + 0.1 * atr[i]
        if stop_loss <= entry:
            stop_loss = entry + sl_atr * atr[i]
        signal["stop_loss"] = stop_loss
        signal["take_profit"] = entry - tp_atr * atr[i]

    return signal


def round_price(symbol: str, price: float) -> float:
    """Round price to correct precision based on symbol's tick size"""
    symbol_info = client.futures_exchange_info()
    symbol_info = next(s for s in symbol_info['symbols'] if s['symbol'] == symbol)
    tick_size = float(next(f['tickSize'] for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'))
    price_precision = int(round(-np.log10(tick_size), 0))
    return round(price, price_precision)


def round_quantity(symbol: str, quantity: float) -> float:
    """Round quantity to correct precision based on symbol's step size"""
    symbol_info = client.futures_exchange_info()
    symbol_info = next(s for s in symbol_info['symbols'] if s['symbol'] == symbol)
    step_size = float(next(f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'))
    qty_precision = int(round(-np.log10(step_size), 0))
    return round(quantity, qty_precision)


def get_account_balance() -> float:
    """Get available USDT balance from futures account"""
    balance_info = client.futures_account_balance()
    for balance in balance_info:
        if balance['asset'] == 'USDT':
            return float(balance['availableBalance'])
    return 0.0


def calculate_position_size(symbol: str, entry_price: float, stop_loss: float) -> float:
    """Calculate position size based on fixed $100 capital and risk"""
    risk_amount = TOTAL_CAPITAL * RISK_PER_TRADE
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        return 0.0

    # Calculate position size (units of base asset)
    position_size = (risk_amount / stop_distance) / LEVERAGE

    # Round to correct precision
    return round_quantity(symbol, position_size)


def get_open_positions() -> dict:
    """Get currently open futures positions"""
    positions = client.futures_position_information()
    open_pos = {}
    for pos in positions:
        if float(pos['positionAmt']) != 0:
            open_pos[pos['symbol']] = {
                'amount': float(pos['positionAmt']),
                'entry_price': float(pos['entryPrice']),
                'unrealized_pnl': float(pos['unRealizedProfit']),
                'side': 'long' if float(pos['positionAmt']) > 0 else 'short'
            }
    return open_pos


def close_position(symbol: str):
    """Close an open position"""
    pos = open_positions.get(symbol)
    if not pos:
        return

    side = Client.SIDE_SELL if pos['side'] == 'long' else Client.SIDE_BUY
    quantity = abs(pos['amount'])
    quantity = round_quantity(symbol, quantity)

    client.futures_create_order(
        symbol=symbol,
        side=side,
        type=Client.ORDER_TYPE_MARKET,
        quantity=quantity
    )

    logger.info(f"Closed {pos['side']} position for {symbol}")
    del open_positions[symbol]


def open_position(symbol: str, side: str, stop_loss: float, take_profit: float):
    """Open a new position"""
    # Close existing position first if any
    if symbol in open_positions:
        close_position(symbol)
    
    # Only allow one open position total (to stay within $100 capital)
    if len(open_positions) > 0:
        logger.warning("Already have an open position, not opening another")
        return

    # Get current price
    ticker = client.futures_symbol_ticker(symbol=symbol)
    entry_price = float(ticker['price'])

    # Round SL/TP
    stop_loss = round_price(symbol, stop_loss)
    take_profit = round_price(symbol, take_profit)

    # Calculate position size
    quantity = calculate_position_size(symbol, entry_price, stop_loss)
    if quantity <= 0:
        logger.warning(f"Position size too small for {symbol}")
        return

    # Open position
    order_side = Client.SIDE_BUY if side == 'long' else Client.SIDE_SELL

    try:
        client.futures_create_order(
            symbol=symbol,
            side=order_side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )

        # Set stop loss and take profit
        stop_side = Client.SIDE_SELL if side == 'long' else Client.SIDE_BUY
        client.futures_create_order(
            symbol=symbol,
            side=stop_side,
            type=Client.ORDER_TYPE_STOP_MARKET,
            stopPrice=stop_loss,
            closePosition=True
        )

        client.futures_create_order(
            symbol=symbol,
            side=stop_side,
            type=Client.ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=take_profit,
            closePosition=True
        )

        logger.info(f"Opened {side} position for {symbol} at {entry_price}, SL={stop_loss}, TP={take_profit}")
        open_positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'amount': quantity
        }
    except Exception as e:
        logger.error(f"Error opening position for {symbol}: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SMC Live Trader")
    parser.add_argument("--one-check", action="store_true", help="Run one market check and exit")
    args = parser.parse_args()

    logger.info("="*64)
    logger.info("SMC Live Trader - Version 3 Starting")
    logger.info(f"Symbols: {SYMBOLS}, Interval: {INTERVAL}, Leverage: {LEVERAGE}x")
    logger.info("="*64)

    # Set leverage for all symbols
    for symbol in SYMBOLS:
        try:
            client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
            logger.info(f"Set leverage to {LEVERAGE}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")

    # Load initial open positions
    global open_positions
    open_positions = get_open_positions()
    logger.info(f"Loaded {len(open_positions)} open positions")

    def run_market_check():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n=== Market Check @ {now} ===")

        # Refresh open positions
        open_positions = get_open_positions()

        for symbol in SYMBOLS:
            logger.info(f"--- {symbol} ---")

            # Get historical data
            df = get_historical_data(symbol, INTERVAL, 1000)
            if len(df) < 300:
                logger.warning(f"Not enough data for {symbol}")
                continue

            # Generate signal
            signal = generate_smc_signal(df)

            # Check current position
            current_pos = open_positions.get(symbol)

            if current_pos:
                logger.info(f"Current position: {current_pos['side']} @ {current_pos['entry_price']}, PnL: ${current_pos['unrealized_pnl']:.2f}")
                # Check if we need to close (opposite signal)
                if (current_pos['side'] == 'long' and signal['signal'] == 'short') or \
                   (current_pos['side'] == 'short' and signal['signal'] == 'long'):
                    logger.info(f"Opposite signal detected, closing position for {symbol}")
                    close_position(symbol)
            else:
                logger.info("No current position")
                # Open new position if we have a signal
                if signal['signal'] in ['long', 'short']:
                    logger.info(f"Signal: {signal['signal']}, SL: {signal['stop_loss']:.2f}, TP: {signal['take_profit']:.2f}")
                    open_position(symbol, signal['signal'], signal['stop_loss'], signal['take_profit'])
                else:
                    logger.info("No trading signal")

    if args.one_check:
        run_market_check()
        logger.info("One check complete, exiting.")
        return

    while True:
        try:
            run_market_check()
            # Sleep until next check
            logger.info(f"\nWaiting {CHECK_INTERVAL} seconds until next check...")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
