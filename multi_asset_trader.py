
"""
WR64-WINRATE64 - MULTI-ASSET TRADER (BTC, BNB, ETH) - ORIGINAL STRATEGY
"""
import os
import time
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd
import numpy as np

def round_price(client, symbol: str, price: float) -> float:
    """Round a price to the symbol's allowed tick size (avoids -1111 precision errors)"""
    info = client.futures_exchange_info()
    symbol_info = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
    tick_size = next(f["tickSize"] for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER")
    return float(Decimal(str(price)).quantize(Decimal(tick_size), rounding=ROUND_HALF_UP))

# Helper functions (EXACT FROM btc_backtest.py)
def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average (helper function)"""
    n = len(data)
    ema = np.zeros(n)
    if n < period:
        return ema
    ema[period-1] = np.mean(data[:period])
    alpha = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate Average True Range (helper function)"""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate RSI (Relative Strength Index) (helper function)"""
    close = df["close"].values
    n = len(df)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        change = close[i] - close[i-1]
        if change > 0:
            gains[i] = change
        elif change < 0:
            losses[i] = -change
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    return rsi

def generate_signal(df: pd.DataFrame) -> dict:
    """
    EXACT original strategy from btc_backtest.py:
    1. Trend filter: 300-period EMA (slower to avoid whipsaws)
    2. Entry: 20-period EMA crossover with confirmation (close > EMA20 by at least 0.3x ATR)
    3. Stop loss: 0.7x ATR (very tight)
    4. Take profit: 2.1x ATR (3:1 R:R)
    5. RSI filter: Tight (38-62)
    """
    close = df["close"].values
    n = len(df)

    # Calculate indicators
    ema20 = calculate_ema(close, 20)
    ema300 = calculate_ema(close, 300)  # Slower trend filter
    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, 14)

    # Track position and exit levels
    pos = np.zeros(n)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan

    for i in range(300, n):
        # First check if we need to exit current position
        if current_pos != 0:
            # Check stop loss and take profit
            if current_pos == 1:  # Long position
                if df["low"].values[i] <= stop_loss or df["high"].values[i] >= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
            elif current_pos == -1:  # Short position
                if df["high"].values[i] >= stop_loss or df["low"].values[i] <= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan

        # If flat, look for new entry
        if current_pos == 0:
            # Long entry: uptrend (close > EMA300), EMA20 crossover with confirmation
            if (close[i] > ema300[i] and
                ema20[i] > ema20[i-1] and
                close[i] > (ema20[i] + 0.3 * atr[i]) and  # Confirmation: close > EMA20 by 0.3x ATR
                close[i-1] <= ema20[i-1] and
                rsi[i] < 62):  # Tight RSI filter
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (0.7 * atr[i])  # Tight stop
                take_profit = entry_price + (2.1 * atr[i])  # 3:1 R:R
            # Short entry: downtrend (close < EMA300), EMA20 crossunder with confirmation
            elif (close[i] < ema300[i] and
                  ema20[i] < ema20[i-1] and
                  close[i] < (ema20[i] - 0.3 * atr[i]) and  # Confirmation: close < EMA20 by 0.3x ATR
                  close[i-1] >= ema20[i-1] and
                  rsi[i] > 38):  # Tight RSI filter
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (0.7 * atr[i])
                take_profit = entry_price - (2.1 * atr[i])

        pos[i] = current_pos

    return {
        "current_signal": pos[-1],
        "previous_signal": pos[-2],
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

def get_historical_data(client, symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    """Fetch historical klines from Binance Testnet with pagination"""
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

        klines = client.futures_klines(**params)
        
        if not klines:
            break
        
        all_data.extend(klines)
        remaining_bars -= len(klines)
        end_time = klines[0][0] - 1

    all_data.sort(key=lambda k: k[0])  # ensure ascending (oldest -> newest) order

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df

def get_current_position(client, symbol: str) -> dict | None:
    """Get current open position"""
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if float(pos["positionAmt"]) != 0:
                return {
                    "side": "LONG" if float(pos["positionAmt"]) > 0 else "SHORT",
                    "size": abs(float(pos["positionAmt"])),
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"])
                }
        return None
    except Exception as e:
        logging.error(f"Error getting {symbol} position: {e}")
        return None

def calculate_position_size(client, symbol: str, entry_price: float, stop_loss: float, risk_per_trade: float) -> float:
    """Calculate position size based on risk, with hard margin cap"""
    try:
        account = client.futures_account()
        balance = float([b for b in account["assets"] if b["asset"] == "USDT"][0]["walletBalance"])
        risk_amount = balance * risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        position_size = (risk_amount / stop_distance) * 20  # 20x leverage
        
        # Cap at 15% of max possible position per asset (to spread across 3 assets safely)
        max_possible_size = (balance * 20 / entry_price) * 0.15
        position_size = min(position_size, max_possible_size)
        
        symbol_info = client.futures_exchange_info()
        symbol_precision = next((s["quantityPrecision"] for s in symbol_info["symbols"] if s["symbol"] == symbol), 0)
        position_size = round(position_size, symbol_precision)
        
        return position_size
    except Exception as e:
        logging.error(f"Error calculating {symbol} position size: {e}")
        return 0

def enter_position(client, symbol: str, side: str, entry_price: float, stop_loss: float, take_profit: float, position_size: float):
    """Enter a new position"""
    try:
        client.futures_change_leverage(symbol=symbol, leverage=20)

        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=position_size
        )
        logging.info(f"=== {symbol} ORDER PLACED: {side} {position_size} @ ~{entry_price:.2f} ===")

        stop_loss = round_price(client, symbol, stop_loss)
        take_profit = round_price(client, symbol, take_profit)

        sl_order = client.futures_create_order(
            symbol=symbol,
            side="SELL" if side == Client.SIDE_BUY else "BUY",
            type="STOP_MARKET",
            stopPrice=stop_loss,
            quantity=position_size,
            closePosition=True
        )
        logging.info(f"{symbol} Stop Loss placed at {stop_loss:.2f}")
        
        tp_order = client.futures_create_order(
            symbol=symbol,
            side="SELL" if side == Client.SIDE_BUY else "BUY",
            type="TAKE_PROFIT_MARKET",
            stopPrice=take_profit,
            quantity=position_size,
            closePosition=True
        )
        logging.info(f"{symbol} Take Profit placed at {take_profit:.2f}")
        
        return True
    except (BinanceAPIException, BinanceOrderException) as e:
        logging.error(f"Error placing {symbol} order: {e}")
        return False

def main():
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("multi_asset_trades.log"),
            logging.StreamHandler()
        ]
    )
    
    client = Client(
        api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
        api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"),
        testnet=True
    )
    # Explicitly set correct testnet endpoints
    client.API_URL = 'https://testnet.binance.vision/api'
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client.WEBSITE_URL = 'https://testnet.binance.vision'
    
    # Configuration
    SYMBOLS = ["BTCUSDT", "BNBUSDT", "ETHUSDT"]
    INTERVAL = Client.KLINE_INTERVAL_1HOUR
    LOOKBACK = 350  # Need 350 for EMA300
    SLEEP_TIME = 60  # Check every minute
    RISK_PER_TRADE = 0.008  # 0.8% risk per trade for multi-asset
    LEVERAGE = 20
    
    logging.info("="*80)
    logging.info(f"{'WR64-WINRATE64 - MULTI-ASSET TRADER (ORIGINAL STRATEGY)':^80}")
    logging.info("="*80)
    logging.info(f"Trading: {', '.join(SYMBOLS)}")
    logging.info(f"Leverage: {LEVERAGE}x | Risk per trade: {RISK_PER_TRADE*100:.1f}%")
    logging.info(f"Strategy: EMA20/EMA300, 0.3x ATR confirmation, 0.7x ATR SL, 2.1x ATR TP, RSI 38-62")
    logging.info("="*80)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"\n=== MARKET CHECK @ {current_time} ===")
            
            for symbol in SYMBOLS:
                logging.info(f"\n--- {symbol} ---")
                
                # Get data
                df = get_historical_data(client, symbol, INTERVAL, LOOKBACK)
                if len(df) < 300:
                    logging.warning(f"Not enough data for {symbol} (needs 300, got {len(df)})")
                    continue
                
                # Generate signal
                signal_info = generate_signal(df)
                current_signal = signal_info["current_signal"]
                previous_signal = signal_info["previous_signal"]
                logging.info(f"Signal: {current_signal:.0f} (Prev: {previous_signal:.0f})")
                
                # Check position
                current_position = get_current_position(client, symbol)
                if current_position:
                    logging.info(f"Open position: {current_position['side']} {current_position['size']} | PnL: ${current_position['unrealized_pnl']:.2f}")
                    continue
                
                # No position, check for entry
                if current_position is None:
                    if current_signal == 1 and previous_signal == 0:
                        # Enter long
                        entry_price = df["close"].iloc[-1]
                        atr = calculate_atr(df, 14)
                        current_atr = atr[-1]
                        stop_loss = entry_price - (0.7 * current_atr)
                        take_profit = entry_price + (2.1 * current_atr)
                        
                        position_size = calculate_position_size(
                            client, symbol, entry_price, stop_loss, RISK_PER_TRADE
                        )
                        
                        if position_size > 0:
                            enter_position(
                                client, symbol, Client.SIDE_BUY,
                                entry_price, stop_loss, take_profit, position_size
                            )
                    elif current_signal == -1 and previous_signal == 0:
                        # Enter short
                        entry_price = df["close"].iloc[-1]
                        atr = calculate_atr(df, 14)
                        current_atr = atr[-1]
                        stop_loss = entry_price + (0.7 * current_atr)
                        take_profit = entry_price - (2.1 * current_atr)
                        
                        position_size = calculate_position_size(
                            client, symbol, entry_price, stop_loss, RISK_PER_TRADE
                        )
                        
                        if position_size > 0:
                            enter_position(
                                client, symbol, Client.SIDE_SELL,
                                entry_price, stop_loss, take_profit, position_size
                            )
            
            logging.info(f"\nWaiting {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)
            
        except KeyboardInterrupt:
            logging.info("\nStopping trader...")
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()
