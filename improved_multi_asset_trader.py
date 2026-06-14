
"""
Improved Multi-Asset Trader - More Trades, Still Safe
"""
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd
import numpy as np

# --------------------------
# Helper Functions (EXACT)
# --------------------------
def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
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
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
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
    IMPROVED Strategy (from btc_backtest.py but more active):
    - EMA20/EMA50 instead of EMA300 (more trades)
    - 0.1xATR confirmation instead of 0.3x (easier entry)
    - 1.0xATR SL instead of 0.7x (wider stop)
    - 2.0xATR TP instead of 2.1x (same R:R)
    - RSI 30-70 instead of 38-62 (more relaxed)
    """
    close = df["close"].values
    n = len(df)

    # Calculate indicators
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)  # Instead of 300
    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, 14)

    # Track position and exit levels
    pos = np.zeros(n)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan

    for i in range(50, n):  # Only need 50 bars now
        timestamp = df["timestamp"].iloc[i]
        curr_close = close[i]
        
        # First check if we need to exit current position
        if current_pos != 0:
            if current_pos == 1:  # Long position
                if df["low"].iloc[i] <= stop_loss or df["high"].iloc[i] >= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
            elif current_pos == -1:  # Short position
                if df["high"].iloc[i] >= stop_loss or df["low"].iloc[i] <= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan

        # If flat, look for new entry
        if current_pos == 0:
            # Long entry: uptrend (close > EMA50), EMA20 crossover with confirmation
            if (curr_close > ema50[i] and
                ema20[i] > ema20[i-1] and
                curr_close > (ema20[i] + 0.1 * atr[i]) and  # 0.1xATR instead of 0.3x
                close[i-1] <= ema20[i-1] and
                rsi[i] < 70):  # 70 instead of 62
                current_pos = 1
                entry_price = curr_close
                stop_loss = entry_price - (1.0 * atr[i])  # 1.0xATR instead of 0.7x
                take_profit = entry_price + (2.0 * atr[i])  # 2.0xATR instead of 2.1x
                print(f"[{timestamp}] LONG SIGNAL: Close={curr_close:.2f}, EMA20={ema20[i]:.2f}, EMA50={ema50[i]:.2f}, ATR={atr[i]:.2f}, RSI={rsi[i]:.2f}")
                
            # Short entry: downtrend (close < EMA50), EMA20 crossunder with confirmation
            elif (curr_close < ema50[i] and
                  ema20[i] < ema20[i-1] and
                  curr_close < (ema20[i] - 0.1 * atr[i]) and  # 0.1xATR instead of 0.3x
                  close[i-1] >= ema20[i-1] and
                  rsi[i] > 30):  # 30 instead of 38
                current_pos = -1
                entry_price = curr_close
                stop_loss = entry_price + (1.0 * atr[i])  # 1.0xATR instead of 0.7x
                take_profit = entry_price - (2.0 * atr[i])  # 2.0xATR instead of 2.1x
                print(f"[{timestamp}] SHORT SIGNAL: Close={curr_close:.2f}, EMA20={ema20[i]:.2f}, EMA50={ema50[i]:.2f}, ATR={atr[i]:.2f}, RSI={rsi[i]:.2f}")

        pos[i] = current_pos

    return {
        "current_signal": pos[-1],
        "previous_signal": pos[-2],
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

def get_historical_data(client, symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    all_data = []
    end_time = None
    remaining_bars = lookback
    max_limit_per_request = 1000

    while remaining_bars > 0:
        limit = min(remaining_bars, max_limit_per_request)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        klines = client.futures_klines(**params)
        if not klines:
            break
        all_data.extend(klines)
        remaining_bars -= len(klines)
        end_time = klines[0][0] - 1

    all_data.reverse()
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.reset_index(drop=True)
    return df

def get_current_position(client, symbol: str) -> dict | None:
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
    try:
        account = client.futures_account()
        balance = float([b for b in account["assets"] if b["asset"] == "USDT"][0]["walletBalance"])
        risk_amount = balance * risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        position_size = (risk_amount / stop_distance) * 20  # 20x leverage
        max_possible_size = (balance * 20 / entry_price) * 0.20  # 20% per asset
        position_size = min(position_size, max_possible_size)
        
        symbol_info = client.futures_exchange_info()
        symbol_precision = next((s["quantityPrecision"] for s in symbol_info["symbols"] if s["symbol"] == symbol), 0)
        position_size = round(position_size, symbol_precision)
        return position_size
    except Exception as e:
        logging.error(f"Error calculating {symbol} position size: {e}")
        return 0

def enter_position(client, symbol: str, side: str, entry_price: float, stop_loss: float, take_profit: float, position_size: float):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=20)
        
        order = client.futures_create_order(
            symbol=symbol, side=side, type="MARKET", quantity=position_size
        )
        logging.info(f"=== {symbol} ENTRY: {side} {position_size} @ ~{entry_price:.2f} ===")
        
        sl_order = client.futures_create_order(
            symbol=symbol,
            side="SELL" if side == Client.SIDE_BUY else "BUY",
            type="STOP_MARKET",
            stopPrice=stop_loss,
            quantity=position_size,
            closePosition=True
        )
        logging.info(f"{symbol} SL placed @ {stop_loss:.2f}")
        
        tp_order = client.futures_create_order(
            symbol=symbol,
            side="SELL" if side == Client.SIDE_BUY else "BUY",
            type="TAKE_PROFIT_MARKET",
            stopPrice=take_profit,
            quantity=position_size,
            closePosition=True
        )
        logging.info(f"{symbol} TP placed @ {take_profit:.2f}")
        
        return True
    except (BinanceAPIException, BinanceOrderException) as e:
        logging.error(f"Error placing {symbol} order: {e}")
        return False

def main():
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("improved_trades.log"), logging.StreamHandler()]
    )
    
    client = Client(
        api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
        api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"),
        testnet=True
    )
    
    SYMBOLS = ["BTCUSDT", "BNBUSDT", "ETHUSDT"]
    INTERVAL = Client.KLINE_INTERVAL_1HOUR
    LOOKBACK = 200
    SLEEP_TIME = 300  # 5 minutes
    RISK_PER_TRADE = 0.015  # 1.5%
    
    logging.info("="*80)
    logging.info("IMPROVED MULTI-ASSET TRADER")
    logging.info("="*80)
    logging.info(f"Trading: {', '.join(SYMBOLS)}")
    logging.info(f"Leverage: 20x, Risk/Trade: {RISK_PER_TRADE*100:.1f}%")
    logging.info(f"Strategy: EMA20/EMA50, 0.1xATR Confirm, 1.0xATR SL, 2.0xATR TP, RSI30-70")
    logging.info("="*80)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"\n=== MARKET CHECK @ {current_time} ===")
            
            for symbol in SYMBOLS:
                logging.info(f"\n--- {symbol} ---")
                
                df = get_historical_data(client, symbol, INTERVAL, LOOKBACK)
                if len(df) < 50:
                    logging.warning(f"Not enough data for {symbol}")
                    continue
                
                signal_info = generate_signal(df)
                current_signal = signal_info["current_signal"]
                previous_signal = signal_info["previous_signal"]
                
                current_position = get_current_position(client, symbol)
                if current_position:
                    logging.info(f"Position: {current_position['side']} {current_position['size']} | PnL: ${current_position['unrealized_pnl']:.2f}")
                    continue
                
                if current_position is None:
                    if current_signal == 1 and previous_signal == 0:
                        entry_price = df["close"].iloc[-1]
                        atr = calculate_atr(df, 14)
                        current_atr = atr[-1]
                        stop_loss = entry_price - (1.0 * current_atr)
                        take_profit = entry_price + (2.0 * current_atr)
                        position_size = calculate_position_size(client, symbol, entry_price, stop_loss, RISK_PER_TRADE)
                        
                        if position_size > 0:
                            enter_position(client, symbol, Client.SIDE_BUY, entry_price, stop_loss, take_profit, position_size)
                    
                    elif current_signal == -1 and previous_signal == 0:
                        entry_price = df["close"].iloc[-1]
                        atr = calculate_atr(df, 14)
                        current_atr = atr[-1]
                        stop_loss = entry_price + (1.0 * current_atr)
                        take_profit = entry_price - (2.0 * current_atr)
                        position_size = calculate_position_size(client, symbol, entry_price, stop_loss, RISK_PER_TRADE)
                        
                        if position_size > 0:
                            enter_position(client, symbol, Client.SIDE_SELL, entry_price, stop_loss, take_profit, position_size)
            
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
