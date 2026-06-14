
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance.client import Client

# Replicate the helper functions from trader.py
def calculate_ema(data, period):
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

def calculate_atr(df, period=14):
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

def calculate_rsi(df, period=14):
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

def signal(df, config):
    """
    Adjusted Strategy for Binance Testnet - Uses EMA 20/50 instead of 20/300!
    """
    close = df["close"].values
    n = len(df)
    
    # Calculate indicators
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, 14)
    
    # Track position and exit levels
    pos = pd.Series(np.zeros(n), index=df.index)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    
    print("=== Debug Info ===")
    print(f"Number of klines: {n}")
    print(f"Last 5 klines:")
    print(df.tail())
    print(f"\nLast 10 EMA20: {ema20[-10:]}")
    print(f"Last 10 EMA50: {ema50[-10:]}")
    print(f"Last 10 RSI: {rsi[-10:]}")
    print(f"Last 10 ATR: {atr[-10:]}")
    
    for i in range(50, n):
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
            long_conditions = [
                close[i] > ema50[i],
                ema20[i] > ema20[i-1],
                close[i] > (ema20[i] + 0.3 * atr[i]),
                close[i-1] <= ema20[i-1],
                rsi[i] < 62
            ]
            # Short entry: downtrend (close < EMA50), EMA20 crossunder with confirmation
            short_conditions = [
                close[i] < ema50[i],
                ema20[i] < ema20[i-1],
                close[i] < (ema20[i] - 0.3 * atr[i]),
                close[i-1] >= ema20[i-1],
                rsi[i] > 38
            ]
            if all(long_conditions):
                print(f"\nLONG SIGNAL at index {i} (time: {df.index[i]}):")
                print(f"  Close: {close[i]}, EMA20: {ema20[i]}, EMA50: {ema50[i]}")
                print(f"  Confirmation: {close[i]} > {ema20[i] + 0.3 * atr[i]}")
                print(f"  RSI: {rsi[i]}")
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (0.7 * atr[i])  # Tight SL
                take_profit = entry_price + (2.1 * atr[i])  # 3:1 R:R
            elif all(short_conditions):
                print(f"\nSHORT SIGNAL at index {i} (time: {df.index[i]}):")
                print(f"  Close: {close[i]}, EMA20: {ema20[i]}, EMA50: {ema50[i]}")
                print(f"  Confirmation: {close[i]} < {ema20[i] - 0.3 * atr[i]}")
                print(f"  RSI: {rsi[i]}")
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (0.7 * atr[i])
                take_profit = entry_price - (2.1 * atr[i])
        
        pos.iloc[i] = current_pos
    
    print(f"\n=== Final Signal ===")
    print(f"Current signal: {pos.iloc[-1]}, Previous: {pos.iloc[-2]}")
    return pos

def get_historical_data(symbol, interval, lookback, client):
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

        klines = client.get_klines(**params)
        
        if not klines:
            break
        
        all_data.extend(klines)
        remaining_bars -= len(klines)
        end_time = klines[0][0] - 1

    all_data.sort(key=lambda k: k[0])  # ensure ascending (oldest -> newest) order

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def main():
    load_dotenv()
    client = Client(
        api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
        api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
        testnet=True
    )
    config = {
        'leverage': 20,
        'maint_margin': 0.005,
        'start_equity': 50,
        'risk_per_trade': 0.05,
        'symbol': 'BTCUSDT',
        'interval': Client.KLINE_INTERVAL_1HOUR,
        'lookback': 200,  # Get more data for debugging
        'sleep_time': 300,
    }
    
    df = get_historical_data(config['symbol'], config['interval'], config['lookback'], client)
    signals = signal(df, config)

if __name__ == "__main__":
    main()
