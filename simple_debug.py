
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance.client import Client

def main():
    load_dotenv()
    client = Client(
        api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
        api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
        testnet=True
    )
    
    # Get data
    print("Fetching data...")
    klines = client.get_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=200
    )
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Data fetched: {len(df)} klines")
    print("\nLast 20 klines:")
    print(df[['open', 'high', 'low', 'close', 'volume']].tail(20))
    
    # Calculate indicators
    close = df['close'].values
    
    # Calculate EMA20
    ema20 = np.zeros(len(close))
    if len(close) >= 20:
        ema20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close)):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    # Calculate EMA50
    ema50 = np.zeros(len(close))
    if len(close) >= 50:
        ema50[49] = np.mean(close[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close)):
            ema50[i] = alpha * close[i] + (1 - alpha) * ema50[i-1]
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    print("\nLast 20 indicator values:")
    print(f"Close:      {close[-20:]}")
    print(f"EMA20:      {ema20[-20:]}")
    print(f"EMA50:      {ema50[-20:]}")
    print(f"RSI:        {rsi.values[-20:]}")
    
    # Check for signals
    print("\nChecking for signals in last 50 klines:")
    for i in range(len(df)-50, len(df)):
        if i < 50:
            continue
        # Long conditions
        long_conds = [
            close[i] > ema50[i],
            ema20[i] > ema20[i-1],
            close[i] > (ema20[i] + 0.3 * (df['high'].iloc[i] - df['low'].iloc[i])),
            close[i-1] <= ema20[i-1],
            rsi.iloc[i] < 62
        ]
        if all(long_conds):
            print(f"LONG at {df.index[i]} - Close: {close[i]}")
        
        # Short conditions
        short_conds = [
            close[i] < ema50[i],
            ema20[i] < ema20[i-1],
            close[i] < (ema20[i] - 0.3 * (df['high'].iloc[i] - df['low'].iloc[i])),
            close[i-1] >= ema20[i-1],
            rsi.iloc[i] > 38
        ]
        if all(short_conds):
            print(f"SHORT at {df.index[i]} - Close: {close[i]}")

if __name__ == "__main__":
    main()
