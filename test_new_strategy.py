
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance.client import Client
import os

def calculate_ema(data, period):
    n = len(data)
    ema = np.zeros(n)
    if n < period:
        return ema
    ema[period-1] = np.mean(data[:period])
    alpha = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_historical_data(symbol, interval, lookback):
    load_dotenv()
    client = Client(
        api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
        api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
        testnet=True
    )
    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=lookback
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
    return df

def test_strategy(df):
    close = df['close'].values
    n = len(df)
    ema5 = calculate_ema(close, 5)
    ema15 = calculate_ema(close, 15)
    rsi = calculate_rsi(df, 14)
    
    pos = np.zeros(n)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    liq_buffer = (1/20) - 0.005
    sl_buffer = liq_buffer * 0.45
    
    print("=== Testing New Strategy ===")
    print(f"Number of klines: {n}")
    print(f"Liquidation buffer: {liq_buffer:.4%}")
    print(f"SL buffer: {sl_buffer:.4%}")
    print()
    
    signals_count = 0
    for i in range(15, n):
        if current_pos !=0:
            if current_pos == 1:
                if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit:
                    current_pos = 0
            elif current_pos == -1:
                if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit:
                    current_pos =0
        
        if current_pos == 0:
            if ema5[i] > ema15[i] and ema5[i-1] <= ema15[i-1] and rsi.iloc[i] >20 and rsi.iloc[i]<80:
                print(f"LONG signal at {df.index[i]}: close={close[i]}, ema5={ema5[i]:.2f}, ema15={ema15[i]:.2f}, rsi={rsi.iloc[i]:.2f}")
                current_pos =1
                entry_price = close[i]
                stop_loss = entry_price - (sl_buffer * entry_price)
                take_profit = entry_price + (sl_buffer * entry_price * 1.9)
                signals_count +=1
            elif ema5[i] < ema15[i] and ema5[i-1] >= ema15[i-1] and rsi.iloc[i]>20 and rsi.iloc[i]<80:
                print(f"SHORT signal at {df.index[i]}: close={close[i]}, ema5={ema5[i]:.2f}, ema15={ema15[i]:.2f}, rsi={rsi.iloc[i]:.2f}")
                current_pos =-1
                entry_price = close[i]
                stop_loss = entry_price + (sl_buffer * entry_price)
                take_profit = entry_price - (sl_buffer * entry_price * 1.9)
                signals_count +=1
        
        pos[i] = current_pos
    
    print()
    print(f"Total signals found: {signals_count}")
    return pos

def main():
    df = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, 200)
    signals = test_strategy(df)

if __name__ == "__main__":
    main()
