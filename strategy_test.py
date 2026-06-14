
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

def simulate_strategy(df):
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
    liq_buffer = (1/20) - 0.005  # 20x leverage
    sl_buffer = liq_buffer * 0.45
    
    trades = []
    
    print(f"\n{'='*80}")
    print(f"{'STRATEGY SIMULATION RESULTS':^80}")
    print(f"{'='*80}")
    print(f"{'Date':^20} | {'Close':^10} | {'Signal':^8} | {'Action':^15} | {'PnL':^10}")
    print(f"{'-'*80}")
    
    for i in range(15, n):
        action = ""
        pnl = 0.0
        
        # Check exit conditions
        if current_pos != 0:
            if current_pos == 1:
                if df['low'].iloc[i] <= stop_loss:
                    # Hit SL
                    pnl = (stop_loss - entry_price) * 1
                    action = "SL Hit"
                    trades.append(('LONG', entry_price, stop_loss, pnl))
                    current_pos = 0
                elif df['high'].iloc[i] >= take_profit:
                    # Hit TP
                    pnl = (take_profit - entry_price) * 1
                    action = "TP Hit"
                    trades.append(('LONG', entry_price, take_profit, pnl))
                    current_pos = 0
            elif current_pos == -1:
                if df['high'].iloc[i] >= stop_loss:
                    # Hit SL
                    pnl = (entry_price - stop_loss) * 1
                    action = "SL Hit"
                    trades.append(('SHORT', entry_price, stop_loss, pnl))
                    current_pos = 0
                elif df['low'].iloc[i] <= take_profit:
                    # Hit TP
                    pnl = (entry_price - take_profit) * 1
                    action = "TP Hit"
                    trades.append(('SHORT', entry_price, take_profit, pnl))
                    current_pos = 0
        
        # Check entry conditions
        if current_pos == 0:
            if ema5[i] > ema15[i] and ema5[i-1] <= ema15[i-1] and 20 < rsi.iloc[i] < 80:
                # Long entry
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (sl_buffer * entry_price)
                take_profit = entry_price + (sl_buffer * entry_price * 1.9)
                action = "ENTER LONG"
                pos[i] = 1
            elif ema5[i] < ema15[i] and ema5[i-1] >= ema15[i-1] and 20 < rsi.iloc[i] < 80:
                # Short entry
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (sl_buffer * entry_price)
                take_profit = entry_price - (sl_buffer * entry_price * 1.9)
                action = "ENTER SHORT"
                pos[i] = -1
        
        if action != "":
            print(f"{df.index[i]:^20} | {close[i]:^10.2f} | {pos[i]:^8.0f} | {action:^15} | {pnl:^10.2f}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"{'TRADING SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"Total trades made: {len(trades)}")
    
    if len(trades) > 0:
        winning_trades = sum(1 for t in trades if t[3] > 0)
        losing_trades = sum(1 for t in trades if t[3] < 0)
        total_pnl = sum(t[3] for t in trades)
        
        print(f"Winning trades: {winning_trades}")
        print(f"Losing trades: {losing_trades}")
        print(f"Total PnL (1 BTC position): ${total_pnl:.2f}")
        print(f"Average PnL per trade: ${total_pnl/len(trades):.2f}")
        
        if winning_trades > 0:
            avg_win = sum(t[3] for t in trades if t[3] > 0) / winning_trades
            avg_loss = sum(abs(t[3]) for t in trades if t[3] < 0) / losing_trades if losing_trades > 0 else 0
            print(f"Average win: ${avg_win:.2f}")
            print(f"Average loss: ${avg_loss:.2f}")

def main():
    print("=== WR64-WINRATE64 STRATEGY TEST ===")
    df = get_historical_data('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, 300)
    print(f"Got {len(df)} hours of historical data from {df.index[0]} to {df.index[-1]}")
    simulate_strategy(df)

if __name__ == "__main__":
    main()
