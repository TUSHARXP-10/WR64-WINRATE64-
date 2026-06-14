
"""
Test improved strategy on real Binance data
"""
import numpy as np
import pandas as pd
import requests

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

def fetch_binance_ohlcv(symbol: str = "BTCUSDT", interval: str = "1h", total_bars: int = 500) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    remaining_bars = total_bars
    max_limit_per_request = 1000

    while remaining_bars > 0:
        limit = min(remaining_bars, max_limit_per_request)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        response = requests.get(url, params=params)
        response.raise_for_status()
        batch_data = response.json()
        if not batch_data:
            break
        all_data.extend(batch_data)
        remaining_bars -= len(batch_data)
        end_time = batch_data[0][0] - 1

    all_data.reverse()
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.reset_index(drop=True)
    return df

def backtest_improved_strategy(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    close = df["close"].values
    n = len(df)
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, 14)
    
    pos = np.zeros(n)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    trades = []
    
    for i in range(50, n):
        timestamp = df["timestamp"].iloc[i]
        curr_close = close[i]
        
        if current_pos != 0:
            if current_pos == 1:
                if df["low"].iloc[i] <= stop_loss or df["high"].iloc[i] >= take_profit:
                    exit_price = take_profit if df["high"].iloc[i] >= take_profit else stop_loss
                    trades.append({
                        "type": "LONG",
                        "entry_time": df["timestamp"].iloc[entry_idx],
                        "entry_price": entry_price,
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "pnl": exit_price - entry_price
                    })
                    current_pos = 0
            elif current_pos == -1:
                if df["high"].iloc[i] >= stop_loss or df["low"].iloc[i] <= take_profit:
                    exit_price = take_profit if df["low"].iloc[i] <= take_profit else stop_loss
                    trades.append({
                        "type": "SHORT",
                        "entry_time": df["timestamp"].iloc[entry_idx],
                        "entry_price": entry_price,
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "pnl": entry_price - exit_price
                    })
                    current_pos = 0
        
        if current_pos == 0:
            if (curr_close > ema50[i] and ema20[i] > ema20[i-1] and
                curr_close > (ema20[i] + 0.1 * atr[i]) and close[i-1] <= ema20[i-1] and rsi[i] < 70):
                current_pos = 1
                entry_idx = i
                entry_price = curr_close
                stop_loss = entry_price - (1.0 * atr[i])
                take_profit = entry_price + (2.0 * atr[i])
                print(f"[LONG] {timestamp} | Close: {curr_close:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
            elif (curr_close < ema50[i] and ema20[i] < ema20[i-1] and
                  curr_close < (ema20[i] - 0.1 * atr[i]) and close[i-1] >= ema20[i-1] and rsi[i] > 30):
                current_pos = -1
                entry_idx = i
                entry_price = curr_close
                stop_loss = entry_price + (1.0 * atr[i])
                take_profit = entry_price - (2.0 * atr[i])
                print(f"[SHORT] {timestamp} | Close: {curr_close:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
        
        pos[i] = current_pos
    
    print(f"\n=== {symbol} RESULTS ===")
    print(f"Total trades: {len(trades)}")
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (wins / len(trades)) * 100 if len(trades) > 0 else 0
        print(f"Total PnL: {total_pnl:.2f} | Wins: {wins} ({win_rate:.1f}%)")
        for t in trades[-10:]:
            print(f"{t['type']} | {t['entry_time']} → {t['exit_time']} | PnL: {t['pnl']:.2f}")
    return df

def main():
    symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT"]
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Backtesting {symbol}")
        print('='*80)
        df = fetch_binance_ohlcv(symbol, "1h", 200)
        backtest_improved_strategy(df, symbol)

if __name__ == "__main__":
    main()
