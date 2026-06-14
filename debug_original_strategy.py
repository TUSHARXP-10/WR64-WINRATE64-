
"""
Debug script to run original strategy on real Binance data
"""
import numpy as np
import pandas as pd
import requests

# --------------------------
# Helper Functions (EXACT)
# --------------------------
def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average"""
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
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate RSI"""
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
    """Fetch OHLCV data from Binance public API"""
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
        
        print(f"Fetching {limit} bars...")
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

def signal(df: pd.DataFrame) -> pd.Series:
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
    ema300 = calculate_ema(close, 300)
    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, 14)

    # Track position and exit levels
    pos = np.zeros(n)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    
    trades = []

    for i in range(300, n):
        timestamp = df["timestamp"].iloc[i]
        curr_close = close[i]
        
        # First check if we need to exit current position
        if current_pos != 0:
            # Check stop loss and take profit
            if current_pos == 1:  # Long position
                if df["low"].iloc[i] <= stop_loss or df["high"].iloc[i] >= take_profit:
                    # Exit
                    exit_price = take_profit if df["high"].iloc[i] >= take_profit else stop_loss
                    pnl = exit_price - entry_price
                    trades.append({
                        "type": "LONG",
                        "entry_time": df["timestamp"].iloc[entry_idx],
                        "entry_price": entry_price,
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    })
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
            elif current_pos == -1:  # Short position
                if df["high"].iloc[i] >= stop_loss or df["low"].iloc[i] <= take_profit:
                    # Exit
                    exit_price = take_profit if df["low"].iloc[i] <= take_profit else stop_loss
                    pnl = entry_price - exit_price
                    trades.append({
                        "type": "SHORT",
                        "entry_time": df["timestamp"].iloc[entry_idx],
                        "entry_price": entry_price,
                        "exit_time": timestamp,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    })
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan

        # If flat, look for new entry
        if current_pos == 0:
            # Long entry: uptrend (close > EMA300), EMA20 crossover with confirmation
            if (curr_close > ema300[i] and
                ema20[i] > ema20[i-1] and
                curr_close > (ema20[i] + 0.3 * atr[i]) and
                close[i-1] <= ema20[i-1] and
                rsi[i] < 62):
                current_pos = 1
                entry_idx = i
                entry_price = curr_close
                stop_loss = entry_price - (0.7 * atr[i])
                take_profit = entry_price + (2.1 * atr[i])
                print(f"[SIGNAL] {timestamp} | LONG | Close: {curr_close:.2f} | EMA20: {ema20[i]:.2f} | EMA300: {ema300[i]:.2f} | ATR: {atr[i]:.2f} | RSI: {rsi[i]:.2f}")
                
            # Short entry: downtrend (close < EMA300), EMA20 crossunder with confirmation
            elif (curr_close < ema300[i] and
                  ema20[i] < ema20[i-1] and
                  curr_close < (ema20[i] - 0.3 * atr[i]) and
                  close[i-1] >= ema20[i-1] and
                  rsi[i] > 38):
                current_pos = -1
                entry_idx = i
                entry_price = curr_close
                stop_loss = entry_price + (0.7 * atr[i])
                take_profit = entry_price - (2.1 * atr[i])
                print(f"[SIGNAL] {timestamp} | SHORT | Close: {curr_close:.2f} | EMA20: {ema20[i]:.2f} | EMA300: {ema300[i]:.2f} | ATR: {atr[i]:.2f} | RSI: {rsi[i]:.2f}")

        pos[i] = current_pos
    
    # Print trades
    print("\n" + "="*80)
    print(f"Total trades: {len(trades)}")
    print("="*80)
    for trade in trades:
        print(f"{trade['type']} | {trade['entry_time']} → {trade['exit_time']} | Entry: {trade['entry_price']:.2f} | Exit: {trade['exit_price']:.2f} | PnL: {trade['pnl']:.2f}")
    total_pnl = sum(t['pnl'] for t in trades)
    print(f"\nTotal PnL: {total_pnl:.2f}")
    
    return pd.Series(pos, index=df.index)

def main():
    print("="*80)
    print("DEBUGGING ORIGINAL STRATEGY ON REAL BINANCE DATA")
    print("="*80)
    
    # Fetch data for BTCUSDT, BNBUSDT, ETHUSDT
    symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT"]
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Analyzing {symbol}")
        print("="*80)
        df = fetch_binance_ohlcv(symbol, "1h", 500)
        print(f"Fetched {len(df)} bars (from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")
        signals = signal(df)

if __name__ == "__main__":
    main()
