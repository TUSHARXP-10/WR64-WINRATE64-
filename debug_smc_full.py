
"""SUPER DETAILED DEBUG of SMC Strategy - LENIENT!"""
import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import calculate_ema and calculate_atr from walk_forward
from walk_forward import calculate_ema, calculate_atr

# Load env
load_dotenv()
client = Client(
    api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
    api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
    testnet=True
)
client.API_URL = 'https://testnet.binance.vision/api'
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
client.WEBSITE_URL = 'https://testnet.binance.vision'

# Configuration (LENIENT!)
INTERVAL = Client.KLINE_INTERVAL_1HOUR
LOOKBACK = 1000
ATR_PERIOD = 14
TREND_FAST = 50
TREND_SLOW = 200
FVG_LOOKBACK = 40
SL_ATR = 1.0
TP_ATR = 2.0
FVG_MIN_ATR = 0.0
SWEEP_LOOKBACK = 0
TOTAL_CAPITAL = 5000.0  # Back to $5000
RISK_PER_TRADE = 0.01  # 1% risk

def get_historical_data(symbol, interval, lookback):
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

    # Sort by timestamp (oldest first!)
    all_data.sort(key=lambda k: k[0])
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df.reset_index(drop=True)

def debug_strategy(df, symbol):
    print(f"\n\n{'='*120}")
    print(f"DEBUGGING {symbol} ({INTERVAL} bars, last 100 bars)")
    print(f"{'='*120}")

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)

    ema_fast = calculate_ema(c, TREND_FAST)
    ema_slow = calculate_ema(c, TREND_SLOW)
    atr = calculate_atr(df, ATR_PERIOD)

    bull_zones = []
    bear_zones = []
    start = max(TREND_SLOW, ATR_PERIOD, SWEEP_LOOKBACK) + 3
    print(f"Start index: {start}")

    current_pos = 0
    stop_loss = take_profit = np.nan

    # Collect ALL FVG zones first
    for j in range(start, n):
        if l[j] > h[j-2] and (l[j] - h[j-2]) >= FVG_MIN_ATR * atr[j]:
            bull_zones.append((h[j-2], l[j], j))
        if h[j] < l[j-2] and (l[j-2] - h[j]) >= FVG_MIN_ATR * atr[j]:
            bear_zones.append((h[j], l[j-2], j))

    # Only look at LAST 100 bars for debugging
    debug_start = max(start, n - 100)
    for i in range(debug_start, n):
        # Filter zones for current i
        current_bull_zones = [z for z in bull_zones if i - z[2] <= FVG_LOOKBACK]
        current_bear_zones = [z for z in bear_zones if i - z[2] <= FVG_LOOKBACK]

        # Step 2: Check for new entry if no position
        if current_pos == 0:
            bull_bias = ema_fast[i] > ema_slow[i]
            bear_bias = ema_fast[i] < ema_slow[i]
            
            # Check if price is in FVG (anywhere in bar)
            in_bull_zone = any(z[0] <= h[i] and z[1] >= l[i] for z in current_bull_zones)
            in_bear_zone = any(z[0] <= h[i] and z[1] >= l[i] for z in current_bear_zones)

            # Check lenient patterns (last 3 bars)
            bull_pattern = False
            bear_pattern = False
            for offset in range(0, 3):
                k = i - offset
                if k < 2:
                    continue
                
                # Lenient hammer/shooting star
                body_size = abs(c[k] - o[k])
                total_range = h[k] - l[k]
                lower_wick = min(o[k], c[k]) - l[k]
                upper_wick = h[k] - max(o[k], c[k])
                
                if total_range > 0:
                    if lower_wick >= body_size * 1.2 and upper_wick <= body_size * 0.5:
                        bull_pattern = True
                        break
                    if upper_wick >= body_size * 1.2 and lower_wick <= body_size * 0.5:
                        bear_pattern = True
                        break
                
                # Lenient engulfing
                if k >= 1:
                    if c[k] > o[k] and c[k-1] < o[k-1] and o[k] <= c[k-1] and c[k] >= o[k-1]:
                        bull_pattern = True
                        break
                    if c[k] < o[k] and c[k-1] > o[k-1] and o[k] >= c[k-1] and c[k] <= o[k-1]:
                        bear_pattern = True
                        break

            swept_low = swept_high = True

            # Print current bar info
            print(f"\n--- Bar {i} | Price: O={o[i]:.2f}, H={h[i]:.2f}, L={l[i]:.2f}, C={c[i]:.2f}")
            print(f"    EMA Fast/Slow: {ema_fast[i]:.2f} / {ema_slow[i]:.2f} | Bias: {'BULL' if bull_bias else 'BEAR' if bear_bias else 'NONE'}")
            print(f"    ATR: {atr[i]:.2f}")
            print(f"    FVG ZONES - Bull: {len(current_bull_zones)}, Bear: {len(current_bear_zones)}")
            print(f"    In Bull FVG: {in_bull_zone}, In Bear FVG: {in_bear_zone}")
            print(f"    Patterns - Bull: {bull_pattern}, Bear: {bear_pattern}")
            print(f"    Full Signal Check:")
            print(f"      LONG?: BullBias={bull_bias} AND InBullZone={in_bull_zone} AND BullPattern={bull_pattern} = {bull_bias and in_bull_zone and bull_pattern}")
            print(f"      SHORT?: BearBias={bear_bias} AND InBearZone={in_bear_zone} AND BearPattern={bear_pattern} = {bear_bias and in_bear_zone and bear_pattern}")

            # Check for actual signal
            if bull_bias and in_bull_zone and bull_pattern and swept_low:
                print(f"\n{'!'*60}")
                print(f"!!! LONG SIGNAL AT BAR {i} !!!")
                entry = c[i]
                stop_loss = min(l[i], l[i-1]) - 0.1 * atr[i]
                if stop_loss >= entry:
                    stop_loss = entry - SL_ATR * atr[i]
                take_profit = entry + TP_ATR * atr[i]
                print(f"Entry: {entry:.2f}, Stop: {stop_loss:.2f}, TP: {take_profit:.2f}")
                print(f"{'!'*60}\n")
                current_pos = 1
            elif bear_bias and in_bear_zone and bear_pattern and swept_high:
                print(f"\n{'!'*60}")
                print(f"!!! SHORT SIGNAL AT BAR {i} !!!")
                entry = c[i]
                stop_loss = max(h[i], h[i-1]) + 0.1 * atr[i]
                if stop_loss <= entry:
                    stop_loss = entry + SL_ATR * atr[i]
                take_profit = entry - TP_ATR * atr[i]
                print(f"Entry: {entry:.2f}, Stop: {stop_loss:.2f}, TP: {take_profit:.2f}")
                print(f"{'!'*60}\n")
                current_pos = -1

    print(f"\n--- Final Position: {'LONG' if current_pos == 1 else 'SHORT' if current_pos == -1 else 'NONE'} ---")

# Run debug for all symbols
for sym in ["ETHUSDT", "BTCUSDT", "BNBUSDT"]:
    df = get_historical_data(sym, INTERVAL, LOOKBACK)
    debug_strategy(df, sym)

