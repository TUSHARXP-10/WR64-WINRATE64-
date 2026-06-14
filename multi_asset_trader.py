
"""
WR64-WINRATE64 - MULTI-ASSET TRADER (BTC, BNB, ETH)
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

# Helper functions
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

def get_liquidation_buffer(leverage):
    return max(1e-9, (1.0 / leverage) - 0.005)  # 0.5% maintenance margin

def generate_signal(df, leverage):
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
    liq_buffer = get_liquidation_buffer(leverage)
    sl_buffer = liq_buffer * 0.45
    
    for i in range(15, n):
        if current_pos != 0:
            if current_pos == 1:
                if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit:
                    current_pos = 0
            elif current_pos == -1:
                if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit:
                    current_pos = 0
        
        if current_pos == 0:
            if ema5[i] > ema15[i] and ema5[i-1] <= ema15[i-1] and 20 < rsi.iloc[i] < 80:
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (sl_buffer * entry_price)
                take_profit = entry_price + (sl_buffer * entry_price * 1.9)
            elif ema5[i] < ema15[i] and ema5[i-1] >= ema15[i-1] and 20 < rsi.iloc[i] < 80:
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (sl_buffer * entry_price)
                take_profit = entry_price - (sl_buffer * entry_price * 1.9)
        
        pos[i] = current_pos
    
    return {
        'current_signal': pos[-1],
        'previous_signal': pos[-2],
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }

def get_historical_data(client, symbol, interval, lookback):
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

    all_data.reverse()

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def get_current_position(client, symbol):
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if float(pos['positionAmt']) != 0:
                return {
                    'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT',
                    'size': abs(float(pos['positionAmt'])),
                    'entry_price': float(pos['entryPrice']),
                    'unrealized_pnl': float(pos['unRealizedProfit'])
                }
        return None
    except Exception as e:
        logging.error(f"Error getting {symbol} position: {e}")
        return None

def calculate_position_size(client, symbol, entry_price, stop_loss, risk_per_trade):
    try:
        account = client.futures_account()
        balance = float([b for b in account['assets'] if b['asset'] == 'USDT'][0]['walletBalance'])
        risk_amount = balance * risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        position_size = (risk_amount / stop_distance) * 20  # 20x leverage
        
        # Cap at 25% of max possible position per asset (to spread across 3 assets)
        max_possible_size = (balance * 20 / entry_price) * 0.25
        position_size = min(position_size, max_possible_size)
        
        symbol_info = client.futures_exchange_info()
        symbol_precision = next((s['quantityPrecision'] for s in symbol_info['symbols'] if s['symbol'] == symbol), 0)
        position_size = round(position_size, symbol_precision)
        
        return position_size
    except Exception as e:
        logging.error(f"Error calculating {symbol} position size: {e}")
        return 0

def enter_position(client, symbol, side, entry_price, stop_loss, take_profit, position_size):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=20)
        
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=position_size
        )
        logging.info(f"=== {symbol} ORDER PLACED: {side} {position_size} @ ~{entry_price:.2f} ===")
        
        sl_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == Client.SIDE_BUY else 'BUY',
            type='STOP_MARKET',
            stopPrice=stop_loss,
            quantity=position_size,
            closePosition=True
        )
        logging.info(f"{symbol} Stop Loss placed at {stop_loss:.2f}")
        
        tp_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == Client.SIDE_BUY else 'BUY',
            type='TAKE_PROFIT_MARKET',
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
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multi_asset_trades.log'),
            logging.StreamHandler()
        ]
    )
    
    client = Client(
        api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
        api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
        testnet=True
    )
    
    # Configuration
    SYMBOLS = ['BTCUSDT', 'BNBUSDT', 'ETHUSDT']
    INTERVAL = Client.KLINE_INTERVAL_1HOUR
    LOOKBACK = 100
    SLEEP_TIME = 60  # Check every minute
    RISK_PER_TRADE = 0.015  # 1.5% risk per trade (smaller for multi-asset)
    LEVERAGE = 20
    
    logging.info("="*80)
    logging.info(f"{'WR64-WINRATE64 - MULTI-ASSET TRADER':^80}")
    logging.info("="*80)
    logging.info(f"Trading: {', '.join(SYMBOLS)}")
    logging.info(f"Leverage: {LEVERAGE}x | Risk per trade: {RISK_PER_TRADE*100:.1f}%")
    logging.info("="*80)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"\n=== MARKET CHECK @ {current_time} ===")
            
            for symbol in SYMBOLS:
                logging.info(f"\n--- {symbol} ---")
                
                # Get data
                df = get_historical_data(client, symbol, INTERVAL, LOOKBACK)
                if len(df) < 15:
                    logging.warning(f"Not enough data for {symbol}")
                    continue
                
                # Generate signal
                signal_info = generate_signal(df, LEVERAGE)
                current_signal = signal_info['current_signal']
                previous_signal = signal_info['previous_signal']
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
                        entry_price = df['close'].iloc[-1]
                        liq_buffer = get_liquidation_buffer(LEVERAGE)
                        sl_buffer = liq_buffer * 0.45
                        stop_loss = entry_price - (sl_buffer * entry_price)
                        take_profit = entry_price + (sl_buffer * entry_price * 1.9)
                        
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
                        entry_price = df['close'].iloc[-1]
                        liq_buffer = get_liquidation_buffer(LEVERAGE)
                        sl_buffer = liq_buffer * 0.45
                        stop_loss = entry_price + (sl_buffer * entry_price)
                        take_profit = entry_price - (sl_buffer * entry_price * 1.9)
                        
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
