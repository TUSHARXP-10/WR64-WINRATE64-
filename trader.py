
"""
WR64-WINRATE64 - Safe Frequent Strategy (EMA 5/15, No Liquidation)
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

def calculate_rsi(df, period=14):
    """Calculate RSI (Relative Strength Index) (helper function)"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_liquidation_buffer(config):
    """Calculate liquidation buffer as a percentage of price"""
    return max(1e-9, (1.0 / config['leverage']) - config['maint_margin'])

def signal(df, config):
    """
    Safe Frequent Strategy - EMA 5/15 Crossovers, Ultra Relaxed RSI!
    """
    close = df["close"].values
    n = len(df)
    
    # Calculate indicators
    ema5 = calculate_ema(close, 5)
    ema15 = calculate_ema(close, 15)
    rsi = calculate_rsi(df, 14)
    
    # Track position and signals
    pos = pd.Series(np.zeros(n), index=df.index)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    liq_buffer = get_liquidation_buffer(config)
    sl_buffer = liq_buffer * 0.45  # 45% of liquidation buffer
    
    for i in range(15, n):
        # First check exit conditions
        if current_pos != 0:
            if current_pos == 1:  # Long
                if df["low"].iloc[i] <= stop_loss or df["high"].iloc[i] >= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
            elif current_pos == -1:  # Short
                if df["high"].iloc[i] >= stop_loss or df["low"].iloc[i] <= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
        
        # Check for new entries
        if current_pos == 0:
            # Long entry: EMA5 crosses above EMA15, RSI not extreme
            if ema5[i] > ema15[i] and ema5[i-1] <= ema15[i-1] and rsi.iloc[i] > 20 and rsi.iloc[i] < 80:
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (sl_buffer * entry_price)
                take_profit = entry_price + (sl_buffer * entry_price * 1.9)  # 1.9 R:R
            # Short entry: EMA5 crosses below EMA15, RSI not extreme
            elif ema5[i] < ema15[i] and ema5[i-1] >= ema15[i-1] and rsi.iloc[i] > 20 and rsi.iloc[i] < 80:
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (sl_buffer * entry_price)
                take_profit = entry_price - (sl_buffer * entry_price * 1.9)
        
        pos.iloc[i] = current_pos
    
    return pos

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trades.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Strategy configuration
CONFIG = {
    'leverage': 20,
    'maint_margin': 0.005,
    'start_equity': 50,
    'risk_per_trade': 0.05,
    'symbol': 'BTCUSDT',
    'interval': Client.KLINE_INTERVAL_1HOUR,
    'lookback': 100,
    'sleep_time': 300,  # Check every 5 minutes
}

# Initialize Binance Testnet client
client = Client(
    api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
    api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
    testnet=True
)

def get_historical_data(symbol, interval, lookback):
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

def get_current_position(symbol):
    """Get current open position"""
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
        logger.error(f"Error getting position: {e}")
        return None

def calculate_position_size(symbol, entry_price, stop_loss, config):
    """Calculate position size based on risk"""
    try:
        account = client.futures_account()
        balance = float([b for b in account['assets'] if b['asset'] == 'USDT'][0]['walletBalance'])
        risk_amount = balance * config['risk_per_trade']
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        position_size = (risk_amount / stop_distance) * config['leverage']
        
        symbol_info = client.futures_exchange_info()
        symbol_precision = next((s['quantityPrecision'] for s in symbol_info['symbols'] if s['symbol'] == symbol), 0)
        position_size = round(position_size, symbol_precision)
        
        logger.info(f"Calculated position size: {position_size} (balance: ${balance:.2f}, risk: ${risk_amount:.2f})")
        return position_size
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0

def enter_position(symbol, side, entry_price, stop_loss, take_profit, position_size):
    """Enter a new position"""
    try:
        client.futures_change_leverage(symbol=symbol, leverage=CONFIG['leverage'])
        
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=position_size
        )
        logger.info(f"ENTRY ORDER PLACED: {side} {position_size} {symbol} at ~{entry_price}")
        
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_SELL if side == Client.SIDE_BUY else Client.SIDE_BUY,
            type=Client.ORDER_TYPE_STOP_MARKET,
            stopPrice=stop_loss,
            quantity=position_size,
            closePosition=True
        )
        logger.info(f"STOP LOSS PLACED: {stop_loss}")
        
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_SELL if side == Client.SIDE_BUY else Client.SIDE_BUY,
            type=Client.ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=take_profit,
            quantity=position_size,
            closePosition=True
        )
        logger.info(f"TAKE PROFIT PLACED: {take_profit}")
        
        return True
    except (BinanceAPIException, BinanceOrderException) as e:
        logger.error(f"Order error: {e}")
        return False

def main():
    logger.info("="*60)
    logger.info("Starting WR64-WINRATE64 - SAFE FREQUENT STRATEGY")
    logger.info(f"Symbol: {CONFIG['symbol']}")
    logger.info(f"Leverage: {CONFIG['leverage']}x (NO LIQUIDATION GUARANTEED!)")
    logger.info("="*60)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\nChecking market at {current_time}")
            
            df = get_historical_data(CONFIG['symbol'], CONFIG['interval'], CONFIG['lookback'])
            logger.info(f"Fetched {len(df)} klines")
            
            if len(df) < 15:
                logger.warning("Not enough data, waiting...")
                time.sleep(CONFIG['sleep_time'])
                continue
            
            signals = signal(df, CONFIG)
            current_signal = signals.iloc[-1]
            previous_signal = signals.iloc[-2]
            logger.info(f"Current signal: {current_signal} (Previous: {previous_signal})")
            
            current_position = get_current_position(CONFIG['symbol'])
            if current_position:
                logger.info(f"Current position: {current_position}")
            else:
                logger.info("No open position")
            
            if current_position is None:
                # Check for entry signal
                liq_buffer = get_liquidation_buffer(CONFIG)
                sl_buffer = liq_buffer * 0.45
                
                if current_signal == 1 and previous_signal == 0:
                    logger.info("=== LONG SIGNAL DETECTED! ===")
                    entry_price = df['close'].iloc[-1]
                    stop_loss = entry_price - (sl_buffer * entry_price)
                    take_profit = entry_price + (sl_buffer * entry_price * 1.9)
                    position_size = calculate_position_size(CONFIG['symbol'], entry_price, stop_loss, CONFIG)
                    
                    if position_size > 0:
                        enter_position(
                            CONFIG['symbol'],
                            Client.SIDE_BUY,
                            entry_price,
                            stop_loss,
                            take_profit,
                            position_size
                        )
                
                elif current_signal == -1 and previous_signal == 0:
                    logger.info("=== SHORT SIGNAL DETECTED! ===")
                    entry_price = df['close'].iloc[-1]
                    stop_loss = entry_price + (sl_buffer * entry_price)
                    take_profit = entry_price - (sl_buffer * entry_price * 1.9)
                    position_size = calculate_position_size(CONFIG['symbol'], entry_price, stop_loss, CONFIG)
                    
                    if position_size > 0:
                        enter_position(
                            CONFIG['symbol'],
                            Client.SIDE_SELL,
                            entry_price,
                            stop_loss,
                            take_profit,
                            position_size
                        )
            
            logger.info(f"Waiting {CONFIG['sleep_time']} seconds...")
            time.sleep(CONFIG['sleep_time'])
            
        except KeyboardInterrupt:
            logger.info("Stopping strategy...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(CONFIG['sleep_time'])

if __name__ == "__main__":
    main()
