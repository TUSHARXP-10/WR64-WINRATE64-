"""
WR64-WINRATE64 - 20x Scalping Strategy Live Trader (Binance Testnet)
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

# Import our strategy from btc_backtest_2o.py
# First, let's replicate the necessary functions here so it's standalone

def calculate_ema(series, period):
    """Calculate Exponential Moving Average (EMA)"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index (RSI)"""
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
    Version 12.0 - 20x ETH-FIXED (Ultra Relaxed RSI + Higher R:R)
    """
    close = df['close'].values
    n = len(df)
    
    # Calculate indicators
    ema5 = calculate_ema(df['close'], 5)
    ema15 = calculate_ema(df['close'], 15)
    rsi = calculate_rsi(df, 14)
    
    # Track position and signals
    pos = np.zeros(n)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    liq_buffer = get_liquidation_buffer(config)
    sl_buffer = liq_buffer * 0.45
    
    for i in range(1, n):
        # Check exit conditions first
        if current_pos != 0:
            if current_pos == 1:
                if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
            elif current_pos == -1:
                if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit:
                    current_pos = 0
                    entry_price = np.nan
                    stop_loss = np.nan
                    take_profit = np.nan
        
        # Check entry conditions
        if current_pos == 0:
            # Long entry
            if ema5.iloc[i] > ema15.iloc[i] and ema5.iloc[i-1] <= ema15.iloc[i-1] and rsi.iloc[i] > 20 and rsi.iloc[i] < 80:
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (sl_buffer * entry_price)
                take_profit = entry_price + (sl_buffer * entry_price * 1.9)
            # Short entry
            elif ema5.iloc[i] < ema15.iloc[i] and ema5.iloc[i-1] >= ema15.iloc[i-1] and rsi.iloc[i] > 20 and rsi.iloc[i] < 80:
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (sl_buffer * entry_price)
                take_profit = entry_price - (sl_buffer * entry_price * 1.9)
        
        pos[i] = current_pos
    
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
    'start_equity': 100,  # Starting equity for risk calculations
    'risk_per_trade': 0.05,  # 5% risk per trade
    'symbol': 'BTCUSDT',
    'interval': Client.KLINE_INTERVAL_1HOUR,
    'lookback': 100,  # Lookback period for indicators
    'sleep_time': 300,  # Check every 5 minutes (300 seconds)
}

# Initialize Binance Testnet client
client = Client(
    api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
    api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
    testnet=True
)

def get_historical_data(symbol, interval, lookback):
    """Fetch historical klines from Binance Testnet"""
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
    
    # Convert to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)
    
    # Convert timestamp to datetime
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
        # Get account balance
        account = client.futures_account()
        balance = float([b for b in account['assets'] if b['asset'] == 'USDT'][0]['walletBalance'])
        
        # Calculate risk amount
        risk_amount = balance * config['risk_per_trade']
        
        # Calculate stop loss distance
        stop_distance = abs(entry_price - stop_loss)
        
        # Calculate position size
        if stop_distance == 0:
            return 0
        
        position_size = (risk_amount / stop_distance) * config['leverage']
        
        # Get symbol info to determine quantity precision
        symbol_info = client.futures_exchange_info()
        symbol_precision = next((s['quantityPrecision'] for s in symbol_info['symbols'] if s['symbol'] == symbol), 0)
        
        # Round to correct precision
        position_size = round(position_size, symbol_precision)
        
        logger.info(f"Calculated position size: {position_size}")
        return position_size
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0

def enter_position(symbol, side, entry_price, stop_loss, take_profit, position_size):
    """Enter a new position"""
    try:
        # Set leverage
        client.futures_change_leverage(symbol=symbol, leverage=CONFIG['leverage'])
        
        # Place entry order
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=position_size
        )
        
        logger.info(f"Entry order placed: {side} {position_size} {symbol}")
        
        # Place stop loss order
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_SELL if side == Client.SIDE_BUY else Client.SIDE_BUY,
            type=Client.ORDER_TYPE_STOP_MARKET,
            stopPrice=stop_loss,
            quantity=position_size,
            closePosition=True
        )
        
        logger.info(f"Stop loss order placed at {stop_loss}")
        
        # Place take profit order
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_SELL if side == Client.SIDE_BUY else Client.SIDE_BUY,
            type=Client.ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=take_profit,
            quantity=position_size,
            closePosition=True
        )
        
        logger.info(f"Take profit order placed at {take_profit}")
        
        return True
    except (BinanceAPIException, BinanceOrderException) as e:
        logger.error(f"Order error: {e}")
        return False

def main():
    logger.info("="*60)
    logger.info("Starting WR64-WINRATE64 Strategy on Binance Testnet")
    logger.info(f"Symbol: {CONFIG['symbol']}")
    logger.info(f"Leverage: {CONFIG['leverage']}x")
    logger.info("="*60)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\nChecking market at {current_time}")
            
            # Step 1: Fetch historical data
            df = get_historical_data(CONFIG['symbol'], CONFIG['interval'], CONFIG['lookback'])
            if len(df) < CONFIG['lookback']:
                logger.warning("Not enough data, waiting...")
                time.sleep(CONFIG['sleep_time'])
                continue
            
            # Step 2: Generate signals
            signals = signal(df, CONFIG)
            current_signal = signals.iloc[-1]
            previous_signal = signals.iloc[-2]
            
            logger.info(f"Current signal: {current_signal} (Previous: {previous_signal})")
            
            # Step 3: Get current position
            current_position = get_current_position(CONFIG['symbol'])
            
            # Step 4: Execute trades based on signals
            if current_position is None:
                # No position, check for entry
                if current_signal == 1 and previous_signal == 0:
                    logger.info("Long signal detected!")
                    entry_price = df['close'].iloc[-1]
                    liq_buffer = get_liquidation_buffer(CONFIG)
                    sl_buffer = liq_buffer * 0.45
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
                    logger.info("Short signal detected!")
                    entry_price = df['close'].iloc[-1]
                    liq_buffer = get_liquidation_buffer(CONFIG)
                    sl_buffer = liq_buffer * 0.45
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
            
            else:
                logger.info(f"Current position: {current_position}")
            
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
