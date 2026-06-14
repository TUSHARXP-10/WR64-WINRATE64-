"""
WR64-WINRATE64 - Original Strategy Live Trader (Binance Testnet)
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

# Replicate helper functions from btc_backtest.py
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

def get_liquidation_buffer(config):
    """Calculate liquidation buffer as a percentage of price"""
    return max(1e-9, (1.0 / config['leverage']) - config['maint_margin'])

def signal(df, config):
    """
    Original Strategy from btc_backtest.py - Optimized trend-following!
    """
    close = df["close"].values
    n = len(df)
    
    # Calculate indicators
    ema20 = calculate_ema(close, 20)
    ema300 = calculate_ema(close, 300)
    atr = calculate_atr(df, 14)
    rsi = calculate_rsi(df, 14)
    
    # Track position and exit levels
    pos = pd.Series(np.zeros(n), index=df.index)
    current_pos = 0
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    
    for i in range(300, n):
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
            # Long entry: uptrend (close > EMA300), EMA20 crossover with confirmation
            if (close[i] > ema300[i] and
                ema20[i] > ema20[i-1] and
                close[i] > (ema20[i] + 0.3 * atr[i]) and
                close[i-1] <= ema20[i-1] and
                rsi[i] < 62):
                current_pos = 1
                entry_price = close[i]
                stop_loss = entry_price - (0.7 * atr[i])  # Tight SL
                take_profit = entry_price + (2.1 * atr[i])  # 3:1 R:R
            # Short entry: downtrend (close < EMA300), EMA20 crossunder with confirmation
            elif (close[i] < ema300[i] and
                  ema20[i] < ema20[i-1] and
                  close[i] < (ema20[i] - 0.3 * atr[i]) and
                  close[i-1] >= ema20[i-1] and
                  rsi[i] > 38):
                current_pos = -1
                entry_price = close[i]
                stop_loss = entry_price + (0.7 * atr[i])
                take_profit = entry_price - (2.1 * atr[i])
        
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

# Strategy configuration (matches btc_backtest.py defaults)
CONFIG = {
    'leverage': 20,
    'maint_margin': 0.005,
    'start_equity': 50,  # Starting equity for risk calculations
    'risk_per_trade': 0.05,  # 5% risk per trade
    'symbol': 'BTCUSDT',
    'interval': Client.KLINE_INTERVAL_1HOUR,
    'lookback': 350,  # Need enough data for EMA300
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
    logger.info("Starting Original Strategy from btc_backtest.py")
    logger.info(f"Symbol: {CONFIG['symbol']}")
    logger.info(f"Leverage: {CONFIG['leverage']}x")
    logger.info("="*60)
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"\nChecking market at {current_time}")
            
            # Step 1: Fetch historical data (need 350 for EMA300)
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
                    atr = calculate_atr(df, 14)
                    current_atr = atr[-1]
                    stop_loss = entry_price - (0.7 * current_atr)
                    take_profit = entry_price + (2.1 * current_atr)
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
                    atr = calculate_atr(df, 14)
                    current_atr = atr[-1]
                    stop_loss = entry_price + (0.7 * current_atr)
                    take_profit = entry_price - (2.1 * current_atr)
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
