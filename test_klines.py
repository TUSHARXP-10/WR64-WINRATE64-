
import os
from dotenv import load_dotenv
from binance.client import Client
import pandas as pd

load_dotenv()

client = Client(
    api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
    api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
    testnet=True
)

print("Testing Binance Testnet get_klines...")
try:
    # Try to get 350 1h klines
    klines = client.get_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=350
    )
    print(f"Number of klines received: {len(klines)}")
    print(f"First kline timestamp: {klines[0][0]}")
    print(f"Last kline timestamp: {klines[-1][0]}")
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    print(f"DataFrame length: {len(df)}")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
