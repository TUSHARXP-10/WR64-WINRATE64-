
from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()

client = Client(
    api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
    api_secret=os.getenv("BINANCE_TESTNET_API_SECRET"),
    testnet=True
)

# Explicitly set the futures testnet URL
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
client.API_URL = 'https://testnet.binance.vision/api'

print("Testing client.futures_klines...")
try:
    klines = client.futures_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=10)
    print(f"Success! Got {len(klines)} klines!")
    print(f"Last kline close: {float(klines[-1][4]):.2f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting client.futures_exchange_info...")
try:
    info = client.futures_exchange_info()
    print("Success! Got exchange info!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
