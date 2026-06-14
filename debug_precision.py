from binance.client import Client
from dotenv import load_dotenv
import os

load_dotenv()
client = Client(os.getenv("BINANCE_TESTNET_API_KEY"), os.getenv("BINANCE_TESTNET_API_SECRET"), testnet=True)

symbols = ["BNBUSDT"]
info = client.futures_exchange_info()

for symbol in symbols:
    symbol_info = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
    if symbol_info:
        print(f"\n{symbol} full info:")
        print(f"  quantityPrecision: {symbol_info['quantityPrecision']}")
        print(f"  All filters: {symbol_info['filters']}")

