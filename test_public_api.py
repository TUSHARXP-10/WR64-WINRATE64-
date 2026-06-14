
import requests
import pandas as pd

def test_public_klines():
    # Test mainnet first to check connectivity
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 10}
    print("Testing mainnet public klines...")
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"Success! Got {len(data)} klines. Last close: {float(data[-1][4]):.2f}")
    except Exception as e:
        print(f"Mainnet test failed: {e}")
    
    # Test testnet public klines
    print("\nTesting testnet public klines...")
    try:
        url_testnet = "https://testnet.binancefuture.com/fapi/v1/klines"
        response_testnet = requests.get(url_testnet, params=params, timeout=10)
        response_testnet.raise_for_status()
        data_testnet = response_testnet.json()
        print(f"Success! Got {len(data_testnet)} klines. Last close: {float(data_testnet[-1][4]):.2f}")
    except Exception as e:
        print(f"Testnet test failed: {e}")

if __name__ == "__main__":
    test_public_klines()
