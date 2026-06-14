
"""
Close all open positions on Binance Testnet
"""
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

def main():
    load_dotenv()
    client = Client(
        api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
        api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
        testnet=True
    )
    
    SYMBOLS = ['BTCUSDT', 'BNBUSDT', 'ETHUSDT']
    
    print("Closing any open positions...")
    
    for symbol in SYMBOLS:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            position_size = float(pos['positionAmt'])
            if position_size != 0:
                print(f"Closing {symbol} position: {position_size}")
                
                try:
                    side = Client.SIDE_SELL if position_size > 0 else Client.SIDE_BUY
                    client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=abs(position_size)
                    )
                    print(f"Successfully closed {symbol} position")
                except (BinanceAPIException, BinanceOrderException) as e:
                    print(f"Error closing {symbol} position: {e}")
    
    # Cancel all open orders too
    print("\nCanceling all open orders...")
    for symbol in SYMBOLS:
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
            print(f"Canceled all {symbol} orders")
        except (BinanceAPIException, BinanceOrderException) as e:
            print(f"Error canceling {symbol} orders: {e}")
    
    print("\nAll positions closed and orders canceled!")

if __name__ == "__main__":
    main()
