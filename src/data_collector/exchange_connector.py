"""
Exchange Connector
Handles connections to multiple cryptocurrency exchanges
"""

import ccxt
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd


class ExchangeConnector:
    """Connects to and manages multiple exchange connections"""

    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
    }

    def __init__(self, exchange_name: str = 'binance', api_key: str = '', api_secret: str = '', testnet: bool = False):
        """
        Initialize exchange connector

        Args:
            exchange_name: Name of the exchange (binance, bybit, etc.)
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            testnet: Whether to use testnet
        """
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange = None

        self._initialize_exchange()

    def _initialize_exchange(self):
        """Initialize the exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)

            config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if self.exchange_name == 'binance' else 'swap',
                }
            }

            if self.testnet:
                config['options']['defaultType'] = 'future'
                if self.exchange_name == 'binance':
                    config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binancefuture.com/fapi/v1',
                            'private': 'https://testnet.binancefuture.com/fapi/v1',
                        }
                    }

            self.exchange = exchange_class(config)
            logger.info(f"Connected to {self.exchange_name} exchange (testnet={self.testnet})")

        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            raise

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if not self.exchange:
                self._initialize_exchange()

            # Fetch OHLCV data
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                None,
                limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add symbol and timeframe columns
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data

        Args:
            symbol: Trading pair

        Returns:
            Ticker data dictionary
        """
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_ticker,
                symbol
            )
            return ticker

        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch order book data

        Args:
            symbol: Trading pair
            limit: Depth of order book

        Returns:
            Order book dictionary
        """
        try:
            order_book = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_order_book,
                symbol,
                limit
            )
            return order_book

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}

    async def fetch_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Fetch recent trades

        Args:
            symbol: Trading pair
            limit: Number of trades to fetch

        Returns:
            List of recent trades
        """
        try:
            trades = await asyncio.get_event_loop().run_in_executor(
                None,
                self.exchange.fetch_trades,
                symbol,
                None,
                limit
            )
            return trades

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return []

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        try:
            markets = self.exchange.load_markets()
            return [symbol for symbol in markets.keys() if '/USDT' in symbol]
        except Exception as e:
            logger.error(f"Error getting supported symbols: {e}")
            return []

    async def close(self):
        """Close exchange connection"""
        try:
            if self.exchange:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.exchange.close
                )
                logger.info(f"Closed {self.exchange_name} connection")
        except Exception as e:
            logger.error(f"Error closing exchange connection: {e}")


class MultiExchangeConnector:
    """Manages connections to multiple exchanges"""

    def __init__(self, exchange_configs: List[Dict[str, Any]]):
        """
        Initialize multi-exchange connector

        Args:
            exchange_configs: List of exchange configuration dictionaries
        """
        self.exchanges: Dict[str, ExchangeConnector] = {}

        for config in exchange_configs:
            name = config.get('name', 'binance')
            api_key = config.get('api_key', '')
            api_secret = config.get('api_secret', '')
            testnet = config.get('testnet', False)

            self.exchanges[name] = ExchangeConnector(
                exchange_name=name,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )

    async def fetch_ohlcv_from_all(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV from all connected exchanges

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Number of candles

        Returns:
            Dictionary mapping exchange name to DataFrame
        """
        tasks = {}
        for name, exchange in self.exchanges.items():
            tasks[name] = exchange.fetch_ohlcv(symbol, timeframe, limit)

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error fetching from {name}: {e}")
                results[name] = pd.DataFrame()

        return results

    async def close_all(self):
        """Close all exchange connections"""
        for name, exchange in self.exchanges.items():
            await exchange.close()
