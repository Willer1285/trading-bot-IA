"""
Market Data Manager
Manages real-time market data collection and storage
"""

import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from .exchange_connector import ExchangeConnector
from .timeframe_aggregator import TimeframeAggregator


class MarketDataManager:
    """Manages real-time collection of market data across multiple symbols and timeframes"""

    def __init__(
        self,
        exchange: ExchangeConnector,
        symbols: List[str],
        timeframes: List[str],
        update_interval: int = 60
    ):
        """
        Initialize Market Data Manager

        Args:
            exchange: Exchange connector instance
            symbols: List of trading pairs to monitor
            timeframes: List of timeframes to track
            update_interval: Update interval in seconds
        """
        self.exchange = exchange
        self.symbols = symbols
        self.timeframes = timeframes
        self.update_interval = update_interval

        # Data storage
        self.market_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Timeframe aggregator
        self.aggregator = TimeframeAggregator()

        # Control flags
        self.is_running = False
        self._tasks: Set[asyncio.Task] = set()

        # Initialize storage
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize data storage structure"""
        for symbol in self.symbols:
            self.market_data[symbol] = {}
            for timeframe in self.timeframes:
                self.market_data[symbol][timeframe] = pd.DataFrame()

        logger.info(f"Initialized storage for {len(self.symbols)} symbols across {len(self.timeframes)} timeframes")

    async def start(self):
        """Start continuous data collection"""
        self.is_running = True
        logger.info("Starting Market Data Manager...")

        # Initial data fetch
        await self._fetch_initial_data()

        # Start update loop for each symbol
        for symbol in self.symbols:
            task = asyncio.create_task(self._update_loop(symbol))
            self._tasks.add(task)

        logger.info(f"Market Data Manager started for {len(self.symbols)} symbols")

    async def stop(self):
        """Stop data collection"""
        self.is_running = False
        logger.info("Stopping Market Data Manager...")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("Market Data Manager stopped")

    async def _fetch_initial_data(self):
        """Fetch initial historical data for all symbols and timeframes"""
        logger.info("Fetching initial historical data...")

        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task = self._fetch_and_store(symbol, timeframe, limit=500)
                tasks.append(task)

        await asyncio.gather(*tasks)
        logger.info("Initial data fetch completed")

    async def _fetch_and_store(self, symbol: str, timeframe: str, limit: int = 100):
        """
        Fetch and store OHLCV data

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Number of candles to fetch
        """
        try:
            df = await self.exchange.fetch_ohlcv(symbol, timeframe, limit)

            if not df.empty:
                # Store or update data
                if self.market_data[symbol][timeframe].empty:
                    self.market_data[symbol][timeframe] = df
                else:
                    # Merge with existing data, avoiding duplicates
                    existing = self.market_data[symbol][timeframe]
                    combined = pd.concat([existing, df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()

                    # Keep only recent data (e.g., last 1000 candles)
                    self.market_data[symbol][timeframe] = combined.tail(1000)

                logger.debug(f"Updated {symbol} {timeframe}: {len(self.market_data[symbol][timeframe])} candles")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")

    async def _update_loop(self, symbol: str):
        """
        Continuous update loop for a symbol

        Args:
            symbol: Trading pair to update
        """
        while self.is_running:
            try:
                # Fetch data for all timeframes
                for timeframe in self.timeframes:
                    await self._fetch_and_store(symbol, timeframe, limit=10)

                # Wait before next update
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                logger.info(f"Update loop cancelled for {symbol}")
                break
            except Exception as e:
                logger.error(f"Error in update loop for {symbol}: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get latest market data

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Number of recent candles (None for all)

        Returns:
            DataFrame with latest data
        """
        try:
            df = self.market_data.get(symbol, {}).get(timeframe, pd.DataFrame())

            if df.empty:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return pd.DataFrame()

            if limit:
                return df.tail(limit)
            return df

        except Exception as e:
            logger.error(f"Error getting latest data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol

        Args:
            symbol: Trading pair

        Returns:
            Current close price or None
        """
        try:
            # Get latest 1m data
            df = self.get_latest_data(symbol, '1m', limit=1)

            if not df.empty:
                return float(df['close'].iloc[-1])
            return None

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_multi_timeframe_data(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for all timeframes of a symbol

        Args:
            symbol: Trading pair
            limit: Number of recent candles per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        result = {}
        for timeframe in self.timeframes:
            result[timeframe] = self.get_latest_data(symbol, timeframe, limit)

        return result

    def get_all_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for all symbols

        Returns:
            Dictionary mapping symbol to current price
        """
        prices = {}
        for symbol in self.symbols:
            price = self.get_current_price(symbol)
            if price:
                prices[symbol] = price

        return prices

    async def get_realtime_ticker(self, symbol: str) -> Dict:
        """
        Get real-time ticker data

        Args:
            symbol: Trading pair

        Returns:
            Ticker data dictionary
        """
        return await self.exchange.fetch_ticker(symbol)

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Get order book data

        Args:
            symbol: Trading pair
            limit: Order book depth

        Returns:
            Order book data
        """
        return await self.exchange.fetch_order_book(symbol, limit)

    def get_data_summary(self) -> Dict:
        """
        Get summary of collected data

        Returns:
            Summary statistics
        """
        summary = {
            'symbols': len(self.symbols),
            'timeframes': len(self.timeframes),
            'is_running': self.is_running,
            'data_points': {}
        }

        for symbol in self.symbols:
            summary['data_points'][symbol] = {}
            for timeframe in self.timeframes:
                df = self.market_data.get(symbol, {}).get(timeframe, pd.DataFrame())
                summary['data_points'][symbol][timeframe] = len(df)

        return summary
