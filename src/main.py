"""
Main Application
Trading Bot with AI-powered signal generation
"""

import asyncio
from typing import Dict, List
from datetime import datetime
from loguru import logger
import signal as sys_signal

from config import config
from utils.logger import setup_logger
from utils.performance_tracker import PerformanceTracker

from data_collector.exchange_connector import ExchangeConnector
from data_collector.market_data_manager import MarketDataManager

from ai_engine.market_analyzer import MarketAnalyzer

from signal_generator.signal_generator import SignalGenerator
from signal_generator.signal_filter import SignalFilter
from signal_generator.risk_manager import RiskManager

from telegram_bot.telegram_bot import TelegramBot


class TradingBot:
    """Main trading bot orchestrator"""

    def __init__(self):
        """Initialize Trading Bot"""
        # Setup logging
        setup_logger(
            log_level=config.log_level,
            log_file=config.get('logging.log_file', 'logs/trading_bot.log')
        )

        logger.info("=" * 60)
        logger.info("AI Trading Signal Bot Starting...")
        logger.info("=" * 60)

        # Performance tracking
        self.performance = PerformanceTracker()

        # Initialize components
        self.exchange = None
        self.market_data_manager = None
        self.analyzer = None
        self.signal_generator = None
        self.telegram_bot = None

        # Control flags
        self.is_running = False
        self.analysis_interval = 60  # seconds

        # Initialize all components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all bot components"""
        try:
            # Exchange connector
            logger.info("Initializing exchange connector...")
            self.exchange = ExchangeConnector(
                exchange_name='binance',
                api_key=config.exchange.binance_api_key,
                api_secret=config.exchange.binance_api_secret,
                testnet=not config.is_production
            )

            # Market data manager
            logger.info("Initializing market data manager...")
            self.market_data_manager = MarketDataManager(
                exchange=self.exchange,
                symbols=config.symbols,
                timeframes=config.timeframes_list,
                update_interval=60
            )

            # AI Analyzer
            logger.info("Initializing AI analyzer...")
            self.analyzer = MarketAnalyzer(enable_training=False)

            # Try to load pre-trained models
            try:
                self.analyzer.load_models('models')
                logger.info("Loaded pre-trained models")
            except Exception as e:
                logger.warning(f"Could not load models: {e}. Using default models.")

            # Signal components
            logger.info("Initializing signal generator...")
            signal_filter = SignalFilter(
                max_signals_per_day=config.get('signals.risk_management.max_signals_per_day', 10),
                max_signals_per_pair=config.get('signals.risk_management.max_signals_per_pair', 3)
            )

            risk_manager = RiskManager(
                default_risk_reward=config.get('signals.risk_management.min_risk_reward', 2.0),
                atr_multiplier_sl=config.get('signals.risk_management.stop_loss_atr_multiplier', 1.5)
            )

            self.signal_generator = SignalGenerator(
                analyzer=self.analyzer,
                signal_filter=signal_filter,
                risk_manager=risk_manager,
                min_confidence=config.ai.confidence_threshold,
                min_strength=config.ai.min_signal_score
            )

            # Telegram bot
            logger.info("Initializing Telegram bot...")
            self.telegram_bot = TelegramBot(
                bot_token=config.telegram.bot_token,
                channel_id=config.telegram.channel_id,
                enable_charts=config.get('telegram.include_charts', True)
            )

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    async def start(self):
        """Start the trading bot"""
        try:
            self.is_running = True

            logger.info("Starting Trading Bot...")

            # Test Telegram connection
            if self.telegram_bot:
                telegram_ok = await self.telegram_bot.test_connection()
                if not telegram_ok:
                    logger.warning("Telegram connection test failed, but continuing...")

            # Send startup message
            await self.telegram_bot.send_message(
                "🚀 **AI Trading Bot Started**\n\n"
                f"Monitoring {len(config.symbols)} symbols across {len(config.timeframes_list)} timeframes.\n\n"
                f"Environment: {config.environment}\n"
                f"Min Confidence: {config.ai.confidence_threshold:.0%}\n"
                f"Min Signal Strength: {config.ai.min_signal_score}/100"
            )

            # Start market data collection
            logger.info("Starting market data collection...")
            await self.market_data_manager.start()

            # Wait for initial data
            logger.info("Waiting for initial data collection...")
            await asyncio.sleep(30)

            # Start main loop
            logger.info("Starting analysis loop...")
            await self._run_main_loop()

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Trading Bot...")

        self.is_running = False

        # Stop market data collection
        if self.market_data_manager:
            await self.market_data_manager.stop()

        # Send shutdown message
        if self.telegram_bot:
            await self.telegram_bot.send_message(
                "🛑 **AI Trading Bot Stopped**\n\n"
                f"Total signals generated: {self.performance.signals_generated}\n"
                f"Uptime: {self.performance.get_statistics()['uptime_hours']:.2f} hours"
            )

        logger.info("Trading Bot stopped")

    async def _run_main_loop(self):
        """Main analysis and signal generation loop"""
        logger.info("Main loop started")

        while self.is_running:
            try:
                # Analyze all symbols
                for symbol in config.symbols:
                    await self._analyze_and_generate_signal(symbol)

                # Wait before next iteration
                await asyncio.sleep(self.analysis_interval)

                # Periodic tasks (every hour)
                if datetime.utcnow().minute == 0:
                    await self._run_periodic_tasks()

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.performance.record_error("main_loop", str(e))
                await asyncio.sleep(10)

    async def _analyze_and_generate_signal(self, symbol: str):
        """
        Analyze market and generate signal for a symbol

        Args:
            symbol: Trading pair to analyze
        """
        try:
            start_time = datetime.utcnow()

            # Get multi-timeframe data
            mtf_data = self.market_data_manager.get_multi_timeframe_data(symbol, limit=200)

            if not mtf_data or all(df.empty for df in mtf_data.values()):
                logger.debug(f"{symbol}: No data available yet")
                return

            # Analyze all timeframes
            analyses = self.analyzer.analyze_multi_timeframe(mtf_data, symbol)

            # Get current price
            current_price = self.market_data_manager.get_current_price(symbol)

            if not current_price:
                logger.warning(f"{symbol}: Could not get current price")
                return

            # Generate signal
            signal = self.signal_generator.generate_signal(
                symbol=symbol,
                multi_tf_analyses=analyses,
                current_price=current_price
            )

            # If signal generated, send to Telegram
            if signal:
                logger.info(f"Signal generated for {symbol}: {signal.signal_type}")

                # Prepare market data for chart
                chart_data = {
                    'data': mtf_data.get(signal.timeframe, None)
                }

                # Send to Telegram
                await self.telegram_bot.send_signal(signal, chart_data)

                # Record performance
                self.performance.record_signal(symbol, signal.signal_type)

            # Record analysis time
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.performance.record_analysis_time(elapsed)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            self.performance.record_error("analysis", str(e), symbol)

    async def _run_periodic_tasks(self):
        """Run periodic maintenance tasks"""
        try:
            # Send performance update
            health = self.performance.get_health_status()
            logger.info(f"Health check: {health['status']}")

            # Daily summary at midnight UTC
            if datetime.utcnow().hour == 0:
                summary = self.performance.get_daily_summary()
                await self.telegram_bot.send_daily_summary(summary)
                self.performance.reset_daily_metrics()

        except Exception as e:
            logger.error(f"Error in periodic tasks: {e}")

    async def run_backtest(self, symbol: str, days: int = 30):
        """
        Run backtest on historical data

        Args:
            symbol: Trading pair
            days: Number of days to backtest
        """
        logger.info(f"Running backtest for {symbol} over {days} days...")

        # This is a placeholder for backtesting functionality
        # You would implement historical data fetching and signal generation here

        logger.info("Backtest completed")


async def main():
    """Main entry point"""
    bot = TradingBot()

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    sys_signal.signal(sys_signal.SIGINT, signal_handler)
    sys_signal.signal(sys_signal.SIGTERM, signal_handler)

    # Start bot
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await bot.stop()
        raise


if __name__ == "__main__":
    # Run the bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        raise
