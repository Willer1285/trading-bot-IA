"""
Main Application for MT5 Trading Bot
Trading Bot with AI-powered signal generation and automatic execution on MT5
"""

import asyncio
import os
from typing import Dict, List
from datetime import datetime
from loguru import logger
import signal as sys_signal

from config import config
from utils.logger import setup_logger
from utils.performance_tracker import PerformanceTracker

from data_collector.mt5_connector import MT5Connector, MT5OrderExecutor
from data_collector.mt5_market_data_manager import MT5MarketDataManager

from ai_engine.market_analyzer import MarketAnalyzer

from signal_generator.signal_generator import SignalGenerator
from signal_generator.signal_filter import SignalFilter
from signal_generator.risk_manager import RiskManager

from telegram_bot.telegram_bot import TelegramBot


class MT5TradingBot:
    """Main MT5 trading bot orchestrator with automatic execution"""

    def __init__(self):
        """Initialize MT5 Trading Bot"""
        # Setup logging
        setup_logger(
            log_level=config.log_level,
            log_file=config.get('logging.log_file', 'logs/trading_bot.log')
        )

        logger.info("=" * 60)
        logger.info("AI MT5 Trading Bot Starting...")
        logger.info("=" * 60)

        # Performance tracking
        self.performance = PerformanceTracker()

        # Initialize components
        self.mt5_connector = None
        self.order_executor = None
        self.market_data_manager = None
        self.analyzer = None
        self.signal_generator = None
        self.telegram_bot = None

        # Control flags
        self.is_running = False
        self.analysis_interval = 60  # seconds
        self.auto_trading_enabled = config.get('mt5.auto_trading_enabled', True)

        # Risk management
        self.risk_per_trade = float(os.getenv('MT5_RISK_PER_TRADE', 1.0))
        self.max_open_positions = int(os.getenv('MT5_MAX_OPEN_POSITIONS', 3))

        # Initialize all components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all bot components"""
        try:
            # MT5 connector
            logger.info("Initializing MT5 connector...")
            mt5_login = config.get('mt5.login')
            mt5_password = config.get('mt5.password')
            mt5_server = config.get('mt5.server')
            mt5_path = config.get('mt5.path')

            self.mt5_connector = MT5Connector(
                login=int(mt5_login) if mt5_login else None,
                password=mt5_password,
                server=mt5_server,
                path=mt5_path
            )

            if not self.mt5_connector.is_connected:
                raise Exception("Failed to connect to MT5")

            # Order executor
            logger.info("Initializing order executor...")
            self.order_executor = MT5OrderExecutor(
                connector=self.mt5_connector,
                magic_number=config.get('mt5.magic_number', 234000)
            )

            # Get account info
            account_info = self.mt5_connector.get_account_info()
            if account_info:
                logger.info(f"MT5 Account: {account_info['login']}, "
                           f"Balance: {account_info['balance']} {account_info['currency']}, "
                           f"Server: {account_info['server']}")

            # Log all available symbols for debugging
            available_symbols = self.mt5_connector.get_available_symbols()
            logger.info(f"Available symbols from broker: {available_symbols}")

            # Market data manager
            logger.info("Initializing market data manager...")
            self.market_data_manager = MT5MarketDataManager(
                connector=self.mt5_connector,
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
                logger.warning(f"Could not load models: {e}. Training new models...")
                self.analyzer.enable_training = True
                
                # Fetch data for training
                logger.info("Fetching historical data for initial training...")
                training_symbol = config.symbols[0]
                training_timeframe = '1h'
                
                # Use the connector to get historical data
                training_df = self.mt5_connector.get_historical_data(
                    training_symbol,
                    training_timeframe,
                    number_of_candles=2000
                )
                
                if training_df is not None and not training_df.empty:
                    self.analyzer.train(training_df)
                    logger.info("Initial model training completed.")
                    # Optional: Save the newly trained models
                    self.analyzer.save_models('models')
                else:
                    logger.error("Could not fetch data for training. Bot will use rule-based models only.")

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

            logger.info("Starting MT5 Trading Bot...")

            # Test Telegram connection
            if self.telegram_bot:
                telegram_ok = await self.telegram_bot.test_connection()
                if not telegram_ok:
                    logger.warning("Telegram connection test failed, but continuing...")

            # Get account info
            account_info = self.mt5_connector.get_account_info()

            # Send startup message
            await self.telegram_bot.send_message(
                f"🚀 **AI MT5 Trading Bot Started**\n\n"
                f"**Account:** {account_info['login']}\n"
                f"**Balance:** {account_info['balance']} {account_info['currency']}\n"
                f"**Server:** {account_info['server']}\n"
                f"**Leverage:** 1:{account_info['leverage']}\n\n"
                f"Monitoring {len(config.symbols)} symbols: {', '.join(config.symbols)}\n"
                f"Timeframes: {', '.join(config.timeframes_list)}\n\n"
                f"**Auto-Trading:** {'✅ ENABLED' if self.auto_trading_enabled else '❌ DISABLED'}\n"
                f"**Risk per Trade:** {self.risk_per_trade}%\n"
                f"**Max Positions:** {self.max_open_positions}\n"
                f"**Min Confidence:** {config.ai.confidence_threshold:.0%}\n"
                f"**Min Signal Strength:** {config.ai.min_signal_score}/100"
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
        logger.info("Stopping MT5 Trading Bot...")

        self.is_running = False

        # Stop market data collection
        if self.market_data_manager:
            await self.market_data_manager.stop()

        # Close MT5 connection
        if self.mt5_connector:
            self.mt5_connector.shutdown()

        # Send shutdown message
        if self.telegram_bot:
            account_info = self.mt5_connector.get_account_info()
            await self.telegram_bot.send_message(
                f"🛑 **AI MT5 Trading Bot Stopped**\n\n"
                f"**Final Balance:** {account_info['balance'] if account_info else 'N/A'}\n"
                f"**Total Signals:** {self.performance.signals_generated}\n"
                f"**Uptime:** {self.performance.get_statistics()['uptime_hours']:.2f} hours"
            )

        logger.info("MT5 Trading Bot stopped")

    async def _run_main_loop(self):
        """Main analysis and signal generation loop"""
        logger.info("Main loop started")

        while self.is_running:
            try:
                logger.info("=" * 30 + " Starting New Analysis Cycle " + "=" * 30)
                # Check MT5 connection
                if not self.mt5_connector.check_connection():
                    logger.error("MT5 connection lost!")
                    await self.telegram_bot.send_error_alert(
                        "MT5 connection lost!",
                        "Attempting to reconnect..."
                    )
                    await asyncio.sleep(10)
                    continue

                # Analyze all symbols
                for symbol in config.symbols:
                    await self._analyze_and_execute(symbol)

                logger.info("=" * 30 + " Analysis Cycle Completed " + "=" * 32)
                logger.info(f"Waiting for {self.analysis_interval} seconds until next cycle...")
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

    async def _analyze_and_execute(self, symbol: str):
        """
        Analyze market and execute trade automatically

        Args:
            symbol: Trading symbol to analyze
        """
        try:
            start_time = datetime.utcnow()

            # Get multi-timeframe data
            mtf_data = self.market_data_manager.get_multi_timeframe_data(symbol, limit=1000)

            if not mtf_data or all(df.empty for df in mtf_data.values()):
                logger.debug(f"{symbol}: No data available yet")
                return

            logger.info(f"Analyzing {symbol}...")
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

            # If signal generated
            if signal:
                logger.success(f"Signal generated for {symbol}: {signal.signal_type} | Confidence: {signal.confidence:.2%} | Strength: {signal.strength}/100")

                # Prepare market data for chart
                chart_data = {
                    'data': mtf_data.get(signal.timeframe, None)
                }

                # ALWAYS send signal to Telegram (regardless of execution limits)
                await self.telegram_bot.send_signal(signal, chart_data)

                # Execute trade automatically if enabled AND within execution limits
                if self.auto_trading_enabled:
                    # Check execution limits before executing on MT5
                    if self.signal_generator.signal_filter.should_execute(symbol, signal.signal_type):
                        logger.info(f"{symbol}: Executing order on MT5...")
                        await self._execute_signal(signal)
                    else:
                        logger.info(f"{symbol}: Signal sent to Telegram but NOT executed on MT5 (execution limits reached)")
                        # Record the signal even if not executed
                        self.performance.record_signal(symbol, signal.signal_type)
                else:
                    # If not auto-trading, record the signal
                    self.performance.record_signal(symbol, signal.signal_type)
            else:
                logger.info(f"Analysis for {symbol} complete. No signal generated (HOLD).")

            # Record analysis time
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"Analysis for {symbol} took {elapsed:.2f} seconds.")
            self.performance.record_analysis_time(elapsed)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            self.performance.record_error("analysis", str(e), symbol)

    async def _execute_signal(self, signal):
        """
        Execute signal automatically on MT5

        Args:
            signal: TradingSignal object
        """
        try:
            logger.info(f"Executing signal: {signal.signal_type} {signal.symbol}")

            # Check position limits BEFORE executing on MT5
            open_positions = self.order_executor.get_open_positions(signal.symbol)
            total_positions = len(self.order_executor.get_open_positions())

            if len(open_positions) > 0:
                logger.warning(f"{signal.symbol}: Already has open position, NOT executing on MT5")
                return

            if total_positions >= self.max_open_positions:
                logger.warning(f"Max positions ({self.max_open_positions}) reached, NOT executing {signal.symbol} on MT5")
                return

            # Get account info
            account_info = self.mt5_connector.get_account_info()

            if not account_info:
                logger.error("Could not get account info")
                return

            # Get symbol info
            symbol_info = self.mt5_connector.get_symbol_info(signal.symbol)

            if not symbol_info:
                logger.error(f"Could not get symbol info for {signal.symbol}")
                return

            # Calculate lot size based on risk
            stop_loss_pips = abs(signal.entry_price - signal.stop_loss) / symbol_info['point'] / 10

            lot_size = self.order_executor.calculate_lot_size(
                symbol=signal.symbol,
                risk_percent=self.risk_per_trade,
                stop_loss_pips=stop_loss_pips,
                account_balance=account_info['balance']
            )

            logger.info(f"Calculated lot size: {lot_size} (Risk: {self.risk_per_trade}%, SL: {stop_loss_pips} pips)")

            # Execute order
            # MT5 comment limit: 31 characters
            symbol_clean = signal.symbol.replace(" ", "")
            comment = f"Bot-{symbol_clean}-{signal.signal_type}"[:31]

            result = self.order_executor.execute_market_order(
                symbol=signal.symbol,
                order_type=signal.signal_type,
                volume=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit_levels[0] if signal.take_profit_levels else None,
                comment=comment
            )

            if result:
                # Log execution confirmation internally (NOT sent to Telegram)
                logger.info(
                    f"✅ ORDER EXECUTED - "
                    f"Ticket: {result['ticket']}, "
                    f"Symbol: {result['symbol']}, "
                    f"Type: {result['type']}, "
                    f"Volume: {result['volume']} lots, "
                    f"Entry: {result['price']}, "
                    f"SL: {result['sl']}, "
                    f"TP: {result['tp']}, "
                    f"Time: {result['time'].strftime('%Y-%m-%d %H:%M:%S')}"
                )

                # Store execution info for internal tracking
                self.performance.record_signal(signal.symbol, signal.signal_type)

            else:
                # Only log error internally (NOT sent to Telegram)
                logger.error(f"Failed to execute order for {signal.symbol}")
                self.performance.record_error("order_execution", f"Failed to execute {signal.signal_type}", signal.symbol)

        except Exception as e:
            # Only log error internally (NOT sent to Telegram)
            logger.error(f"Error executing signal: {e}")
            self.performance.record_error("signal_execution", str(e), signal.symbol)

    async def _run_periodic_tasks(self):
        """Run periodic maintenance tasks"""
        try:
            # Get account info
            account_info = self.mt5_connector.get_account_info()

            # Get open positions
            positions = self.order_executor.get_open_positions()

            # Send status update
            health = self.performance.get_health_status()
            logger.info(f"Health check: {health['status']}")

            # Hourly update (internal logging only, optional Telegram)
            # Uncomment below if you want hourly updates in Telegram
            # await self.telegram_bot.send_message(
            #     f"📊 **Hourly Status Update**\n\n"
            #     f"**Balance:** {account_info['balance']} {account_info['currency']}\n"
            #     f"**Equity:** {account_info['equity']}\n"
            #     f"**Profit:** {account_info['profit']}\n"
            #     f"**Open Positions:** {len(positions)}\n"
            #     f"**Signals Today:** {self.signal_generator.get_signal_statistics()['total_signals']}\n"
            #     f"**Health:** {health['status']}"
            # )

            # Internal logging
            logger.info(
                f"Hourly status - Balance: {account_info['balance']}, "
                f"Equity: {account_info['equity']}, "
                f"Profit: {account_info['profit']}, "
                f"Open Positions: {len(positions)}, "
                f"Signals: {self.signal_generator.get_signal_statistics()['total_signals']}"
            )

            # Daily summary at midnight UTC
            if datetime.utcnow().hour == 0:
                summary = self.performance.get_daily_summary()
                await self.telegram_bot.send_daily_summary(summary)
                self.performance.reset_daily_metrics()

        except Exception as e:
            logger.error(f"Error in periodic tasks: {e}")


async def main():
    """Main entry point"""
    bot = MT5TradingBot()

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
