import asyncio
import sys
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add src directory to Python path to allow module imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import config
from src.data_collector.mt5_connector import MT5Connector, MT5OrderExecutor
from src.telegram_bot.telegram_bot import TelegramBot

async def test_telegram():
    """Tests the Telegram bot connection and sends a message."""
    logger.info("=" * 30 + " Testing Telegram Connection " + "=" * 30)
    if not config.telegram.bot_token or not config.telegram.channel_id:
        logger.error("Telegram BOT_TOKEN or CHANNEL_ID are not set in .env file. Skipping test.")
        return

    telegram_bot = TelegramBot(
        bot_token=config.telegram.bot_token,
        channel_id=config.telegram.channel_id
    )
    
    logger.info(f"Attempting to connect to Telegram with Bot Token: ...{config.telegram.bot_token[-4:]}")
    logger.info(f"Attempting to send a message to Channel ID: {config.telegram.channel_id}")

    success = await telegram_bot.test_connection()

    if success:
        logger.success("Telegram connection test PASSED. A test message was sent.")
    else:
        logger.error("Telegram connection test FAILED. Check your BOT_TOKEN and CHANNEL_ID.")
    logger.info("=" * 30 + " Telegram Test Finished " + "=" * 32)


async def test_mt5_execution():
    """Tests the MT5 connection and order execution capabilities."""
    logger.info("\n" + "=" * 30 + " Testing MT5 Order Execution " + "=" * 30)
    
    mt5_connector = MT5Connector(
        login=int(os.getenv('MT5_LOGIN')),
        password=os.getenv('MT5_PASSWORD'),
        server=os.getenv('MT5_SERVER')
    )

    if not mt5_connector.is_connected:
        logger.error("Failed to connect to MT5. Skipping order execution test.")
        return

    order_executor = MT5OrderExecutor(
        connector=mt5_connector,
        magic_number=config.get('mt5.magic_number', 234000)
    )

    # Use the first symbol from the config for the test
    test_symbol = config.symbols[0]
    logger.info(f"Using symbol '{test_symbol}' for execution test.")

    # Get symbol info to determine minimum volume
    symbol_info = mt5_connector.get_symbol_info(test_symbol)
    if not symbol_info:
        logger.error(f"Could not get info for symbol {test_symbol}. Aborting test.")
        return
    
    min_volume = symbol_info.get('volume_min', 0.01)
    
    logger.info(f"Attempting to place a small BUY order for {min_volume} lots of {test_symbol}...")

    # Execute a small market order
    result = order_executor.execute_market_order(
        symbol=test_symbol,
        order_type='BUY',
        volume=min_volume,
        comment="Component Test Order"
    )

    if result and result.get('ticket'):
        ticket = result['ticket']
        logger.success(f"Order execution test PASSED. Order placed successfully with ticket: {ticket}")
        
        logger.info(f"Waiting for 2 seconds before closing the test order...")
        await asyncio.sleep(2)
        
        closed = order_executor.close_order(ticket)
        if closed:
            logger.success(f"Successfully closed test order with ticket: {ticket}")
        else:
            logger.error(f"Failed to close test order with ticket: {ticket}. Please close it manually in MT5.")
    else:
        logger.error("Order execution test FAILED. The bot could not place an order.")
        
    logger.info("=" * 30 + " MT5 Execution Test Finished " + "=" * 29)
    mt5_connector.shutdown()


async def main():
    """Runs all component tests."""
    await test_telegram()
    await test_mt5_execution()


if __name__ == "__main__":
    logger.add("logs/test_components.log", rotation="10 MB", level="INFO")
    logger.info("Starting component tests...")
    asyncio.run(main())
    logger.info("Component tests finished.")
