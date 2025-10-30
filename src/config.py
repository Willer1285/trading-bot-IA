import os
from typing import List
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class Config:
    """
    Clase de configuración centralizada que carga todos los parámetros
    desde las variables de entorno.
    """

    def __init__(self):
        # Configuración de Telegram
        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_channel_id: str = os.getenv("TELEGRAM_CHANNEL_ID")
        self.telegram_include_charts: bool = os.getenv("TELEGRAM_INCLUDE_CHARTS", "true").lower() == "true"

        # Configuración de MT5
        self.mt5_login: int = int(os.getenv("MT5_LOGIN"))
        self.mt5_password: str = os.getenv("MT5_PASSWORD")
        self.mt5_server: str = os.getenv("MT5_SERVER")
        self.mt5_path: str = os.getenv("MT5_PATH", "")
        self.mt5_magic_number: int = int(os.getenv("MT5_MAGIC_NUMBER", 234000))
        self.mt5_auto_trading: bool = os.getenv("MT5_AUTO_TRADING", "true").lower() == "true"
        self.mt5_lot_size: float = float(os.getenv("MT5_LOT_SIZE", 0.01))
        self.mt5_max_open_positions: int = int(os.getenv("MT5_MAX_OPEN_POSITIONS", 3))

        # Símbolos y Timeframes de Trading
        self.trading_symbols: List[str] = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "").split(',') if s.strip()]
        self.timeframes: List[str] = [tf.strip() for tf in os.getenv("TIMEFRAMES", "1h").split(',') if tf.strip()]

        # Configuración de IA y Señales
        self.confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

        # Gestión de Riesgos
        self.atr_period: int = int(os.getenv("ATR_PERIOD", 14))
        self.max_signals_per_day: int = int(os.getenv("MAX_SIGNALS_PER_DAY", 10))
        self.max_signals_per_pair: int = int(os.getenv("MAX_SIGNALS_PER_PAIR", 3))
        self.stop_loss_atr_multiplier: float = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", 1.5))
        self.take_profit_1_atr_multiplier: float = float(os.getenv("TAKE_PROFIT_1_ATR_MULTIPLIER", 2.0))
        self.take_profit_2_atr_multiplier: float = float(os.getenv("TAKE_PROFIT_2_ATR_MULTIPLIER", 4.0))
        self.enforce_gainx_buy_only: bool = os.getenv("ENFORCE_GAINX_BUY_ONLY", "true").lower() == "true"
        self.enforce_painx_sell_only: bool = os.getenv("ENFORCE_PAINX_SELL_ONLY", "true").lower() == "true"

        # Gestión de Trades
        self.enable_break_even: bool = os.getenv("ENABLE_BREAK_EVEN", "true").lower() == "true"
        self.break_even_trigger_atr_multiplier: float = float(os.getenv("BREAK_EVEN_TRIGGER_ATR_MULTIPLIER", 1.0))
        self.break_even_profit_lock_atr_multiplier: float = float(os.getenv("BREAK_EVEN_PROFIT_LOCK_ATR_MULTIPLIER", 0.2))
        self.enable_trailing_stop: bool = os.getenv("ENABLE_TRAILING_STOP", "true").lower() == "true"
        self.trailing_stop_trigger_atr_multiplier: float = float(os.getenv("TRAILING_STOP_TRIGGER_ATR_MULTIPLIER", 2.0))
        self.trailing_stop_distance_atr_multiplier: float = float(os.getenv("TRAILING_STOP_DISTANCE_ATR_MULTIPLIER", 1.5))

        # Configuración de Lotaje Dinámico
        self.enable_dynamic_lot_size: bool = os.getenv("ENABLE_DYNAMIC_LOT_SIZE", "true").lower() == "true"
        self.min_lot_size: float = float(os.getenv("MIN_LOT_SIZE", 0.10))
        self.max_lot_size: float = float(os.getenv("MAX_LOT_SIZE", 1.00))

        # Configuración de Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_file: str = os.getenv("LOG_FILE", "logs/trading_bot.log")

# Crear una instancia única de la configuración para ser importada en otros módulos
config = Config()
