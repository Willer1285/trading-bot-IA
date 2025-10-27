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
        self.min_signal_score: int = int(os.getenv("MIN_SIGNAL_SCORE", 80))

        # Gestión de Riesgos
        self.max_signals_per_day: int = int(os.getenv("MAX_SIGNALS_PER_DAY", 10))
        self.max_signals_per_pair: int = int(os.getenv("MAX_SIGNALS_PER_PAIR", 3))
        self.stop_loss_points: int = int(os.getenv("STOP_LOSS_POINTS", 500))
        self.take_profit_1_points: int = int(os.getenv("TAKE_PROFIT_1_POINTS", 1000))
        self.take_profit_2_points: int = int(os.getenv("TAKE_PROFIT_2_POINTS", 2000))

        # Configuración de Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_file: str = os.getenv("LOG_FILE", "logs/trading_bot.log")

# Crear una instancia única de la configuración para ser importada en otros módulos
config = Config()
