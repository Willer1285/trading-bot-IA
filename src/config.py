"""
Configuration Manager
Handles all configuration settings from environment variables and config files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class TelegramConfig(BaseModel):
    bot_token: str = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    channel_id: str = Field(default_factory=lambda: os.getenv("TELEGRAM_CHANNEL_ID", ""))

class ExchangeConfig(BaseModel):
    binance_api_key: str = Field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = Field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    bybit_api_key: str = Field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    bybit_api_secret: str = Field(default_factory=lambda: os.getenv("BYBIT_API_SECRET", ""))

class AIConfig(BaseModel):
    model_type: str = Field(default_factory=lambda: os.getenv("AI_MODEL_TYPE", "ensemble"))
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")))
    min_signal_score: int = Field(default_factory=lambda: int(os.getenv("MIN_SIGNAL_SCORE", "80")))

class DatabaseConfig(BaseModel):
    redis_host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = Field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    mongodb_uri: str = Field(default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
    mongodb_db: str = Field(default_factory=lambda: os.getenv("MONGODB_DB", "trading_bot"))

class Config:
    """Main configuration class"""

    def __init__(self, config_path: str = "config.yaml"):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / config_path

        # Load YAML configuration
        self.yaml_config = self._load_yaml_config()

        # Initialize config sections
        self.telegram = TelegramConfig()
        self.exchange = ExchangeConfig()
        self.ai = AIConfig()
        self.database = DatabaseConfig()

        # Trading configuration
        self.trading_pairs = os.getenv("TRADING_PAIRS", "BTC/USDT,ETH/USDT").split(",")
        self.timeframes = os.getenv("TIMEFRAMES", "1m,5m,15m,1h,4h,1d").split(",")

        # Environment
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from YAML config"""
        keys = key.split('.')
        value = self.yaml_config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    @property
    def symbols(self) -> List[str]:
        """Get trading symbols from config"""
        # Prioritize environment variable TRADING_SYMBOLS
        trading_symbols_env = os.getenv("TRADING_SYMBOLS")
        if trading_symbols_env:
            return [s.strip() for s in trading_symbols_env.split(',')]
        
        # Fallback to YAML config file
        default_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        return self.get('market.symbols', default_symbols)

    @property
    def timeframes_list(self) -> List[str]:
        """Get timeframes from config"""
        return self.get('market.timeframes', self.timeframes)

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

# Global configuration instance
config = Config()
