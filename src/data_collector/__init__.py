"""
Módulo de Recopilación de Datos
Recopilación de datos de mercado en tiempo real desde MetaTrader 5.
"""

from .mt5_connector import MT5Connector, MT5OrderExecutor
from .mt5_market_data_manager import MT5MarketDataManager
from .timeframe_aggregator import TimeframeAggregator

__all__ = [
    'MT5Connector',
    'MT5OrderExecutor',
    'MT5MarketDataManager',
    'TimeframeAggregator'
]
