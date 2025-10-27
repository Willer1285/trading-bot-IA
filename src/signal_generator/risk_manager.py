"""
Risk Manager
Calcula los parámetros de riesgo para las señales de trading.
"""

from typing import Dict, List, Optional
from loguru import logger

from config import config
from ai_engine.market_analyzer import MarketAnalysis

class RiskManager:
    """Gestiona los parámetros de riesgo para las señales de trading."""

    def __init__(self):
        """Inicializa el Risk Manager."""
        logger.info(f"Risk Manager inicializado con SL fijo de {config.stop_loss_points} puntos y TPs de {config.take_profit_1_points}/{config.take_profit_2_points} puntos.")

    def calculate_risk_parameters(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        analysis: MarketAnalysis
    ) -> Optional[Dict]:
        """
        Calcula la entrada, el stop loss y los niveles de take profit usando puntos fijos.

        Args:
            symbol: Par de trading.
            signal_type: BUY o SELL.
            entry_price: Precio de entrada.
            analysis: Análisis de mercado (ya no se usa para el cálculo, pero se mantiene por la firma de la función).

        Returns:
            Diccionario con los parámetros de riesgo.
        """
        try:
            # El tamaño de un punto varía según el símbolo. Asumimos un tamaño de punto estándar.
            # Para índices sintéticos, 1 punto = 0.01 o 0.001. Usaremos 0.01 como un valor razonable.
            # NOTA: Esto podría necesitar ajuste si los símbolos tienen diferentes precisiones.
            point_size = 0.01  # Asunción general para índices sintéticos

            sl_distance = config.stop_loss_points * point_size
            tp1_distance = config.take_profit_1_points * point_size
            tp2_distance = config.take_profit_2_points * point_size

            if signal_type == 'BUY':
                stop_loss = entry_price - sl_distance
                take_profit_1 = entry_price + tp1_distance
                take_profit_2 = entry_price + tp2_distance
            elif signal_type == 'SELL':
                stop_loss = entry_price + sl_distance
                take_profit_1 = entry_price - tp1_distance
                take_profit_2 = entry_price - tp2_distance
            else:
                logger.warning(f"Tipo de señal desconocido: {signal_type}")
                return None

            if stop_loss <= 0 or take_profit_1 <= 0 or take_profit_2 <= 0:
                logger.warning(f"{symbol}: SL ({stop_loss}) o TPs ({take_profit_1}, {take_profit_2}) inválidos calculados.")
                return None

            risk_amount = abs(entry_price - stop_loss)
            reward_amount_1 = abs(take_profit_1 - entry_price)
            risk_reward_ratio_1 = reward_amount_1 / risk_amount if risk_amount > 0 else 0

            logger.info(f"Parámetros de riesgo para {symbol} ({signal_type}): SL={stop_loss:.5f}, TP1={take_profit_1:.5f}, TP2={take_profit_2:.5f}, RR1={risk_reward_ratio_1:.2f}")

            return {
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit_levels': [float(take_profit_1), float(take_profit_2)],
                'risk_amount': float(risk_amount),
                'risk_reward_ratio': float(risk_reward_ratio_1),
            }

        except Exception as e:
            logger.error(f"Error al calcular los parámetros de riesgo para {symbol}: {e}")
            return None
