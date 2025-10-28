"""
Chart Generator
Genera gráficos de velas de alta calidad para las señales de trading.
"""
import io
from typing import Dict, Optional
import pandas as pd
import mplfinance as mpf
from loguru import logger

from signal_generator.signal_generator import TradingSignal

class ChartGenerator:
    """Genera gráficos de trading para las señales."""

    @staticmethod
    def generate_signal_chart(signal: TradingSignal, market_data: pd.DataFrame) -> Optional[io.BytesIO]:
        """
        Genera un gráfico de velas para una señal de trading.

        Args:
            signal: El objeto TradingSignal.
            market_data: DataFrame de pandas con los datos de mercado (OHLCV).

        Returns:
            Un objeto BytesIO que contiene la imagen del gráfico, o None si falla.
        """
        try:
            if market_data is None or market_data.empty:
                logger.warning(f"No se proporcionaron datos de mercado para el gráfico de {signal.symbol}.")
                return None

            # Preparar los datos: mplfinance espera columnas con nombres específicos
            df = market_data.copy()
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            # Usar solo las últimas 100 velas para una mejor visualización
            df = df.tail(100)

            # Definir las líneas horizontales para Entry, SL y TPs
            hlines = {
                'hlines': [signal.entry_price, signal.stop_loss] + signal.take_profit_levels,
                'colors': ['w', 'r', 'g', 'g'],
                'linestyle': '--'
            }

            # Crear un estilo de gráfico oscuro y profesional
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            style = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds')

            # Crear un buffer en memoria para guardar la imagen
            buf = io.BytesIO()

            # Generar el gráfico
            mpf.plot(
                df,
                type='candle',
                style=style,
                title=f"{signal.symbol} - {signal.signal_type} Signal",
                ylabel='Price (USDT)',
                volume=True,
                ylabel_lower='Volume',
                hlines=hlines,
                savefig=dict(fname=buf, dpi=300, pad_inches=0.25), # Alta resolución
                figsize=(12, 6)
            )

            buf.seek(0)
            logger.info(f"Gráfico generado exitosamente para la señal {signal.signal_id}.")
            return buf

        except Exception as e:
            logger.error(f"Error al generar el gráfico para la señal {signal.signal_id}: {e}")
            return None
