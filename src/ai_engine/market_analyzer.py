"""
Market Analyzer
Motor principal de IA para el análisis y la predicción del mercado.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime

from .feature_engineering import FeatureEngineer
from .technical_indicators import TechnicalIndicators
from .ai_models import EnsembleModel


class MarketAnalysis:
    """Contenedor para los resultados del análisis de mercado."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        prediction: int,
        confidence: float,
        probabilities: Dict[str, float],
        indicators: Dict,
        patterns: Dict[str, bool],
        support_resistance: Dict[str, float],
        timestamp: datetime
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.prediction = prediction  # 0=SELL, 1=HOLD, 2=BUY
        self.confidence = confidence
        self.probabilities = probabilities
        self.indicators = indicators
        self.patterns = patterns
        self.support_resistance = support_resistance
        self.timestamp = timestamp

    @property
    def signal(self) -> str:
        """Obtiene la señal como una cadena de texto."""
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return signal_map.get(self.prediction, 'HOLD')

    @property
    def is_actionable(self) -> bool:
        """Verifica si la señal es accionable (BUY o SELL)."""
        return self.prediction in [0, 2]

    def to_dict(self) -> Dict:
        """Convierte el objeto a un diccionario."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal': self.signal,
            'confidence': round(self.confidence, 4),
            'probabilities': {
                'sell': round(self.probabilities['sell'], 4),
                'hold': round(self.probabilities['hold'], 4),
                'buy': round(self.probabilities['buy'], 4)
            },
            'indicators': self.indicators,
            'patterns': self.patterns,
            'support_resistance': self.support_resistance,
            'timestamp': self.timestamp.isoformat()
        }


class MarketAnalyzer:
    """Analizador de mercado avanzado impulsado por IA."""

    def __init__(self, enable_training: bool = False):
        self.feature_engineer = FeatureEngineer()
        self.technical_indicators = TechnicalIndicators()
        self.model = EnsembleModel()
        self.enable_training = enable_training
        self.is_trained = False
        logger.info("Market Analyzer inicializado con sistema de puntuación híbrido mejorado.")

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Optional[MarketAnalysis]:
        """Realiza un análisis de mercado completo."""
        try:
            if df.empty or len(df) < 100:
                logger.warning(f"Datos insuficientes para el análisis: {len(df)} velas")
                return None

            df_features = self.feature_engineer.extract_features(df)
            df_clean = self.feature_engineer.prepare_for_model(df_features)

            if df_clean.empty:
                logger.warning("No hay características válidas después de la preparación")
                return None

            latest_features = df_clean.tail(1)

            # --- Lógica de Predicción Híbrida ---
            # 1. Obtener la dirección del modelo primario (basado en reglas)
            primary_prediction = self.model.base_models['pattern_based'].predict(latest_features)[0]

            if primary_prediction == 1:  # HOLD
                prediction = 1
                confidence = 0.6  # Confianza moderada para mantener
                probabilities = np.array([0.2, 0.6, 0.2]) # SELL, HOLD, BUY
            else:
                # 2. Si hay una señal de COMPRA/VENTA, usar el meta-modelo para obtener la confianza
                meta_confidence = self.model.predict_proba(latest_features)[0][1] # Probabilidad de que la señal sea buena
                
                prediction = primary_prediction
                confidence = meta_confidence

                # 3. Construir el array de probabilidades final
                if prediction == 2: # BUY
                    probabilities = np.array([(1 - confidence) / 2, (1 - confidence) / 2, confidence])
                else: # SELL
                    probabilities = np.array([confidence, (1 - confidence) / 2, (1 - confidence) / 2])

            indicators = self._extract_indicator_summary(df)
            patterns = self.technical_indicators.detect_patterns(df)
            support_resistance = self.technical_indicators.calculate_support_resistance(df)

            temp_analysis = MarketAnalysis(
                symbol=symbol, timeframe=timeframe, prediction=int(prediction),
                confidence=float(confidence), probabilities={'sell': float(probabilities[0]), 'hold': float(probabilities[1]), 'buy': float(probabilities[2])},
                indicators=indicators, patterns=patterns, support_resistance=support_resistance,
                timestamp=datetime.utcnow()
            )

            logger.debug(f"Análisis para {symbol} {timeframe}: {temp_analysis.signal} (Confianza: {temp_analysis.confidence:.2%})")
            return temp_analysis

        except Exception as e:
            logger.error(f"Error analizando {symbol} {timeframe}: {e}")
            return None

    def analyze_multi_timeframe(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict[str, Optional[MarketAnalysis]]:
        """Analiza múltiples timeframes."""
        results = {}
        for timeframe, df in data_dict.items():
            analysis = self.analyze(df, symbol, timeframe)
            results[timeframe] = analysis
        return results

    def _extract_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """Extrae los valores clave de los indicadores."""
        latest = df.iloc[-1]
        summary = {'price': float(latest['close'])}
        indicators_to_extract = [
            'rsi_14', 'macd', 'macd_signal', 'adx', 'sma_25', 'sma_50',
            'ema_21', 'ema_50', 'bb_high', 'bb_low', 'atr', 'obv'
        ]
        for indicator in indicators_to_extract:
            if indicator in df.columns:
                value = latest.get(indicator)
                if pd.notna(value):
                    summary[indicator] = float(value)
        return summary

    def save_models(self, directory: str = "models"):
        self.model.save_all(directory)
        logger.info(f"Modelos guardados en {directory}")

    def load_models(self, directory: str = "models"):
        self.model.load_all(directory)
        self.is_trained = True
        logger.info(f"Modelos cargados desde {directory}")
