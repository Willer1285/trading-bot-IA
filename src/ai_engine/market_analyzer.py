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
from .ai_models import EnsembleModel, create_labels


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
        strength: int,  # Nuevo atributo para el score de fuerza
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
        self.strength = strength  # Score de 0 a 100
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
            'strength': self.strength,
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
            prediction = self.model.predict(latest_features)[0]
            confidence = self.model.get_confidence(latest_features)[0]
            probabilities = self.model.predict_proba(latest_features)[0]

            indicators = self._extract_indicator_summary(df)
            patterns = self.technical_indicators.detect_patterns(df)
            support_resistance = self.technical_indicators.calculate_support_resistance(df)

            temp_analysis = MarketAnalysis(
                symbol=symbol, timeframe=timeframe, prediction=int(prediction),
                confidence=float(confidence), probabilities={'sell': float(probabilities[0]), 'hold': float(probabilities[1]), 'buy': float(probabilities[2])},
                indicators=indicators, patterns=patterns, support_resistance=support_resistance,
                strength=0, timestamp=datetime.utcnow()
            )

            strength_score = self._calculate_signal_strength(temp_analysis)
            temp_analysis.strength = strength_score

            logger.debug(f"Análisis para {symbol} {timeframe}: {temp_analysis.signal} (Confianza: {temp_analysis.confidence:.2%}, Fuerza: {temp_analysis.strength}/100)")
            return temp_analysis

        except Exception as e:
            logger.error(f"Error analizando {symbol} {timeframe}: {e}")
            return None

    def _calculate_signal_strength(self, analysis: MarketAnalysis) -> int:
        """
        Calcula la fuerza de la señal (0-100) usando un sistema de puntuación granular.
        """
        score = 0
        reason = []

        # 1. Puntuación Base de la IA (Máx 30 puntos)
        if analysis.is_actionable:
            score += 15
            reason.append("IA Accionable (+15)")
            confidence_score = round(analysis.confidence * 15)
            score += confidence_score
            reason.append(f"Confianza IA {analysis.confidence:.0%} (+{confidence_score})")

        # 2. Puntuación de Momento (Máx 40 puntos)
        indicators = analysis.indicators
        rsi = indicators.get('rsi_14', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)

        if analysis.signal == 'BUY':
            # RSI: Más puntos cuanto más bajo (mejor si < 40)
            if rsi < 40:
                rsi_score = round((40 - rsi) / 40 * 20)
                score += rsi_score
                reason.append(f"RSI bajo ({rsi:.1f}) (+{rsi_score})")
            # MACD: Puntos si hay un cruce alcista
            if macd > macd_signal:
                score += 20
                reason.append("Cruce MACD (+20)")
        elif analysis.signal == 'SELL':
            # RSI: Más puntos cuanto más alto (mejor si > 60)
            if rsi > 60:
                rsi_score = round((rsi - 60) / 40 * 20)
                score += rsi_score
                reason.append(f"RSI alto ({rsi:.1f}) (+{rsi_score})")
            # MACD: Puntos si hay un cruce bajista
            if macd < macd_signal:
                score += 20
                reason.append("Cruce MACD (+20)")

        # 3. Puntuación de Tendencia (Máx 30 puntos)
        price = indicators.get('price', 0)
        sma_50 = indicators.get('sma_50', 0)
        adx = indicators.get('adx', 0)

        # Alineación con la media móvil
        if sma_50 > 0:
            if analysis.signal == 'BUY' and price > sma_50:
                score += 15
                reason.append("Precio > SMA50 (+15)")
            elif analysis.signal == 'SELL' and price < sma_50:
                score += 15
                reason.append("Precio < SMA50 (+15)")
        
        # Fuerza de la tendencia con ADX
        if adx > 25:
            score += 15
            reason.append(f"ADX > 25 ({adx:.0f}) (+15)")

        final_score = min(int(score), 100)
        logger.debug(f"Cálculo de Fuerza para {analysis.symbol} ({analysis.signal}): {final_score}/100. Razón: {', '.join(reason)}")
        return final_score

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

    def train(self, historical_data: pd.DataFrame):
        """Entrena los modelos de IA con datos históricos."""
        if not self.enable_training:
            logger.warning("El entrenamiento está deshabilitado")
            return
        try:
            logger.info("Iniciando entrenamiento del modelo...")
            df_features = self.feature_engineer.extract_features(historical_data)
            labels = create_labels(df_features, forward_window=5, threshold=None)
            df_features['label'] = labels
            if len(labels.unique()) < 2:
                logger.error("No hay suficiente diversidad de etiquetas para el entrenamiento.")
                return
            df_clean = self.feature_engineer.prepare_for_model(df_features, target_column='label')
            if len(df_clean) < 100:
                logger.warning("Datos insuficientes para el entrenamiento")
                return
            X = df_clean.drop('label', axis=1)
            y = df_clean['label']
            self.model.fit(X, y)
            self.is_trained = True
            logger.info(f"Entrenamiento del modelo completado en {len(X)} muestras")
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
