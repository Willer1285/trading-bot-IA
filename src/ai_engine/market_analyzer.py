"""
Market Analyzer
Main AI engine for market analysis and prediction
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
    """Container for market analysis results"""

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
        """Get signal as string"""
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return signal_map.get(self.prediction, 'HOLD')

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (BUY or SELL)"""
        return self.prediction in [0, 2]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
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
    """Advanced AI-powered market analyzer"""

    def __init__(self, enable_training: bool = False):
        """
        Initialize Market Analyzer

        Args:
            enable_training: Whether to enable model training
        """
        self.feature_engineer = FeatureEngineer()
        self.technical_indicators = TechnicalIndicators()
        self.model = EnsembleModel()

        self.enable_training = enable_training
        self.is_trained = False

        logger.info("Market Analyzer initialized")

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Optional[MarketAnalysis]:
        """
        Perform comprehensive market analysis

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            MarketAnalysis object or None
        """
        try:
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for analysis: {len(df)} candles")
                return None

            # Extract features
            df_features = self.feature_engineer.extract_features(df)

            # Prepare data for model
            df_clean = self.feature_engineer.prepare_for_model(df_features)

            if df_clean.empty:
                logger.warning("No valid features after preparation")
                return None

            # Get latest data point
            latest_features = df_clean.tail(1)

            # Make prediction
            raw_predictions = self.model.predict_individual(latest_features)
            for model_name, pred in raw_predictions.items():
                logger.debug(f"Model '{model_name}' prediction: {pred[0]}")

            prediction = self.model.predict(latest_features)[0]
            confidence = self.model.get_confidence(latest_features)[0]
            probabilities = self.model.predict_proba(latest_features)[0]

            # Get technical indicators summary
            latest_candle = df.iloc[-1]
            indicators = self._extract_indicator_summary(df)

            # Detect patterns
            patterns = self.technical_indicators.detect_patterns(df)

            # Calculate support/resistance
            support_resistance = self.technical_indicators.calculate_support_resistance(df)

            # Create analysis object
            analysis = MarketAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                prediction=int(prediction),
                confidence=float(confidence),
                probabilities={
                    'sell': float(probabilities[0]),
                    'hold': float(probabilities[1]),
                    'buy': float(probabilities[2])
                },
                indicators=indicators,
                patterns=patterns,
                support_resistance=support_resistance,
                timestamp=datetime.utcnow()
            )

            logger.debug(
                f"Analysis for {symbol} {timeframe}: "
                f"{analysis.signal} (confidence: {analysis.confidence:.2%})"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
            return None

    def analyze_multi_timeframe(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict[str, Optional[MarketAnalysis]]:
        """
        Analyze multiple timeframes

        Args:
            data_dict: Dictionary mapping timeframe to DataFrame
            symbol: Trading pair

        Returns:
            Dictionary mapping timeframe to MarketAnalysis
        """
        results = {}

        for timeframe, df in data_dict.items():
            analysis = self.analyze(df, symbol, timeframe)
            results[timeframe] = analysis

        return results

    def get_consensus_signal(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]]
    ) -> Tuple[str, float]:
        """
        Get consensus signal using the ensemble model.
        This now acts as a wrapper around the main analyze method, 
        but it processes multi-timeframe data to get a single prediction.
        """
        # We need to create a single feature vector that represents the multi-timeframe analysis
        # For simplicity, we'll use the features from the primary timeframe (e.g., 1h)
        # A more complex approach could involve combining features from all timeframes.
        
        primary_analysis = None
        for tf in ['1h', '4h', '15m', '5m', '1m']:
            if tf in analyses and analyses[tf]:
                primary_analysis = analyses[tf]
                break
        
        if not primary_analysis:
            return 'HOLD', 0.0

        # The 'analyze' method already uses the ensemble model. We just need to extract its prediction.
        # The confidence is the probability of the predicted class.
        prediction = primary_analysis.prediction
        confidence = primary_analysis.confidence
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return signal_map.get(prediction, 'HOLD'), float(confidence)

    def train(self, historical_data: pd.DataFrame):
        """
        Train AI models on historical data

        Args:
            historical_data: DataFrame with historical OHLCV data
        """
        if not self.enable_training:
            logger.warning("Training is disabled")
            return

        try:
            logger.info("Starting model training...")

            # Extract features
            df_features = self.feature_engineer.extract_features(historical_data)

            # Create labels
            labels = create_labels(df_features, forward_window=5, threshold=0.02)
            df_features['label'] = labels

            # Prepare data
            df_clean = self.feature_engineer.prepare_for_model(
                df_features,
                target_column='label'
            )

            if len(df_clean) < 100:
                logger.warning("Insufficient data for training")
                return

            # Split features and labels
            X = df_clean.drop('label', axis=1)
            y = df_clean['label']

            # Train ensemble
            self.model.fit(X, y)

            self.is_trained = True
            logger.info(f"Model training completed on {len(X)} samples")

        except Exception as e:
            logger.error(f"Error during training: {e}")

    def _extract_indicator_summary(self, df: pd.DataFrame) -> Dict:
        """Extract key indicator values"""
        latest = df.iloc[-1]

        summary = {
            'price': float(latest['close']),
        }

        # Add indicators if they exist
        indicators_to_extract = [
            'rsi_14', 'macd', 'macd_signal', 'adx',
            'sma_25', 'sma_50', 'ema_21', 'ema_50',
            'bb_high', 'bb_low', 'atr', 'obv'
        ]

        for indicator in indicators_to_extract:
            if indicator in df.columns:
                value = latest.get(indicator)
                if pd.notna(value):
                    summary[indicator] = float(value)

        return summary

    def save_models(self, directory: str = "models"):
        """Save trained models"""
        self.model.save_all(directory)
        logger.info(f"Models saved to {directory}")

    def load_models(self, directory: str = "models"):
        """Load trained models"""
        self.model.load_all(directory)
        self.is_trained = True
        logger.info(f"Models loaded from {directory}")

    def calculate_signal_strength(self, analysis: MarketAnalysis) -> int:
        """
        Calculate overall signal strength (0-100)

        Args:
            analysis: MarketAnalysis object

        Returns:
            Signal strength score
        """
        score = 0

        # Base score from confidence
        score += analysis.confidence * 40

        # Indicator alignment
        indicators = analysis.indicators

        if analysis.signal == 'BUY':
            if indicators.get('rsi_14', 50) < 40:
                score += 10
            if indicators.get('price', 0) > indicators.get('sma_50', 0):
                score += 10
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                score += 10

        elif analysis.signal == 'SELL':
            if indicators.get('rsi_14', 50) > 60:
                score += 10
            if indicators.get('price', 0) < indicators.get('sma_50', 0):
                score += 10
            if indicators.get('macd', 0) < indicators.get('macd_signal', 0):
                score += 10

        # Pattern confirmation
        patterns = analysis.patterns
        if patterns.get('bullish_engulfing') or patterns.get('morning_star'):
            if analysis.signal == 'BUY':
                score += 15
        if patterns.get('bearish_engulfing') or patterns.get('evening_star'):
            if analysis.signal == 'SELL':
                score += 15

        # Volume confirmation
        if indicators.get('obv', 0) > 0:
            score += 5

        return min(int(score), 100)
