"""
System Tests
Basic tests for the trading bot components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_data(periods=100):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1h')

    data = {
        'open': np.random.uniform(40000, 45000, periods),
        'high': np.random.uniform(41000, 46000, periods),
        'low': np.random.uniform(39000, 44000, periods),
        'close': np.random.uniform(40000, 45000, periods),
        'volume': np.random.uniform(1000, 10000, periods)
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp'

    # Ensure OHLC logic
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df


class TestTechnicalIndicators:
    """Test technical indicators module"""

    def test_calculate_all(self):
        """Test calculating all indicators"""
        from src.ai_engine.technical_indicators import TechnicalIndicators

        df = create_sample_data()
        df_with_indicators = TechnicalIndicators.calculate_all(df)

        # Check that indicators were added
        assert 'rsi_14' in df_with_indicators.columns
        assert 'macd' in df_with_indicators.columns
        assert 'sma_25' in df_with_indicators.columns
        assert len(df_with_indicators) == len(df)

    def test_detect_patterns(self):
        """Test pattern detection"""
        from src.ai_engine.technical_indicators import TechnicalIndicators

        df = create_sample_data()
        patterns = TechnicalIndicators.detect_patterns(df)

        assert isinstance(patterns, dict)
        # Should have pattern detection results
        assert len(patterns) > 0


class TestFeatureEngineering:
    """Test feature engineering module"""

    def test_extract_features(self):
        """Test feature extraction"""
        from src.ai_engine.feature_engineering import FeatureEngineer

        engineer = FeatureEngineer()
        df = create_sample_data()
        df_features = engineer.extract_features(df)

        # Should have more columns than original
        assert len(df_features.columns) > len(df.columns)

        # Should have feature columns stored
        assert len(engineer.feature_columns) > 0

    def test_prepare_for_model(self):
        """Test data preparation for ML"""
        from src.ai_engine.feature_engineering import FeatureEngineer

        engineer = FeatureEngineer()
        df = create_sample_data()
        df_features = engineer.extract_features(df)
        df_clean = engineer.prepare_for_model(df_features)

        # Should have no NaN values
        assert df_clean.isnull().sum().sum() == 0

        # Should have no infinite values
        assert not np.isinf(df_clean.select_dtypes(include=[np.number])).any().any()


class TestMarketAnalyzer:
    """Test market analyzer"""

    def test_analyze(self):
        """Test market analysis"""
        from src.ai_engine.market_analyzer import MarketAnalyzer

        analyzer = MarketAnalyzer(enable_training=False)
        df = create_sample_data(200)

        analysis = analyzer.analyze(df, 'BTC/USDT', '1h')

        assert analysis is not None
        assert analysis.signal in ['BUY', 'SELL', 'HOLD']
        assert 0 <= analysis.confidence <= 1
        assert isinstance(analysis.indicators, dict)


class TestRiskManager:
    """Test risk management"""

    def test_calculate_risk_parameters(self):
        """Test risk parameter calculation"""
        from src.signal_generator.risk_manager import RiskManager
        from src.ai_engine.market_analyzer import MarketAnalyzer

        # Create mock analysis
        df = create_sample_data(200)
        analyzer = MarketAnalyzer()
        analysis = analyzer.analyze(df, 'BTC/USDT', '1h')

        if analysis:
            risk_manager = RiskManager()

            params = risk_manager.calculate_risk_parameters(
                symbol='BTC/USDT',
                signal_type='BUY',
                entry_price=42000.0,
                analysis=analysis
            )

            assert params is not None
            assert 'entry_price' in params
            assert 'stop_loss' in params
            assert 'take_profit_levels' in params
            assert len(params['take_profit_levels']) > 0


class TestMessageFormatter:
    """Test Telegram message formatting"""

    def test_format_signal(self):
        """Test signal message formatting"""
        from src.telegram_bot.message_formatter import MessageFormatter
        from src.signal_generator.signal_generator import TradingSignal

        signal = TradingSignal(
            signal_id='test_123',
            symbol='BTC/USDT',
            signal_type='BUY',
            entry_price=42000.0,
            stop_loss=41500.0,
            take_profit_levels=[42500.0, 43000.0, 43500.0],
            confidence=0.85,
            strength=87,
            timeframe='1h',
            timestamp=datetime.utcnow(),
            analysis={},
            risk_reward_ratio=2.5,
            reason='Test signal'
        )

        formatter = MessageFormatter()
        message = formatter.format_signal(signal)

        assert 'BUY SIGNAL' in message
        assert 'BTC/USDT' in message
        assert '42000' in message


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
