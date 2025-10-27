"""
Feature Engineering
Advanced feature extraction for AI models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from .technical_indicators import TechnicalIndicators


class FeatureEngineer:
    """Engineer features for machine learning models"""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.feature_columns = []

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from OHLCV data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for feature extraction")
            return df

        try:
            # Start with a copy to avoid modifying the original DataFrame
            df_features = df.copy()

            # Calculate technical indicators first, as other features may depend on them
            df_features = self.indicators.calculate_all(df_features)

            # Create a dictionary to hold all new features
            new_features = {}

            # Add features from different categories
            new_features.update(self._get_price_action_features(df_features))
            new_features.update(self._get_temporal_features(df_features))
            new_features.update(self._get_statistical_features(df_features))
            new_features.update(self._get_market_regime_features(df_features))

            # Assign all new features at once
            df_features = df_features.assign(**new_features)

            # Store feature column names (excluding OHLCV)
            self.feature_columns = [
                col for col in df_features.columns
                if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']
            ]

            logger.debug(f"Extracted {len(self.feature_columns)} features")

            return df_features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return df

    def _get_price_action_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate price action based features and return as a dictionary."""
        features = {}
        
        body = abs(df['close'] - df['open'])
        high_low_range = df['high'] - df['low']
        
        features['body'] = body
        features['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        features['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        features['body_ratio'] = (body / high_low_range).fillna(0)
        features['is_bullish'] = (df['close'] > df['open']).astype(int)
        features['price_change_pct'] = df['close'].pct_change()
        features['high_change_pct'] = df['high'].pct_change()
        features['low_change_pct'] = df['low'].pct_change()
        features['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        features['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        features['price_position'] = ((df['close'] - df['low']) / high_low_range).fillna(0.5)
        features['consecutive_bullish'] = self._count_consecutive(df['close'] > df['open'])
        features['consecutive_bearish'] = self._count_consecutive(df['close'] < df['open'])
        
        return features

    def _get_temporal_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate time-based features and return as a dictionary."""
        features = {}
        
        if df.index.name != 'timestamp' and 'timestamp' not in df.columns:
            return features

        try:
            if df.index.name == 'timestamp':
                dt_index = df.index
            else:
                dt_index = pd.to_datetime(df['timestamp'])

            hour = dt_index.hour
            day_of_week = dt_index.dayofweek
            
            features['hour'] = hour
            features['day_of_week'] = day_of_week
            features['day_of_month'] = dt_index.day
            features['month'] = dt_index.month
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            features['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
            features['is_european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
            features['is_us_session'] = ((hour >= 16) & (hour < 24)).astype(int)

        except Exception as e:
            logger.debug(f"Could not add temporal features: {e}")

        return features

    def _get_statistical_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate statistical features and return as a dictionary."""
        features = {}
        windows = [7, 14, 30]
        pct_change = df['close'].pct_change()

        for window in windows:
            rolling_pct = pct_change.rolling(window)
            features[f'returns_mean_{window}'] = rolling_pct.mean()
            features[f'returns_std_{window}'] = rolling_pct.std()
            features[f'returns_skew_{window}'] = rolling_pct.skew()
            features[f'returns_kurt_{window}'] = rolling_pct.kurt()
            
            high_max = df['high'].rolling(window).max()
            low_min = df['low'].rolling(window).min()
            features[f'high_max_{window}'] = high_max
            features[f'low_min_{window}'] = low_min
            features[f'dist_from_high_{window}'] = (high_max - df['close']) / df['close']
            features[f'dist_from_low_{window}'] = (df['close'] - low_min) / df['close']

        close_rolling_20 = df['close'].rolling(20)
        features['zscore_close'] = (df['close'] - close_rolling_20.mean()) / close_rolling_20.std()
        
        return features

    def _get_market_regime_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate market regime features and return as a dictionary."""
        features = {}
        
        features['trend_20'] = np.where(df['close'] > df['sma_25'], 1, -1)
        features['trend_50'] = np.where(df['close'] > df['sma_50'], 1, -1)

        vol_20 = df['close'].pct_change().rolling(20).std()
        vol_percentile = vol_20.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        features['volatility_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 0.33, 0.66, 1.0],
            labels=[0, 1, 2]  # Low, Medium, High
        ).astype(float)

        vol_avg = df['volume'].rolling(20).mean()
        features['volume_regime'] = np.where(df['volume'] > vol_avg * 1.5, 1, 0)

        adx_value = df['adx'].fillna(0)
        features['market_phase'] = np.where(adx_value > 25, 1, 0)  # 1=Trending, 0=Ranging
        
        return features

    @staticmethod
    def _count_consecutive(series: pd.Series) -> pd.Series:
        """Count consecutive True values"""
        cumsum = series.cumsum()
        reset = cumsum - cumsum.where(~series).ffill().fillna(0)
        return reset.astype(int)

    def prepare_for_model(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Prepare features for machine learning

        Args:
            df: DataFrame with features
            target_column: Name of target column (if exists)
            fill_method: Method to fill NaN values

        Returns:
            Cleaned DataFrame ready for ML
        """
        df_clean = df.copy()

        # Select only feature columns
        if self.feature_columns:
            cols_to_keep = self.feature_columns.copy()
            if target_column and target_column in df_clean.columns:
                cols_to_keep.append(target_column)

            # Keep only existing columns
            cols_to_keep = [col for col in cols_to_keep if col in df_clean.columns]
            df_clean = df_clean[cols_to_keep]

        # Handle missing values
        if fill_method == 'ffill':
            df_clean = df_clean.ffill()
        elif fill_method == 'bfill':
            df_clean = df_clean.bfill()
        elif fill_method == 'zero':
            df_clean = df_clean.fillna(0)

        # Remove any remaining NaN
        df_clean = df_clean.dropna()

        # Replace infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], 0)

        return df_clean

    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_columns.copy()
