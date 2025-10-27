"""
Timeframe Aggregator
Aggregates lower timeframe data into higher timeframes
"""

import pandas as pd
from typing import Dict, List
from loguru import logger


class TimeframeAggregator:
    """Aggregates OHLCV data across different timeframes"""

    TIMEFRAME_MULTIPLIERS = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }

    @staticmethod
    def aggregate_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Aggregate OHLCV data to a higher timeframe

        Args:
            df: DataFrame with 1m OHLCV data
            target_timeframe: Target timeframe (5m, 15m, 1h, etc.)

        Returns:
            Aggregated DataFrame
        """
        try:
            if df.empty:
                return pd.DataFrame()

            # Determine resampling rule
            rule_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D',
            }

            rule = rule_map.get(target_timeframe, '1H')

            # Resample and aggregate
            aggregated = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Remove rows with NaN (incomplete periods)
            aggregated = aggregated.dropna()

            # Add metadata if present in original
            if 'symbol' in df.columns:
                aggregated['symbol'] = df['symbol'].iloc[0]
            if 'timeframe' in df.columns:
                aggregated['timeframe'] = target_timeframe

            logger.debug(f"Aggregated to {target_timeframe}: {len(aggregated)} candles")
            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating to {target_timeframe}: {e}")
            return pd.DataFrame()

    @staticmethod
    def align_timeframes(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple timeframe DataFrames to the same timestamp index

        Args:
            data_dict: Dictionary mapping timeframe to DataFrame

        Returns:
            Dictionary with aligned DataFrames
        """
        try:
            if not data_dict:
                return {}

            # Find common timestamp range
            min_time = max(df.index.min() for df in data_dict.values() if not df.empty)
            max_time = min(df.index.max() for df in data_dict.values() if not df.empty)

            # Align all dataframes to this range
            aligned = {}
            for tf, df in data_dict.items():
                if not df.empty:
                    aligned[tf] = df[min_time:max_time]
                else:
                    aligned[tf] = df

            return aligned

        except Exception as e:
            logger.error(f"Error aligning timeframes: {e}")
            return data_dict

    @staticmethod
    def calculate_timeframe_confluence(
        data_dict: Dict[str, pd.DataFrame],
        indicator_func,
        **kwargs
    ) -> Dict[str, pd.Series]:
        """
        Calculate an indicator across multiple timeframes

        Args:
            data_dict: Dictionary mapping timeframe to DataFrame
            indicator_func: Function to calculate indicator
            **kwargs: Arguments for indicator function

        Returns:
            Dictionary mapping timeframe to indicator values
        """
        results = {}

        for tf, df in data_dict.items():
            try:
                if not df.empty:
                    results[tf] = indicator_func(df, **kwargs)
            except Exception as e:
                logger.error(f"Error calculating indicator for {tf}: {e}")

        return results
