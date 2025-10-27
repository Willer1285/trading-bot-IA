"""
Signal Filter
Filters trading signals based on various criteria
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from loguru import logger

from ai_engine.market_analyzer import MarketAnalysis


class SignalFilter:
    """Filters signals to ensure high quality"""

    def __init__(
        self,
        min_volume_24h: float = 1000000,
        max_spread_percent: float = 0.5,
        max_signals_per_day: int = 10,
        max_signals_per_pair: int = 3,
        liquidity_check: bool = True
    ):
        """
        Initialize Signal Filter

        Args:
            min_volume_24h: Minimum 24h volume in USD
            max_spread_percent: Maximum spread percentage
            max_signals_per_day: Maximum signals per day
            max_signals_per_pair: Maximum signals per pair per day
            liquidity_check: Enable liquidity checking
        """
        self.min_volume_24h = min_volume_24h
        self.max_spread_percent = max_spread_percent
        self.max_signals_per_day = max_signals_per_day
        self.max_signals_per_pair = max_signals_per_pair
        self.liquidity_check = liquidity_check

        # Signal tracking
        self.daily_signal_count = 0
        self.pair_signal_count: Dict[str, int] = {}
        self.last_reset = datetime.utcnow()
        self.recent_signals: List[Dict] = []

        logger.info("Signal Filter initialized")

    def should_trade(
        self,
        symbol: str,
        signal_type: str,
        analyses: Dict[str, Optional[MarketAnalysis]]
    ) -> bool:
        """
        Check if signal should be traded

        Args:
            symbol: Trading pair
            signal_type: BUY or SELL
            analyses: Multi-timeframe analyses

        Returns:
            True if signal passes all filters
        """
        self._reset_daily_counts_if_needed()

        # Check daily limit
        if self.daily_signal_count >= self.max_signals_per_day:
            logger.debug(f"Daily signal limit reached: {self.daily_signal_count}/{self.max_signals_per_day}")
            return False

        # Check per-pair limit
        pair_count = self.pair_signal_count.get(symbol, 0)
        if pair_count >= self.max_signals_per_pair:
            logger.debug(f"{symbol}: Pair signal limit reached: {pair_count}/{self.max_signals_per_pair}")
            return False

        # Check timeframe confluence
        if not self._check_timeframe_confluence(analyses, signal_type):
            logger.debug(f"{symbol}: Insufficient timeframe confluence")
            return False

        # Check trend alignment
        if not self._check_trend_alignment(analyses, signal_type):
            logger.debug(f"{symbol}: Trend not aligned")
            return False

        # Check volatility
        if not self._check_volatility(analyses):
            logger.debug(f"{symbol}: Volatility too high")
            return False

        # Check for conflicting signals
        if self._has_conflicting_recent_signal(symbol, signal_type):
            logger.debug(f"{symbol}: Conflicting recent signal")
            return False

        # All checks passed
        self._record_signal(symbol, signal_type)
        return True

    def _check_timeframe_confluence(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> bool:
        """Check if multiple timeframes agree on signal"""
        valid_analyses = [a for a in analyses.values() if a is not None]

        if not valid_analyses:
            return False

        # Count agreements
        agreements = sum(1 for a in valid_analyses if a.signal == signal_type)

        # Need at least 60% agreement
        agreement_ratio = agreements / len(valid_analyses)

        return agreement_ratio >= 0.6

    def _check_trend_alignment(
        self,
        analyses: Dict[str, Optional[MarketAnalysis]],
        signal_type: str
    ) -> bool:
        """Check if signal aligns with higher timeframe trends"""
        # Get higher timeframe analysis (4h or 1d)
        higher_tf_analysis = None

        for tf in ['1d', '4h', '1h']:
            if tf in analyses and analyses[tf]:
                higher_tf_analysis = analyses[tf]
                break

        if not higher_tf_analysis:
            return True  # Can't check, so allow

        # For BUY signals, prefer uptrends
        # For SELL signals, prefer downtrends
        indicators = higher_tf_analysis.indicators

        price = indicators.get('price', 0)
        sma_50 = indicators.get('sma_50', 0)

        if signal_type == 'BUY':
            # Allow BUY in uptrend or near support
            return price >= sma_50 * 0.98  # Within 2% of SMA50

        else:  # SELL
            # Allow SELL in downtrend or near resistance
            return price <= sma_50 * 1.02  # Within 2% of SMA50

    def _check_volatility(self, analyses: Dict[str, Optional[MarketAnalysis]]) -> bool:
        """Check if volatility is acceptable"""
        # Get 1h analysis
        hourly_analysis = analyses.get('1h')

        if not hourly_analysis:
            return True  # Can't check, so allow

        indicators = hourly_analysis.indicators
        atr = indicators.get('atr', 0)
        price = indicators.get('price', 1)

        # ATR should be less than 5% of price
        atr_percent = (atr / price) * 100 if price > 0 else 0

        return atr_percent < 5.0

    def _has_conflicting_recent_signal(self, symbol: str, signal_type: str) -> bool:
        """Check for conflicting signals in recent history"""
        # Look back 1 hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        recent_for_symbol = [
            s for s in self.recent_signals
            if s['symbol'] == symbol and s['timestamp'] > cutoff_time
        ]

        for signal in recent_for_symbol:
            if signal['type'] != signal_type:
                return True  # Conflicting signal found

        return False

    def _record_signal(self, symbol: str, signal_type: str):
        """Record that a signal was generated"""
        self.daily_signal_count += 1
        self.pair_signal_count[symbol] = self.pair_signal_count.get(symbol, 0) + 1

        self.recent_signals.append({
            'symbol': symbol,
            'type': signal_type,
            'timestamp': datetime.utcnow()
        })

        # Keep only last 100 signals
        if len(self.recent_signals) > 100:
            self.recent_signals = self.recent_signals[-100:]

    def _reset_daily_counts_if_needed(self):
        """Reset daily counts at start of new day"""
        now = datetime.utcnow()

        if now.date() > self.last_reset.date():
            logger.info("Resetting daily signal counts")
            self.daily_signal_count = 0
            self.pair_signal_count = {}
            self.last_reset = now

    def get_statistics(self) -> Dict:
        """Get filter statistics"""
        return {
            'daily_signals': self.daily_signal_count,
            'max_daily': self.max_signals_per_day,
            'pair_counts': self.pair_signal_count,
            'recent_signals': len(self.recent_signals)
        }
