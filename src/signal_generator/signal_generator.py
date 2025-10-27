"""
Signal Generator
Generates high-probability trading signals from AI analysis
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
from loguru import logger

from ai_engine.market_analyzer import MarketAnalysis, MarketAnalyzer
from .signal_filter import SignalFilter
from .risk_manager import RiskManager


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_id: str
    symbol: str
    signal_type: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    confidence: float
    strength: int  # 0-100
    timeframe: str
    timestamp: datetime
    analysis: Dict
    risk_reward_ratio: float
    reason: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def get_formatted_message(self) -> str:
        """Get formatted signal message for Telegram"""
        signal_emoji = "🟢" if self.signal_type == "BUY" else "🔴"

        message = f"""
{signal_emoji} **{self.signal_type} SIGNAL** {signal_emoji}

**Symbol:** {self.symbol}
**Timeframe:** {self.timeframe}
**Confidence:** {self.confidence:.1%}
**Strength:** {self.strength}/100

📊 **Entry Price:** ${self.entry_price:.4f}
🛑 **Stop Loss:** ${self.stop_loss:.4f}
🎯 **Take Profit:**
"""
        for i, tp in enumerate(self.take_profit_levels, 1):
            message += f"   TP{i}: ${tp:.4f}\n"

        message += f"""
📈 **Risk/Reward:** 1:{self.risk_reward_ratio:.2f}

💡 **Reason:** {self.reason}

⏰ {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return message


class SignalGenerator:
    """Generates trading signals from market analysis"""

    def __init__(
        self,
        analyzer: MarketAnalyzer,
        signal_filter: SignalFilter,
        risk_manager: RiskManager,
        min_confidence: float = 0.75,
        min_strength: int = 80
    ):
        """
        Initialize Signal Generator

        Args:
            analyzer: Market analyzer instance
            signal_filter: Signal filter instance
            risk_manager: Risk manager instance
            min_confidence: Minimum confidence threshold
            min_strength: Minimum signal strength
        """
        self.analyzer = analyzer
        self.signal_filter = signal_filter
        self.risk_manager = risk_manager

        self.min_confidence = min_confidence
        self.min_strength = min_strength

        self.generated_signals: List[TradingSignal] = []
        self.signal_history: Dict[str, List[TradingSignal]] = {}

        logger.info("Signal Generator initialized")

    def generate_signal(
        self,
        symbol: str,
        multi_tf_analyses: Dict[str, Optional[MarketAnalysis]],
        current_price: float
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal from multi-timeframe analysis

        Args:
            symbol: Trading pair
            multi_tf_analyses: Multi-timeframe analysis results
            current_price: Current market price

        Returns:
            TradingSignal or None
        """
        try:
            # Get primary timeframe analysis (1h or 4h preferred)
            primary_analysis = None
            for tf in ['4h', '1h', '15m']:
                if tf in multi_tf_analyses and multi_tf_analyses[tf]:
                    primary_analysis = multi_tf_analyses[tf]
                    break

            if not primary_analysis:
                logger.warning(f"{symbol}: No primary analysis available")
                return None

            # Get consensus signal
            consensus_signal, consensus_confidence = self.analyzer.get_consensus_signal(
                multi_tf_analyses
            )

            strength = self.analyzer.calculate_signal_strength(primary_analysis)

            logger.info(f"Analysis result for {symbol}: Signal={consensus_signal}, Confidence={consensus_confidence:.2%}, Strength={strength}/100")

            # Check if signal is actionable
            if consensus_signal == 'HOLD':
                logger.info(f"{symbol}: ⏸️  Consensus signal is HOLD, skipping signal generation")
                return None

            # Check minimum thresholds
            if consensus_confidence < self.min_confidence:
                logger.warning(
                    f"{symbol}: ❌ Confidence {consensus_confidence:.2%} below threshold {self.min_confidence:.2%}"
                )
                return None

            if strength < self.min_strength:
                logger.warning(
                    f"{symbol}: ❌ Strength {strength} below threshold {self.min_strength}"
                )
                return None

            logger.info(f"{symbol}: ✅ Passed thresholds (Confidence: {consensus_confidence:.2%} >= {self.min_confidence:.2%}, Strength: {strength} >= {self.min_strength})")

            # Apply signal filters
            if not self.signal_filter.should_trade(symbol, consensus_signal, multi_tf_analyses):
                logger.warning(f"{symbol}: ❌ Signal filtered out by signal_filter.should_trade()")
                return None

            logger.info(f"{symbol}: ✅ Passed signal filter")

            # Calculate entry, stop loss, and take profit
            risk_params = self.risk_manager.calculate_risk_parameters(
                symbol=symbol,
                signal_type=consensus_signal,
                entry_price=current_price,
                analysis=primary_analysis
            )

            if not risk_params:
                logger.warning(f"{symbol}: Could not calculate risk parameters")
                return None

            # Generate signal ID
            signal_id = f"{symbol}_{consensus_signal}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            # Create reason
            reason = self._generate_reason(primary_analysis, multi_tf_analyses)

            # Create trading signal
            signal = TradingSignal(
                signal_id=signal_id,
                symbol=symbol,
                signal_type=consensus_signal,
                entry_price=risk_params['entry_price'],
                stop_loss=risk_params['stop_loss'],
                take_profit_levels=risk_params['take_profit_levels'],
                confidence=consensus_confidence,
                strength=strength,
                timeframe=primary_analysis.timeframe,
                timestamp=datetime.utcnow(),
                analysis=primary_analysis.to_dict(),
                risk_reward_ratio=risk_params['risk_reward_ratio'],
                reason=reason
            )

            # Store signal
            self.generated_signals.append(signal)
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            self.signal_history[symbol].append(signal)

            logger.info(
                f"Generated {consensus_signal} signal for {symbol} "
                f"(confidence: {consensus_confidence:.2%}, strength: {strength})"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _generate_reason(
        self,
        primary_analysis: MarketAnalysis,
        multi_tf_analyses: Dict[str, Optional[MarketAnalysis]]
    ) -> str:
        """Generate human-readable reason for signal"""
        reasons = []

        # Multi-timeframe alignment
        aligned_count = sum(
            1 for analysis in multi_tf_analyses.values()
            if analysis and analysis.signal == primary_analysis.signal
        )
        total_count = len([a for a in multi_tf_analyses.values() if a is not None])

        if aligned_count / total_count > 0.7:
            reasons.append(f"{aligned_count}/{total_count} timeframes aligned")

        # Technical indicators
        indicators = primary_analysis.indicators

        if primary_analysis.signal == 'BUY':
            if indicators.get('rsi_14', 50) < 40:
                reasons.append("RSI oversold")
            if indicators.get('price', 0) > indicators.get('ema_50', 0):
                reasons.append("Above EMA50")
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                reasons.append("MACD bullish")
        else:  # SELL
            if indicators.get('rsi_14', 50) > 60:
                reasons.append("RSI overbought")
            if indicators.get('price', 0) < indicators.get('ema_50', 0):
                reasons.append("Below EMA50")
            if indicators.get('macd', 0) < indicators.get('macd_signal', 0):
                reasons.append("MACD bearish")

        # Patterns
        patterns = primary_analysis.patterns
        bullish_patterns = ['bullish_engulfing', 'morning_star', 'hammer']
        bearish_patterns = ['bearish_engulfing', 'evening_star', 'shooting_star']

        for pattern in bullish_patterns:
            if patterns.get(pattern) and primary_analysis.signal == 'BUY':
                reasons.append(f"{pattern.replace('_', ' ').title()}")

        for pattern in bearish_patterns:
            if patterns.get(pattern) and primary_analysis.signal == 'SELL':
                reasons.append(f"{pattern.replace('_', ' ').title()}")

        return ", ".join(reasons) if reasons else "AI consensus signal"

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 10
    ) -> List[TradingSignal]:
        """
        Get recent signals

        Args:
            symbol: Filter by symbol (None for all)
            limit: Maximum number of signals

        Returns:
            List of recent signals
        """
        if symbol:
            signals = self.signal_history.get(symbol, [])
        else:
            signals = self.generated_signals

        return signals[-limit:]

    def get_signal_statistics(self) -> Dict:
        """Get statistics about generated signals"""
        if not self.generated_signals:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0,
                'avg_strength': 0
            }

        buy_signals = [s for s in self.generated_signals if s.signal_type == 'BUY']
        sell_signals = [s for s in self.generated_signals if s.signal_type == 'SELL']

        return {
            'total_signals': len(self.generated_signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': sum(s.confidence for s in self.generated_signals) / len(self.generated_signals),
            'avg_strength': sum(s.strength for s in self.generated_signals) / len(self.generated_signals),
            'symbols': list(self.signal_history.keys())
        }

    def clear_old_signals(self, days: int = 7):
        """Clear signals older than specified days"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        self.generated_signals = [
            s for s in self.generated_signals
            if s.timestamp > cutoff_time
        ]

        for symbol in self.signal_history:
            self.signal_history[symbol] = [
                s for s in self.signal_history[symbol]
                if s.timestamp > cutoff_time
            ]

        logger.info(f"Cleared signals older than {days} days")
