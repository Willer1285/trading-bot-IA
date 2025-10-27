"""
Risk Manager
Calculates risk parameters for trading signals
"""

from typing import Dict, List, Optional
from loguru import logger

from ai_engine.market_analyzer import MarketAnalysis


class RiskManager:
    """Manages risk parameters for trading signals"""

    def __init__(
        self,
        default_risk_reward: float = 2.0,
        atr_multiplier_sl: float = 1.5,
        take_profit_levels: List[float] = None
    ):
        """
        Initialize Risk Manager

        Args:
            default_risk_reward: Default risk/reward ratio
            atr_multiplier_sl: ATR multiplier for stop loss
            take_profit_levels: Take profit multipliers
        """
        self.default_risk_reward = default_risk_reward
        self.atr_multiplier_sl = atr_multiplier_sl
        self.take_profit_levels = take_profit_levels or [1.5, 2.0, 3.0]

        logger.info("Risk Manager initialized")

    def calculate_risk_parameters(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        analysis: MarketAnalysis
    ) -> Optional[Dict]:
        """
        Calculate entry, stop loss, and take profit levels

        Args:
            symbol: Trading pair
            signal_type: BUY or SELL
            entry_price: Entry price
            analysis: Market analysis

        Returns:
            Dictionary with risk parameters
        """
        try:
            indicators = analysis.indicators
            atr = indicators.get('atr', entry_price * 0.02)  # Default 2% if no ATR

            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(
                signal_type, entry_price, atr, analysis
            )
            logger.info(f"Calculated Stop Loss for {symbol}: {stop_loss:.6f} (Entry: {entry_price:.6f}, ATR: {atr:.6f})")

            if stop_loss <= 0:
                logger.warning(f"{symbol}: Invalid stop loss calculated: {stop_loss}")
                return None

            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss)

            # Calculate take profit levels
            take_profits = self._calculate_take_profits(
                signal_type, entry_price, risk_amount
            )
            logger.info(f"Calculated Take Profit levels for {symbol}: {take_profits}")

            # Calculate actual risk/reward ratio
            if take_profits:
                reward_amount = abs(take_profits[0] - entry_price)
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            else:
                risk_reward_ratio = self.default_risk_reward

            return {
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'take_profit_levels': [float(tp) for tp in take_profits],
                'risk_amount': float(risk_amount),
                'risk_reward_ratio': float(risk_reward_ratio),
                'atr': float(atr)
            }

        except Exception as e:
            logger.error(f"Error calculating risk parameters for {symbol}: {e}")
            return None

    def _calculate_stop_loss(
        self,
        signal_type: str,
        entry_price: float,
        atr: float,
        analysis: MarketAnalysis
    ) -> float:
        """Calculate stop loss level"""
        # Use ATR-based stop loss
        sl_distance = atr * self.atr_multiplier_sl

        # Consider support/resistance levels
        sr_levels = analysis.support_resistance

        if signal_type == 'BUY':
            # Stop loss below entry
            atr_sl = entry_price - sl_distance

            # Also consider support level
            support = sr_levels.get('support', 0)
            if support > 0:
                # Use the lower of ATR-based or below support
                support_sl = support * 0.995  # Slightly below support
                stop_loss = max(atr_sl, support_sl)  # Don't go too far
            else:
                stop_loss = atr_sl

        else:  # SELL
            # Stop loss above entry
            atr_sl = entry_price + sl_distance

            # Also consider resistance level
            resistance = sr_levels.get('resistance', 0)
            if resistance > 0:
                # Use the higher of ATR-based or above resistance
                resistance_sl = resistance * 1.005  # Slightly above resistance
                stop_loss = min(atr_sl, resistance_sl)  # Don't go too far
            else:
                stop_loss = atr_sl

        return stop_loss

    def _calculate_take_profits(
        self,
        signal_type: str,
        entry_price: float,
        risk_amount: float
    ) -> List[float]:
        """Calculate multiple take profit levels"""
        take_profits = []

        for multiplier in self.take_profit_levels:
            reward = risk_amount * multiplier

            if signal_type == 'BUY':
                tp = entry_price + reward
            else:  # SELL
                tp = entry_price - reward

            if tp > 0:
                take_profits.append(tp)

        return take_profits

    def calculate_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """
        Calculate position size based on risk

        Args:
            account_balance: Total account balance
            risk_percent: Percentage of account to risk (e.g., 1.0 for 1%)
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size in base currency
        """
        risk_amount_usd = account_balance * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0

        position_size = risk_amount_usd / price_risk

        return position_size

    def validate_risk_parameters(self, params: Dict) -> bool:
        """
        Validate risk parameters

        Args:
            params: Risk parameters dictionary

        Returns:
            True if valid
        """
        try:
            entry = params['entry_price']
            sl = params['stop_loss']
            tps = params['take_profit_levels']

            # Check that values are positive
            if entry <= 0 or sl <= 0:
                return False

            # Check that stop loss makes sense
            risk_percent = abs(entry - sl) / entry

            # Risk should be between 0.5% and 5%
            if risk_percent < 0.005 or risk_percent > 0.05:
                logger.warning(f"Risk percent {risk_percent:.2%} outside acceptable range")
                return False

            # Check take profits
            if not tps:
                return False

            # Ensure TPs are in correct direction
            for tp in tps:
                if tp <= 0:
                    return False

            # Check risk/reward ratio
            rr = params.get('risk_reward_ratio', 0)
            if rr < 1.0:  # Minimum 1:1
                logger.warning(f"Risk/reward ratio {rr} too low")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating risk parameters: {e}")
            return False
