"""
Message Formatter
Formats messages for Telegram
"""

from typing import Dict
from datetime import datetime
from signal_generator.signal_generator import TradingSignal


class MessageFormatter:
    """Formats trading signals and updates for Telegram"""

    @staticmethod
    def format_signal(signal: TradingSignal) -> str:
        """
        Format trading signal for Telegram

        Args:
            signal: Trading signal

        Returns:
            Formatted message string
        """
        # Get signal emoji
        if signal.signal_type == 'BUY':
            signal_emoji = "🟢"
            direction_emoji = "📈"
        else:
            signal_emoji = "🔴"
            direction_emoji = "📉"

        # Format confidence
        confidence_bar = MessageFormatter._create_bar(signal.confidence * 100, 100)

        # Create message
        message = f"""
{signal_emoji} **{signal.signal_type} SIGNAL** {signal_emoji}

━━━━━━━━━━━━━━━━━━━━
**Symbol:** `{signal.symbol}`
**Timeframe:** {signal.timeframe}

{direction_emoji} **ENTRY**
💵 Price: `${signal.entry_price:.6f}`

🛑 **STOP LOSS**
💵 Price: `${signal.stop_loss:.6f}`

🎯 **TAKE PROFIT LEVELS**
"""

        # Add take profit levels
        for i, tp in enumerate(signal.take_profit_levels, 1):
            profit_pct = abs((tp - signal.entry_price) / signal.entry_price * 100)
            message += f"   TP{i}: `${tp:.6f}` (+{profit_pct:.2f}%)\n"

        # Add risk/reward and confidence
        message += f"""
━━━━━━━━━━━━━━━━━━━━
📊 **ANALYSIS**

**Risk/Reward:** 1:{signal.risk_reward_ratio:.2f}

**Confidence:** {signal.confidence:.1%}
{confidence_bar}

💡 **Reason:**
_{signal.reason}_

━━━━━━━━━━━━━━━━━━━━
⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC

_Powered by AI Trading Bot_ 🤖
"""

        return message

    @staticmethod
    def format_daily_summary(summary: Dict) -> str:
        """
        Format daily summary

        Args:
            summary: Summary statistics

        Returns:
            Formatted message
        """
        total_signals = summary.get('total_signals', 0)
        buy_signals = summary.get('buy_signals', 0)
        sell_signals = summary.get('sell_signals', 0)
        avg_confidence = summary.get('avg_confidence', 0)

        message = f"""
📊 **DAILY SUMMARY**

━━━━━━━━━━━━━━━━━━━━
**Signals Generated:** {total_signals}

🟢 **Buy Signals:** {buy_signals}
🔴 **Sell Signals:** {sell_signals}

**Average Confidence:** {avg_confidence:.1%}

━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.utcnow().strftime('%Y-%m-%d')}

_Stay disciplined and follow your risk management!_ 💪
"""

        return message

    @staticmethod
    def format_performance(performance: Dict) -> str:
        """
        Format performance metrics

        Args:
            performance: Performance data

        Returns:
            Formatted message
        """
        message = f"""
📈 **PERFORMANCE UPDATE**

━━━━━━━━━━━━━━━━━━━━
"""

        for key, value in performance.items():
            formatted_key = key.replace('_', ' ').title()

            if isinstance(value, float):
                if 'percent' in key or 'rate' in key:
                    message += f"**{formatted_key}:** {value:.2f}%\n"
                else:
                    message += f"**{formatted_key}:** {value:.2f}\n"
            else:
                message += f"**{formatted_key}:** {value}\n"

        message += f"""
━━━━━━━━━━━━━━━━━━━━
⏰ {MessageFormatter.get_timestamp()}
"""

        return message

    @staticmethod
    def format_market_update(symbol: str, price: float, change_24h: float, volume: float) -> str:
        """
        Format market update

        Args:
            symbol: Trading pair
            price: Current price
            change_24h: 24h price change percentage
            volume: 24h volume

        Returns:
            Formatted message
        """
        change_emoji = "📈" if change_24h >= 0 else "📉"
        color = "🟢" if change_24h >= 0 else "🔴"

        message = f"""
{color} **{symbol}** Market Update

**Price:** ${price:.6f}
{change_emoji} **24h Change:** {change_24h:+.2f}%
**24h Volume:** ${volume:,.0f}

⏰ {MessageFormatter.get_timestamp()}
"""

        return message

    @staticmethod
    def _create_bar(value: float, max_value: float, length: int = 10) -> str:
        """
        Create a visual progress bar

        Args:
            value: Current value
            max_value: Maximum value
            length: Bar length

        Returns:
            Progress bar string
        """
        filled = int((value / max_value) * length)
        empty = length - filled

        bar = "█" * filled + "░" * empty

        return f"[{bar}] {value:.0f}%"

    @staticmethod
    def get_timestamp() -> str:
        """Get formatted timestamp"""
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

    @staticmethod
    def format_break_even(position: Dict, new_sl: float) -> str:
        """Formats a break even notification."""
        return f"""
        🛡️ **Protege tu Operación (Break Even)** 🛡️

        **Símbolo:** `{position['symbol']}`
        **Tipo:** {position['type']}

        Mueve el SL para minimizar perdidas y asegurar ganancias.

        **Nuevo SL:** `${new_sl:.5f}`
        """

    @staticmethod
    def format_trailing_stop(position: Dict, new_sl: float) -> str:
        """Formats a trailing stop notification."""
        return f"""
        📈 **Ganancia Asegurada (Trailing Stop)** 📈

        **Símbolo:** `{position['symbol']}`
        **Tipo:** {position['type']}
        **Ticket:** `{position['ticket']}`

        El Stop Loss ha sido actualizado para proteger las ganancias.

        **Nuevo SL:** `${new_sl:.5f}`
        """
