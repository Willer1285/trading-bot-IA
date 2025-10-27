"""
Chart Generator
Generates trading charts for Telegram
"""

import io
from typing import Dict, Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger

from signal_generator.signal_generator import TradingSignal


class ChartGenerator:
    """Generates charts for trading signals"""

    def __init__(self):
        # Set style
        plt.style.use('dark_background')

    def generate_signal_chart(
        self,
        signal: TradingSignal,
        market_data: Dict
    ) -> Optional[io.BytesIO]:
        """
        Generate chart for trading signal

        Args:
            signal: Trading signal
            market_data: Market OHLCV data

        Returns:
            BytesIO image buffer or None
        """
        try:
            df = market_data.get('data')

            if df is None or df.empty:
                logger.warning("No market data provided for chart")
                return None

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

            # Plot candlestick chart
            self._plot_candlesticks(ax1, df)

            # Plot indicators
            self._plot_indicators(ax1, df, signal)

            # Plot signal entry/exit
            self._plot_signal_levels(ax1, signal)

            # Plot volume
            self._plot_volume(ax2, df)

            # Format
            self._format_chart(ax1, ax2, signal)

            # Save to buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            plt.close(fig)

            return buf

        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None

    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot candlestick chart"""
        # Simple candlestick representation
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]

        # Plot up candles (green)
        ax.bar(up.index, up['close'] - up['open'], bottom=up['open'],
               color='#26a69a', width=0.6, alpha=0.8)
        ax.bar(up.index, up['high'] - up['close'], bottom=up['close'],
               color='#26a69a', width=0.1)
        ax.bar(up.index, up['open'] - up['low'], bottom=up['low'],
               color='#26a69a', width=0.1)

        # Plot down candles (red)
        ax.bar(down.index, down['open'] - down['close'], bottom=down['close'],
               color='#ef5350', width=0.6, alpha=0.8)
        ax.bar(down.index, down['high'] - down['open'], bottom=down['open'],
               color='#ef5350', width=0.1)
        ax.bar(down.index, down['close'] - down['low'], bottom=down['low'],
               color='#ef5350', width=0.1)

    def _plot_indicators(self, ax, df: pd.DataFrame, signal: TradingSignal):
        """Plot technical indicators"""
        # Plot moving averages if available
        if 'ema_21' in df.columns:
            ax.plot(df.index, df['ema_21'], label='EMA 21', color='yellow', linewidth=1)

        if 'ema_50' in df.columns:
            ax.plot(df.index, df['ema_50'], label='EMA 50', color='orange', linewidth=1)

        # Plot Bollinger Bands if available
        if 'bb_high' in df.columns and 'bb_low' in df.columns:
            ax.plot(df.index, df['bb_high'], label='BB Upper', color='gray',
                   linewidth=0.5, linestyle='--', alpha=0.5)
            ax.plot(df.index, df['bb_low'], label='BB Lower', color='gray',
                   linewidth=0.5, linestyle='--', alpha=0.5)
            ax.fill_between(df.index, df['bb_low'], df['bb_high'],
                           alpha=0.1, color='gray')

    def _plot_signal_levels(self, ax, signal: TradingSignal):
        """Plot signal entry, stop loss, and take profit levels"""
        # Entry price
        ax.axhline(y=signal.entry_price, color='white', linestyle='--',
                  linewidth=1.5, label=f'Entry: ${signal.entry_price:.4f}')

        # Stop loss
        sl_color = '#ef5350' if signal.signal_type == 'BUY' else '#26a69a'
        ax.axhline(y=signal.stop_loss, color=sl_color, linestyle='--',
                  linewidth=1.5, label=f'Stop Loss: ${signal.stop_loss:.4f}')

        # Take profit levels
        colors = ['#4caf50', '#66bb6a', '#81c784']
        for i, (tp, color) in enumerate(zip(signal.take_profit_levels, colors), 1):
            ax.axhline(y=tp, color=color, linestyle='--',
                      linewidth=1, alpha=0.7, label=f'TP{i}: ${tp:.4f}')

    def _plot_volume(self, ax, df: pd.DataFrame):
        """Plot volume bars"""
        colors = ['#26a69a' if close >= open_ else '#ef5350'
                 for close, open_ in zip(df['close'], df['open'])]

        ax.bar(df.index, df['volume'], color=colors, alpha=0.5)
        ax.set_ylabel('Volume')

    def _format_chart(self, ax1, ax2, signal: TradingSignal):
        """Format chart appearance"""
        # Set titles and labels
        ax1.set_title(f"{signal.symbol} - {signal.signal_type} Signal", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Time', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

        # Rotate date labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
