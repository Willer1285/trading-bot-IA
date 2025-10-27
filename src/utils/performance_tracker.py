"""
Performance Tracker
Tracks system and trading performance
"""

from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


class PerformanceTracker:
    """Tracks performance metrics for the trading bot"""

    def __init__(self):
        self.start_time = datetime.utcnow()

        # Signal tracking
        self.signals_generated = 0
        self.signals_by_type: Dict[str, int] = defaultdict(int)
        self.signals_by_symbol: Dict[str, int] = defaultdict(int)

        # System metrics
        self.errors: List[Dict] = []
        self.api_calls = 0
        self.api_errors = 0

        # Performance metrics
        self.analysis_times: List[float] = []
        self.data_fetch_times: List[float] = []

        logger.info("Performance Tracker initialized")

    def record_signal(self, symbol: str, signal_type: str):
        """
        Record a generated signal

        Args:
            symbol: Trading pair
            signal_type: BUY or SELL
        """
        self.signals_generated += 1
        self.signals_by_type[signal_type] += 1
        self.signals_by_symbol[symbol] += 1

    def record_error(self, error_type: str, message: str, context: str = ""):
        """
        Record an error

        Args:
            error_type: Type of error
            message: Error message
            context: Additional context
        """
        self.errors.append({
            'type': error_type,
            'message': message,
            'context': context,
            'timestamp': datetime.utcnow()
        })

    def record_api_call(self, success: bool = True):
        """
        Record API call

        Args:
            success: Whether call was successful
        """
        self.api_calls += 1
        if not success:
            self.api_errors += 1

    def record_analysis_time(self, time_seconds: float):
        """Record analysis execution time"""
        self.analysis_times.append(time_seconds)

    def record_data_fetch_time(self, time_seconds: float):
        """Record data fetch time"""
        self.data_fetch_times.append(time_seconds)

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        uptime = datetime.utcnow() - self.start_time

        stats = {
            'uptime_hours': uptime.total_seconds() / 3600,
            'signals_generated': self.signals_generated,
            'signals_per_hour': self.signals_generated / (uptime.total_seconds() / 3600) if uptime.total_seconds() > 0 else 0,
            'signals_by_type': dict(self.signals_by_type),
            'signals_by_symbol': dict(self.signals_by_symbol),
            'api_calls': self.api_calls,
            'api_errors': self.api_errors,
            'api_success_rate': (self.api_calls - self.api_errors) / self.api_calls * 100 if self.api_calls > 0 else 0,
            'error_count': len(self.errors),
            'recent_errors': self.errors[-10:] if self.errors else []
        }

        # Add timing statistics
        if self.analysis_times:
            stats['avg_analysis_time'] = sum(self.analysis_times) / len(self.analysis_times)
            stats['max_analysis_time'] = max(self.analysis_times)

        if self.data_fetch_times:
            stats['avg_fetch_time'] = sum(self.data_fetch_times) / len(self.data_fetch_times)
            stats['max_fetch_time'] = max(self.data_fetch_times)

        return stats

    def get_daily_summary(self) -> Dict:
        """Get daily summary for reporting"""
        stats = self.get_statistics()

        summary = {
            'total_signals': self.signals_generated,
            'buy_signals': self.signals_by_type.get('BUY', 0),
            'sell_signals': self.signals_by_type.get('SELL', 0),
            'most_active_symbol': max(self.signals_by_symbol.items(), key=lambda x: x[1])[0] if self.signals_by_symbol else 'N/A',
            'api_success_rate': stats['api_success_rate'],
            'errors': len(self.errors)
        }

        return summary

    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.signals_generated = 0
        self.signals_by_type = defaultdict(int)
        self.signals_by_symbol = defaultdict(int)
        self.errors = []
        self.api_calls = 0
        self.api_errors = 0

        logger.info("Daily metrics reset")

    def get_health_status(self) -> Dict:
        """Get system health status"""
        stats = self.get_statistics()

        # Determine health
        health = "HEALTHY"

        if stats['error_count'] > 10:
            health = "WARNING"

        if stats['api_success_rate'] < 90:
            health = "WARNING"

        if stats['api_success_rate'] < 70:
            health = "CRITICAL"

        if stats['error_count'] > 50:
            health = "CRITICAL"

        return {
            'status': health,
            'uptime_hours': stats['uptime_hours'],
            'api_success_rate': stats['api_success_rate'],
            'error_count': stats['error_count'],
            'signals_generated': stats['signals_generated']
        }
