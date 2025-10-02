"""
Real-time Data Integration Module

Provides real-time market data streaming, processing, and optimization
capabilities for live portfolio management.
"""

from .data_stream_manager import DataStreamManager
from .market_data_provider import MarketDataProvider
from .event_processor import EventProcessor
from .streaming_optimizer import StreamingOptimizer

__all__ = [
    'DataStreamManager',
    'MarketDataProvider', 
    'EventProcessor',
    'StreamingOptimizer'
]