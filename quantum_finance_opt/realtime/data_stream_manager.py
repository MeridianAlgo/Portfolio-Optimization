"""
Real-time Data Stream Manager

Manages multiple data streams from various providers with error handling,
retry logic, and data quality validation.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Streaming libraries with fallbacks
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class MarketDataPoint:
    """Single market data point"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    source: str = "unknown"
    quality_score: float = 1.0


@dataclass
class StreamConfig:
    """Configuration for data stream"""
    provider: str
    symbols: List[str]
    data_types: List[str]
    update_frequency: str  # 'realtime', '1s', '5s', '1m', etc.
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    max_retries: int = 3
    timeout: float = 30.0
    buffer_size: int = 1000


class DataStreamManager:
    """
    Manages real-time data streams from multiple providers
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.logger = logging.getLogger(__name__)
        self.redis_url = redis_url
        self.redis_client = None
        
        # Stream management
        self.active_streams = {}
        self.stream_configs = {}
        self.data_callbacks = []
        self.error_callbacks = []
        
        # Data quality tracking
        self.quality_metrics = {}
        self.last_update_times = {}
        
        # Initialize Redis connection
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection for data caching"""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis.from_url(self.redis_url)
                self.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        else:
            self.logger.warning("Redis not available - using in-memory caching")
    
    def add_stream(self, stream_id: str, config: StreamConfig):
        """Add a new data stream"""
        self.stream_configs[stream_id] = config
        self.quality_metrics[stream_id] = {
            'total_messages': 0,
            'error_count': 0,
            'last_error': None,
            'avg_latency': 0.0,
            'data_quality_score': 1.0
        }
        self.logger.info(f"Added stream {stream_id} for provider {config.provider}")
    
    def start_stream(self, stream_id: str):
        """Start a specific data stream"""
        if stream_id not in self.stream_configs:
            raise QuantumFinanceOptError(f"Stream {stream_id} not configured")
        
        config = self.stream_configs[stream_id]
        
        # Create appropriate stream handler based on provider
        if config.provider.lower() == 'alpha_vantage':
            stream_handler = self._create_alpha_vantage_stream(config)
        elif config.provider.lower() == 'iex_cloud':
            stream_handler = self._create_iex_stream(config)
        elif config.provider.lower() == 'polygon':
            stream_handler = self._create_polygon_stream(config)
        elif config.provider.lower() == 'websocket':
            stream_handler = self._create_websocket_stream(config)
        else:
            stream_handler = self._create_mock_stream(config)
        
        self.active_streams[stream_id] = stream_handler
        
        # Start the stream in background
        asyncio.create_task(self._run_stream(stream_id, stream_handler))
        
        self.logger.info(f"Started stream {stream_id}")
    
    def stop_stream(self, stream_id: str):
        """Stop a specific data stream"""
        if stream_id in self.active_streams:
            # Signal stream to stop
            self.active_streams[stream_id]['stop_signal'] = True
            del self.active_streams[stream_id]
            self.logger.info(f"Stopped stream {stream_id}")
    
    def stop_all_streams(self):
        """Stop all active streams"""
        for stream_id in list(self.active_streams.keys()):
            self.stop_stream(stream_id)
    
    def add_data_callback(self, callback: Callable[[MarketDataPoint], None]):
        """Add callback function for new data points"""
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add callback function for stream errors"""
        self.error_callbacks.append(callback)
    
    async def _run_stream(self, stream_id: str, stream_handler: Dict[str, Any]):
        """Run a data stream with error handling and retry logic"""
        config = self.stream_configs[stream_id]
        retry_count = 0
        
        while stream_id in self.active_streams and not stream_handler.get('stop_signal', False):
            try:
                # Fetch data from stream
                data_points = await self._fetch_stream_data(stream_handler, config)
                
                # Process and validate data
                for data_point in data_points:
                    validated_data = self._validate_data_point(data_point, stream_id)
                    if validated_data:
                        # Cache data
                        self._cache_data_point(validated_data)
                        
                        # Notify callbacks
                        for callback in self.data_callbacks:
                            try:
                                callback(validated_data)
                            except Exception as e:
                                self.logger.error(f"Data callback error: {e}")
                
                # Update quality metrics
                self._update_quality_metrics(stream_id, len(data_points), None)
                
                # Reset retry count on success
                retry_count = 0
                
                # Wait before next fetch
                await asyncio.sleep(self._get_sleep_interval(config.update_frequency))
                
            except Exception as e:
                self.logger.error(f"Stream {stream_id} error: {e}")
                
                # Update error metrics
                self._update_quality_metrics(stream_id, 0, e)
                
                # Notify error callbacks
                for callback in self.error_callbacks:
                    try:
                        callback(stream_id, e)
                    except Exception as cb_error:
                        self.logger.error(f"Error callback failed: {cb_error}")
                
                # Retry logic
                retry_count += 1
                if retry_count <= config.max_retries:
                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff
                    self.logger.info(f"Retrying stream {stream_id} in {wait_time}s (attempt {retry_count})")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Stream {stream_id} failed after {config.max_retries} retries")
                    break  
  
    async def _fetch_stream_data(self, stream_handler: Dict[str, Any], config: StreamConfig) -> List[MarketDataPoint]:
        """Fetch data from stream handler"""
        fetch_function = stream_handler.get('fetch_function')
        if fetch_function:
            return await fetch_function(config)
        else:
            return []
    
    def _validate_data_point(self, data_point: MarketDataPoint, stream_id: str) -> Optional[MarketDataPoint]:
        """Validate and clean data point"""
        try:
            # Basic validation
            if not data_point.symbol or data_point.price <= 0:
                return None
            
            # Timestamp validation
            now = datetime.now()
            if data_point.timestamp > now + timedelta(minutes=5):
                self.logger.warning(f"Future timestamp detected: {data_point.timestamp}")
                data_point.timestamp = now
            
            # Price validation (basic outlier detection)
            if self._is_price_outlier(data_point, stream_id):
                data_point.quality_score *= 0.5
                self.logger.warning(f"Potential price outlier: {data_point.symbol} @ {data_point.price}")
            
            # Volume validation
            if data_point.volume < 0:
                data_point.volume = 0
            
            return data_point
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return None
    
    def _is_price_outlier(self, data_point: MarketDataPoint, stream_id: str) -> bool:
        """Simple outlier detection based on recent prices"""
        # Get recent prices from cache
        recent_prices = self._get_recent_prices(data_point.symbol, minutes=10)
        
        if len(recent_prices) < 3:
            return False  # Not enough data for outlier detection
        
        # Simple z-score based outlier detection
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price == 0:
            return False
        
        z_score = abs(data_point.price - mean_price) / std_price
        return z_score > 3.0  # 3-sigma rule
    
    def _get_recent_prices(self, symbol: str, minutes: int = 10) -> List[float]:
        """Get recent prices for a symbol from cache"""
        if not self.redis_client:
            return []
        
        try:
            # Get recent data from Redis
            key = f"prices:{symbol}"
            recent_data = self.redis_client.lrange(key, 0, 100)  # Last 100 points
            
            prices = []
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            for data_json in recent_data:
                try:
                    data = json.loads(data_json)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if timestamp >= cutoff_time:
                        prices.append(data['price'])
                except Exception:
                    continue
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting recent prices: {e}")
            return []
    
    def _cache_data_point(self, data_point: MarketDataPoint):
        """Cache data point for future reference"""
        if not self.redis_client:
            return
        
        try:
            # Store in Redis with expiration
            key = f"prices:{data_point.symbol}"
            data_json = json.dumps({
                'timestamp': data_point.timestamp.isoformat(),
                'price': data_point.price,
                'volume': data_point.volume,
                'source': data_point.source,
                'quality_score': data_point.quality_score
            })
            
            # Add to list and trim to keep only recent data
            self.redis_client.lpush(key, data_json)
            self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 points
            self.redis_client.expire(key, 3600)  # Expire after 1 hour
            
        except Exception as e:
            self.logger.error(f"Error caching data point: {e}")
    
    def _update_quality_metrics(self, stream_id: str, message_count: int, error: Optional[Exception]):
        """Update quality metrics for stream"""
        metrics = self.quality_metrics[stream_id]
        
        metrics['total_messages'] += message_count
        
        if error:
            metrics['error_count'] += 1
            metrics['last_error'] = str(error)
            
            # Decrease quality score
            error_rate = metrics['error_count'] / max(metrics['total_messages'], 1)
            metrics['data_quality_score'] = max(0.1, 1.0 - error_rate)
        else:
            # Improve quality score gradually
            metrics['data_quality_score'] = min(1.0, metrics['data_quality_score'] + 0.01)
        
        self.last_update_times[stream_id] = datetime.now()
    
    def _get_sleep_interval(self, update_frequency: str) -> float:
        """Convert update frequency string to sleep interval"""
        if update_frequency == 'realtime':
            return 0.1  # 100ms
        elif update_frequency.endswith('s'):
            return float(update_frequency[:-1])
        elif update_frequency.endswith('m'):
            return float(update_frequency[:-1]) * 60
        elif update_frequency.endswith('h'):
            return float(update_frequency[:-1]) * 3600
        else:
            return 1.0  # Default 1 second
    
    def _create_alpha_vantage_stream(self, config: StreamConfig) -> Dict[str, Any]:
        """Create Alpha Vantage data stream handler"""
        async def fetch_alpha_vantage_data(cfg: StreamConfig) -> List[MarketDataPoint]:
            # Mock implementation - replace with actual Alpha Vantage API calls
            data_points = []
            for symbol in cfg.symbols:
                # Simulate market data
                price = 100 + np.random.normal(0, 2)
                volume = int(np.random.exponential(1000))
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=price,
                    volume=volume,
                    source="alpha_vantage"
                )
                data_points.append(data_point)
            
            return data_points
        
        return {'fetch_function': fetch_alpha_vantage_data}
    
    def _create_iex_stream(self, config: StreamConfig) -> Dict[str, Any]:
        """Create IEX Cloud data stream handler"""
        async def fetch_iex_data(cfg: StreamConfig) -> List[MarketDataPoint]:
            # Mock implementation - replace with actual IEX Cloud API calls
            data_points = []
            for symbol in cfg.symbols:
                price = 100 + np.random.normal(0, 1.5)
                volume = int(np.random.exponential(800))
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=price,
                    volume=volume,
                    bid=price - 0.01,
                    ask=price + 0.01,
                    source="iex_cloud"
                )
                data_points.append(data_point)
            
            return data_points
        
        return {'fetch_function': fetch_iex_data}
    
    def _create_polygon_stream(self, config: StreamConfig) -> Dict[str, Any]:
        """Create Polygon.io data stream handler"""
        async def fetch_polygon_data(cfg: StreamConfig) -> List[MarketDataPoint]:
            # Mock implementation - replace with actual Polygon API calls
            data_points = []
            for symbol in cfg.symbols:
                price = 100 + np.random.normal(0, 1.8)
                volume = int(np.random.exponential(1200))
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=price,
                    volume=volume,
                    high=price + np.random.uniform(0, 2),
                    low=price - np.random.uniform(0, 2),
                    source="polygon"
                )
                data_points.append(data_point)
            
            return data_points
        
        return {'fetch_function': fetch_polygon_data}
    
    def _create_websocket_stream(self, config: StreamConfig) -> Dict[str, Any]:
        """Create WebSocket data stream handler"""
        async def fetch_websocket_data(cfg: StreamConfig) -> List[MarketDataPoint]:
            # Mock WebSocket implementation
            data_points = []
            for symbol in cfg.symbols:
                price = 100 + np.random.normal(0, 2.2)
                volume = int(np.random.exponential(900))
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=price,
                    volume=volume,
                    source="websocket"
                )
                data_points.append(data_point)
            
            return data_points
        
        return {'fetch_function': fetch_websocket_data}
    
    def _create_mock_stream(self, config: StreamConfig) -> Dict[str, Any]:
        """Create mock data stream for testing"""
        async def fetch_mock_data(cfg: StreamConfig) -> List[MarketDataPoint]:
            data_points = []
            for symbol in cfg.symbols:
                # Generate realistic mock data with trends
                base_price = 100
                trend = np.sin(datetime.now().timestamp() / 3600) * 5  # Hourly trend
                noise = np.random.normal(0, 1)
                price = base_price + trend + noise
                
                volume = int(np.random.exponential(1000))
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=max(0.01, price),  # Ensure positive price
                    volume=volume,
                    bid=price - 0.01,
                    ask=price + 0.01,
                    source="mock"
                )
                data_points.append(data_point)
            
            return data_points
        
        return {'fetch_function': fetch_mock_data}
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get status of all streams"""
        status = {}
        
        for stream_id, config in self.stream_configs.items():
            is_active = stream_id in self.active_streams
            metrics = self.quality_metrics.get(stream_id, {})
            last_update = self.last_update_times.get(stream_id)
            
            status[stream_id] = {
                'active': is_active,
                'provider': config.provider,
                'symbols': config.symbols,
                'update_frequency': config.update_frequency,
                'total_messages': metrics.get('total_messages', 0),
                'error_count': metrics.get('error_count', 0),
                'quality_score': metrics.get('data_quality_score', 1.0),
                'last_update': last_update.isoformat() if last_update else None,
                'last_error': metrics.get('last_error')
            }
        
        return status
    
    def get_latest_data(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest data points for a symbol"""
        if not self.redis_client:
            return []
        
        try:
            key = f"prices:{symbol}"
            recent_data = self.redis_client.lrange(key, 0, limit - 1)
            
            data_points = []
            for data_json in recent_data:
                try:
                    data = json.loads(data_json)
                    data_points.append(data)
                except Exception:
                    continue
            
            return data_points
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return []