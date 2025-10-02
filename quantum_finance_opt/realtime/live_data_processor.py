"""
Live Data Processor

Real-time data processing system that combines user CSV files with live market data,
news sentiment, and real-time feeds to create the ultimate portfolio optimization experience.
"""

import logging
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Real-time data imports
try:
    import yfinance as yf
    from yflive import QuoteStreamer
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWS_API_AVAILABLE = True
except ImportError:
    NEWS_API_AVAILABLE = False

try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    import requests
    import feedparser
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


class LiveDataProcessor:
    """
    Advanced real-time data processor that combines user data with live market feeds
    """
    
    def __init__(self, 
                 api_keys: Dict[str, str] = None,
                 update_frequency: int = 5,
                 enable_news: bool = True,
                 enable_sentiment: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys or {}
        self.update_frequency = update_frequency
        self.enable_news = enable_news
        self.enable_sentiment = enable_sentiment
        
        # Data storage
        self.user_data = None
        self.live_prices = {}
        self.news_sentiment = {}
        self.market_indicators = {}
        
        # Real-time components
        self.quote_streamer = None
        self.finnhub_client = None
        self.news_client = None
        self.sentiment_analyzer = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.stop_event = threading.Event()
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize real-time data services"""
        
        self.logger.info("Initializing live data services...")
        
        # Yahoo Finance Live
        if YF_AVAILABLE:
            try:
                self.quote_streamer = QuoteStreamer()
                self.logger.info("✓ Yahoo Finance Live initialized")
            except Exception as e:
                self.logger.warning(f"Yahoo Finance Live failed: {e}")
        
        # Finnhub
        if FINNHUB_AVAILABLE and 'finnhub' in self.api_keys:
            try:
                self.finnhub_client = finnhub.Client(api_key=self.api_keys['finnhub'])
                self.logger.info("✓ Finnhub initialized")
            except Exception as e:
                self.logger.warning(f"Finnhub initialization failed: {e}")
        
        # News API
        if NEWS_API_AVAILABLE and self.enable_news and 'newsapi' in self.api_keys:
            try:
                self.news_client = NewsApiClient(api_key=self.api_keys['newsapi'])
                self.logger.info("✓ News API initialized")
            except Exception as e:
                self.logger.warning(f"News API initialization failed: {e}")
        
        # Sentiment Analysis
        if SENTIMENT_AVAILABLE and self.enable_sentiment:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.logger.info("✓ Sentiment analysis initialized")
            except Exception as e:
                self.logger.warning(f"Sentiment analysis initialization failed: {e}")
    
    def load_user_portfolio(self, csv_path: str, 
                           price_column: str = 'Close',
                           date_column: str = 'Date',
                           symbol_column: str = 'Symbol') -> pd.DataFrame:
        """
        Load user's portfolio data from CSV file
        
        Args:
            csv_path: Path to user's CSV file
            price_column: Name of price column
            date_column: Name of date column  
            symbol_column: Name of symbol column
            
        Returns:
            Processed portfolio DataFrame
        """
        
        self.logger.info(f"Loading user portfolio from {csv_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Detect CSV format and process accordingly
            if symbol_column in df.columns:
                # Long format: Date, Symbol, Price
                df = self._process_long_format(df, date_column, symbol_column, price_column)
            else:
                # Wide format: Date, AAPL, GOOGL, MSFT, etc.
                df = self._process_wide_format(df, date_column)
            
            # Validate and clean data
            df = self._validate_user_data(df)
            
            # Store user data
            self.user_data = df
            
            # Extract symbols for live data
            symbols = [col for col in df.columns if col != 'Date']
            
            self.logger.info(f"✓ Loaded portfolio with {len(symbols)} assets: {symbols}")
            
            # Start live data feeds for these symbols
            self._start_live_feeds(symbols)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load user portfolio: {e}")
            raise QuantumFinanceOptError(f"Portfolio loading failed: {e}")
    
    def _process_long_format(self, df: pd.DataFrame, 
                           date_col: str, symbol_col: str, price_col: str) -> pd.DataFrame:
        """Process long format CSV (Date, Symbol, Price)"""
        
        # Convert to wide format
        df[date_col] = pd.to_datetime(df[date_col])
        pivot_df = df.pivot(index=date_col, columns=symbol_col, values=price_col)
        pivot_df.reset_index(inplace=True)
        pivot_df.columns.name = None
        
        return pivot_df
    
    def _process_wide_format(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Process wide format CSV (Date, AAPL, GOOGL, etc.)"""
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        return df
    
    def _validate_user_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean user data"""
        
        # Remove non-numeric columns except Date
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        # Keep Date and numeric columns
        keep_cols = list(date_cols) + list(numeric_cols)
        df = df[keep_cols]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove columns with all zeros or NaN
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.dropna(axis=1, how='all')
        
        # Ensure positive prices
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].abs()
        
        return df
    
    def _start_live_feeds(self, symbols: List[str]):
        """Start live data feeds for portfolio symbols"""
        
        self.logger.info(f"Starting live feeds for {len(symbols)} symbols")
        
        # Start price feeds
        if self.quote_streamer:
            self.executor.submit(self._run_price_stream, symbols)
        
        # Start news feeds
        if self.enable_news:
            self.executor.submit(self._run_news_stream, symbols)
        
        # Start market indicators
        self.executor.submit(self._run_market_indicators)
    
    def _run_price_stream(self, symbols: List[str]):
        """Run real-time price streaming"""
        
        self.logger.info("Starting real-time price stream")
        
        while not self.stop_event.is_set():
            try:
                # Get live quotes
                for symbol in symbols:
                    try:
                        # Yahoo Finance real-time
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        if 'regularMarketPrice' in info:
                            price = info['regularMarketPrice']
                            volume = info.get('regularMarketVolume', 0)
                            
                            self.live_prices[symbol] = {
                                'price': price,
                                'volume': volume,
                                'timestamp': datetime.now(),
                                'change': info.get('regularMarketChange', 0),
                                'change_percent': info.get('regularMarketChangePercent', 0),
                                'bid': info.get('bid', price),
                                'ask': info.get('ask', price),
                                'high': info.get('dayHigh', price),
                                'low': info.get('dayLow', price)
                            }
                        
                        # Finnhub real-time (if available)
                        if self.finnhub_client:
                            try:
                                quote = self.finnhub_client.quote(symbol)
                                if quote and 'c' in quote:  # current price
                                    self.live_prices[symbol].update({
                                        'finnhub_price': quote['c'],
                                        'finnhub_change': quote.get('d', 0),
                                        'finnhub_change_percent': quote.get('dp', 0),
                                        'high': quote.get('h', 0),
                                        'low': quote.get('l', 0),
                                        'open': quote.get('o', 0),
                                        'previous_close': quote.get('pc', 0)
                                    })
                            except Exception as e:
                                self.logger.debug(f"Finnhub quote failed for {symbol}: {e}")
                        
                    except Exception as e:
                        self.logger.debug(f"Price update failed for {symbol}: {e}")
                
                # Wait before next update
                time.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Price stream error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _run_news_stream(self, symbols: List[str]):
        """Run real-time news and sentiment analysis"""
        
        if not self.enable_news:
            return
        
        self.logger.info("Starting real-time news stream")
        
        while not self.stop_event.is_set():
            try:
                for symbol in symbols:
                    # Get news from multiple sources
                    news_items = []
                    
                    # News API
                    if self.news_client:
                        try:
                            articles = self.news_client.get_everything(
                                q=f"{symbol} stock",
                                language='en',
                                sort_by='publishedAt',
                                page_size=5,
                                from_param=(datetime.now() - timedelta(hours=1)).isoformat()
                            )
                            
                            if articles and 'articles' in articles:
                                for article in articles['articles']:
                                    news_items.append({
                                        'title': article.get('title', ''),
                                        'description': article.get('description', ''),
                                        'content': article.get('content', ''),
                                        'source': article.get('source', {}).get('name', 'Unknown'),
                                        'published_at': article.get('publishedAt', ''),
                                        'url': article.get('url', '')
                                    })
                        except Exception as e:
                            self.logger.debug(f"News API failed for {symbol}: {e}")
                    
                    # Yahoo Finance news
                    try:
                        ticker = yf.Ticker(symbol)
                        news = ticker.news
                        
                        for item in news[:5]:  # Top 5 news items
                            news_items.append({
                                'title': item.get('title', ''),
                                'description': item.get('summary', ''),
                                'source': item.get('publisher', 'Yahoo Finance'),
                                'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                                'url': item.get('link', '')
                            })
                    except Exception as e:
                        self.logger.debug(f"Yahoo Finance news failed for {symbol}: {e}")
                    
                    # Analyze sentiment
                    if news_items and self.sentiment_analyzer:
                        sentiment_scores = []
                        
                        for item in news_items:
                            text = f"{item['title']} {item['description']}"
                            
                            # VADER sentiment
                            vader_score = self.sentiment_analyzer.polarity_scores(text)
                            
                            # TextBlob sentiment (if available)
                            textblob_score = 0
                            if SENTIMENT_AVAILABLE:
                                try:
                                    blob = TextBlob(text)
                                    textblob_score = blob.sentiment.polarity
                                except:
                                    pass
                            
                            sentiment_scores.append({
                                'vader_compound': vader_score['compound'],
                                'vader_positive': vader_score['pos'],
                                'vader_negative': vader_score['neg'],
                                'vader_neutral': vader_score['neu'],
                                'textblob_polarity': textblob_score,
                                'text': text[:200],  # First 200 chars
                                'timestamp': datetime.now()
                            })
                        
                        # Calculate aggregate sentiment
                        if sentiment_scores:
                            avg_vader = np.mean([s['vader_compound'] for s in sentiment_scores])
                            avg_textblob = np.mean([s['textblob_polarity'] for s in sentiment_scores])
                            
                            self.news_sentiment[symbol] = {
                                'avg_vader_sentiment': avg_vader,
                                'avg_textblob_sentiment': avg_textblob,
                                'news_count': len(news_items),
                                'sentiment_scores': sentiment_scores,
                                'last_update': datetime.now()
                            }
                
                # Wait before next news update (longer interval)
                time.sleep(self.update_frequency * 12)  # Update every minute if update_frequency=5s
                
            except Exception as e:
                self.logger.error(f"News stream error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _run_market_indicators(self):
        """Run market indicators and economic data"""
        
        self.logger.info("Starting market indicators stream")
        
        while not self.stop_event.is_set():
            try:
                # Market indices
                indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow, NASDAQ, VIX
                
                for index in indices:
                    try:
                        ticker = yf.Ticker(index)
                        info = ticker.info
                        
                        if 'regularMarketPrice' in info:
                            self.market_indicators[index] = {
                                'price': info['regularMarketPrice'],
                                'change': info.get('regularMarketChange', 0),
                                'change_percent': info.get('regularMarketChangePercent', 0),
                                'timestamp': datetime.now()
                            }
                    except Exception as e:
                        self.logger.debug(f"Market indicator failed for {index}: {e}")
                
                # Currency rates (if relevant)
                currencies = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
                
                for currency in currencies:
                    try:
                        ticker = yf.Ticker(currency)
                        info = ticker.info
                        
                        if 'regularMarketPrice' in info:
                            self.market_indicators[currency] = {
                                'rate': info['regularMarketPrice'],
                                'change': info.get('regularMarketChange', 0),
                                'timestamp': datetime.now()
                            }
                    except Exception as e:
                        self.logger.debug(f"Currency rate failed for {currency}: {e}")
                
                # Wait before next update (longer interval for market indicators)
                time.sleep(self.update_frequency * 6)  # Update every 30s if update_frequency=5s
                
            except Exception as e:
                self.logger.error(f"Market indicators error: {e}")
                time.sleep(30)
    
    def get_enhanced_portfolio_data(self) -> Dict[str, Any]:
        """
        Get enhanced portfolio data combining user data with live feeds
        
        Returns:
            Dictionary with enhanced portfolio information
        """
        
        if self.user_data is None:
            raise QuantumFinanceOptError("No user portfolio data loaded")
        
        # Get latest user data
        latest_user_data = self.user_data.iloc[-1].to_dict()
        
        # Combine with live data
        enhanced_data = {
            'timestamp': datetime.now(),
            'user_data': latest_user_data,
            'live_prices': self.live_prices.copy(),
            'news_sentiment': self.news_sentiment.copy(),
            'market_indicators': self.market_indicators.copy()
        }
        
        # Calculate enhanced metrics
        symbols = [col for col in self.user_data.columns if col != 'Date']
        
        for symbol in symbols:
            if symbol in self.live_prices:
                live_price = self.live_prices[symbol]['price']
                historical_price = latest_user_data.get(symbol, live_price)
                
                # Calculate real-time return
                if historical_price > 0:
                    rt_return = (live_price - historical_price) / historical_price
                    enhanced_data['live_prices'][symbol]['rt_return'] = rt_return
                
                # Add sentiment impact
                if symbol in self.news_sentiment:
                    sentiment = self.news_sentiment[symbol]['avg_vader_sentiment']
                    enhanced_data['live_prices'][symbol]['sentiment_score'] = sentiment
                    
                    # Sentiment-adjusted return (simple model)
                    sentiment_adjustment = sentiment * 0.1  # 10% max adjustment
                    enhanced_data['live_prices'][symbol]['sentiment_adjusted_return'] = rt_return + sentiment_adjustment
        
        return enhanced_data
    
    def get_optimization_ready_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get data ready for portfolio optimization
        
        Returns:
            Tuple of (price_dataframe, metadata)
        """
        
        if self.user_data is None:
            raise QuantumFinanceOptError("No user portfolio data loaded")
        
        # Create enhanced price DataFrame
        enhanced_df = self.user_data.copy()
        
        # Add live prices as the latest row
        if self.live_prices:
            live_row = {'Date': datetime.now()}
            
            for symbol in enhanced_df.columns:
                if symbol != 'Date' and symbol in self.live_prices:
                    live_row[symbol] = self.live_prices[symbol]['price']
                elif symbol != 'Date':
                    # Use last known price if live data not available
                    live_row[symbol] = enhanced_df[symbol].iloc[-1]
            
            # Add live row
            live_df = pd.DataFrame([live_row])
            enhanced_df = pd.concat([enhanced_df, live_df], ignore_index=True)
        
        # Metadata for optimization
        metadata = {
            'live_data_available': len(self.live_prices) > 0,
            'news_sentiment_available': len(self.news_sentiment) > 0,
            'market_indicators_available': len(self.market_indicators) > 0,
            'last_update': datetime.now(),
            'data_quality_score': self._calculate_data_quality(),
            'sentiment_scores': self.news_sentiment,
            'market_context': self.market_indicators
        }
        
        return enhanced_df, metadata
    
    def _calculate_data_quality(self) -> float:
        """Calculate overall data quality score"""
        
        quality_score = 0.5  # Base score
        
        # Live price data quality
        if self.live_prices:
            live_data_ratio = len(self.live_prices) / len([col for col in self.user_data.columns if col != 'Date'])
            quality_score += 0.3 * live_data_ratio
        
        # News sentiment quality
        if self.news_sentiment:
            sentiment_ratio = len(self.news_sentiment) / len([col for col in self.user_data.columns if col != 'Date'])
            quality_score += 0.2 * sentiment_ratio
        
        return min(1.0, quality_score)
    
    def stop_live_feeds(self):
        """Stop all live data feeds"""
        
        self.logger.info("Stopping live data feeds")
        
        self.stop_event.set()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("✓ Live data feeds stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        
        return {
            'user_data_loaded': self.user_data is not None,
            'live_prices_count': len(self.live_prices),
            'news_sentiment_count': len(self.news_sentiment),
            'market_indicators_count': len(self.market_indicators),
            'services': {
                'yahoo_finance': YF_AVAILABLE,
                'finnhub': FINNHUB_AVAILABLE and self.finnhub_client is not None,
                'news_api': NEWS_API_AVAILABLE and self.news_client is not None,
                'sentiment_analysis': SENTIMENT_AVAILABLE and self.sentiment_analyzer is not None
            },
            'last_update': datetime.now(),
            'data_quality_score': self._calculate_data_quality() if self.user_data is not None else 0.0
        }