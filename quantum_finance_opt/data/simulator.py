"""
Data simulation module for QuantumFinanceOpt.

This module generates realistic financial data, news sentiment data,
and ESG scores for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import random
import string

from ..core.exceptions import DataProcessingError
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class DataSimulator:
    """Simulator for financial data, news, and ESG scores."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize DataSimulator with configuration."""
        self.config = config
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
    
    def generate_sample_data(self, tickers: List[str], periods: int, 
                           start_date: str = '2020-01-01') -> pd.DataFrame:
        """
        Generate realistic financial price data using geometric Brownian motion.
        
        Args:
            tickers: List of ticker symbols
            periods: Number of time periods
            start_date: Start date for data generation
            
        Returns:
            DataFrame with simulated price data
            
        Raises:
            DataProcessingError: If data generation fails
        """
        try:
            logger.info(f"Generating sample data for {len(tickers)} tickers over {periods} periods")
            
            # Create date range
            start = pd.to_datetime(start_date)
            dates = pd.date_range(start=start, periods=periods, freq='D')
            
            # Initialize price data
            price_data = pd.DataFrame(index=dates, columns=tickers)
            
            # Generate realistic parameters for each ticker
            ticker_params = self._generate_ticker_parameters(tickers)
            
            for ticker in tickers:
                params = ticker_params[ticker]
                
                # Generate price series using geometric Brownian motion
                prices = self._generate_gbm_prices(
                    initial_price=params['initial_price'],
                    drift=params['drift'],
                    volatility=params['volatility'],
                    periods=periods
                )
                
                price_data[ticker] = prices
            
            # Add some correlation between assets
            price_data = self._add_market_correlation(price_data)
            
            # Add market events (crashes, rallies)
            price_data = self._add_market_events(price_data)
            
            logger.info(f"Sample data generated successfully. Shape: {price_data.shape}")
            logger.info(f"Price ranges: {price_data.min().to_dict()} to {price_data.max().to_dict()}")
            
            return price_data
            
        except Exception as e:
            raise DataProcessingError(f"Sample data generation failed: {e}")
    
    def _generate_ticker_parameters(self, tickers: List[str]) -> Dict[str, Dict]:
        """Generate realistic parameters for each ticker."""
        
        # Base parameters for different asset classes
        asset_classes = {
            'tech': {'base_price': 150, 'drift': 0.12, 'volatility': 0.25},
            'finance': {'base_price': 80, 'drift': 0.08, 'volatility': 0.20},
            'healthcare': {'base_price': 120, 'drift': 0.10, 'volatility': 0.18},
            'energy': {'base_price': 60, 'drift': 0.06, 'volatility': 0.30},
            'consumer': {'base_price': 100, 'drift': 0.09, 'volatility': 0.22}
        }
        
        ticker_params = {}
        
        for ticker in tickers:
            # Assign asset class based on ticker (simplified)
            if ticker in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']:
                asset_class = 'tech'
            elif ticker in ['JPM', 'BAC', 'WFC', 'GS']:
                asset_class = 'finance'
            elif ticker in ['JNJ', 'PFE', 'UNH', 'ABBV']:
                asset_class = 'healthcare'
            elif ticker in ['XOM', 'CVX', 'COP']:
                asset_class = 'energy'
            else:
                asset_class = 'consumer'
            
            base_params = asset_classes[asset_class]
            
            # Add some randomness to parameters
            ticker_params[ticker] = {
                'initial_price': base_params['base_price'] * (0.8 + 0.4 * np.random.random()),
                'drift': base_params['drift'] * (0.7 + 0.6 * np.random.random()),
                'volatility': base_params['volatility'] * (0.8 + 0.4 * np.random.random()),
                'asset_class': asset_class
            }
        
        return ticker_params
    
    def _generate_gbm_prices(self, initial_price: float, drift: float, 
                           volatility: float, periods: int) -> np.ndarray:
        """Generate prices using geometric Brownian motion."""
        
        dt = 1/252  # Daily time step (252 trading days per year)
        
        # Generate random shocks
        shocks = np.random.normal(0, 1, periods)
        
        # Calculate price changes
        price_changes = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
        
        # Generate price series
        log_prices = np.cumsum(price_changes)
        prices = initial_price * np.exp(log_prices)
        
        return prices
    
    def _add_market_correlation(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add correlation between assets to make data more realistic."""
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Create correlation matrix (simplified)
        n_assets = len(price_data.columns)
        correlation_matrix = np.eye(n_assets)
        
        # Add some correlation
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                correlation = 0.3 + 0.4 * np.random.random()  # Correlation between 0.3 and 0.7
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        # Apply correlation to returns (simplified approach)
        correlated_returns = returns.copy()
        
        # Add market factor
        market_factor = np.random.normal(0, 0.01, len(returns))
        for col in correlated_returns.columns:
            correlated_returns[col] += 0.3 * market_factor
        
        # Reconstruct prices
        correlated_prices = price_data.iloc[:1].copy()  # Keep first row
        
        for i in range(1, len(price_data)):
            new_prices = correlated_prices.iloc[-1] * (1 + correlated_returns.iloc[i-1])
            correlated_prices = pd.concat([correlated_prices, new_prices.to_frame().T])
        
        correlated_prices.index = price_data.index
        
        return correlated_prices
    
    def _add_market_events(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add market events like crashes and rallies."""
        
        modified_data = price_data.copy()
        n_periods = len(price_data)
        
        # Add 1-2 market crashes
        n_crashes = np.random.randint(1, 3)
        for _ in range(n_crashes):
            crash_start = np.random.randint(n_periods // 4, 3 * n_periods // 4)
            crash_duration = np.random.randint(5, 20)
            crash_magnitude = -0.15 - 0.15 * np.random.random()  # -15% to -30%
            
            for i in range(crash_start, min(crash_start + crash_duration, n_periods)):
                daily_impact = crash_magnitude / crash_duration
                modified_data.iloc[i] *= (1 + daily_impact)
        
        # Add 1-2 market rallies
        n_rallies = np.random.randint(1, 3)
        for _ in range(n_rallies):
            rally_start = np.random.randint(n_periods // 4, 3 * n_periods // 4)
            rally_duration = np.random.randint(10, 30)
            rally_magnitude = 0.10 + 0.15 * np.random.random()  # +10% to +25%
            
            for i in range(rally_start, min(rally_start + rally_duration, n_periods)):
                daily_impact = rally_magnitude / rally_duration
                modified_data.iloc[i] *= (1 + daily_impact)
        
        return modified_data
    
    def simulate_news_data(self, tickers: List[str], dates: List[str]) -> pd.DataFrame:
        """
        Generate mock news data for sentiment analysis.
        
        Args:
            tickers: List of ticker symbols
            dates: List of dates
            
        Returns:
            DataFrame with mock news data
        """
        try:
            logger.info(f"Generating mock news data for {len(tickers)} tickers over {len(dates)} dates")
            
            news_data = []
            
            # News templates for different sentiment categories
            positive_templates = [
                "{ticker} reports strong quarterly earnings, beating analyst expectations",
                "{ticker} announces breakthrough innovation in {sector}",
                "{ticker} stock surges on positive analyst upgrade",
                "{ticker} expands operations with new strategic partnership",
                "{ticker} receives regulatory approval for new product launch",
                "{ticker} CEO optimistic about future growth prospects",
                "{ticker} announces increased dividend and share buyback program"
            ]
            
            negative_templates = [
                "{ticker} faces regulatory scrutiny over {issue}",
                "{ticker} reports disappointing quarterly results",
                "{ticker} stock falls on analyst downgrade",
                "{ticker} announces layoffs amid restructuring efforts",
                "{ticker} faces supply chain disruptions",
                "{ticker} CEO steps down amid controversy",
                "{ticker} loses major client contract"
            ]
            
            neutral_templates = [
                "{ticker} maintains steady performance in Q{quarter}",
                "{ticker} announces routine board meeting",
                "{ticker} files standard regulatory documents",
                "{ticker} participates in industry conference",
                "{ticker} releases sustainability report",
                "{ticker} announces minor organizational changes",
                "{ticker} provides business update to investors"
            ]
            
            sectors = ['technology', 'healthcare', 'finance', 'energy', 'consumer goods']
            issues = ['data privacy', 'environmental impact', 'market practices', 'compliance']
            
            for date in dates:
                for ticker in tickers:
                    # Generate 0-3 news items per ticker per date
                    n_news = np.random.poisson(1)  # Average 1 news item per day
                    
                    for _ in range(n_news):
                        # Choose sentiment
                        sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                                        p=[0.4, 0.3, 0.3])
                        
                        if sentiment_type == 'positive':
                            template = np.random.choice(positive_templates)
                            sentiment_score = 0.3 + 0.7 * np.random.random()  # 0.3 to 1.0
                        elif sentiment_type == 'negative':
                            template = np.random.choice(negative_templates)
                            sentiment_score = -0.3 - 0.7 * np.random.random()  # -1.0 to -0.3
                        else:
                            template = np.random.choice(neutral_templates)
                            sentiment_score = -0.2 + 0.4 * np.random.random()  # -0.2 to 0.2
                        
                        # Fill template
                        news_text = template.format(
                            ticker=ticker,
                            sector=np.random.choice(sectors),
                            issue=np.random.choice(issues),
                            quarter=np.random.randint(1, 5)
                        )
                        
                        news_data.append({
                            'date': date,
                            'ticker': ticker,
                            'text': news_text,
                            'sentiment_score': sentiment_score,
                            'sentiment_type': sentiment_type
                        })
            
            news_df = pd.DataFrame(news_data)
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            logger.info(f"Generated {len(news_df)} news items")
            logger.info(f"Sentiment distribution: {news_df['sentiment_type'].value_counts().to_dict()}")
            
            return news_df
            
        except Exception as e:
            raise DataProcessingError(f"News data simulation failed: {e}")
    
    def simulate_esg_scores(self, tickers: List[str]) -> Dict[str, float]:
        """
        Generate realistic ESG scores for tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping tickers to ESG scores (0-1)
        """
        try:
            logger.info(f"Generating ESG scores for {len(tickers)} tickers")
            
            esg_scores = {}
            
            # ESG score distributions by sector
            sector_esg_params = {
                'tech': {'mean': 0.7, 'std': 0.15},
                'healthcare': {'mean': 0.65, 'std': 0.12},
                'finance': {'mean': 0.55, 'std': 0.18},
                'energy': {'mean': 0.35, 'std': 0.20},
                'consumer': {'mean': 0.60, 'std': 0.15}
            }
            
            for ticker in tickers:
                # Determine sector (simplified)
                if ticker in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']:
                    sector = 'tech'
                elif ticker in ['JPM', 'BAC', 'WFC', 'GS']:
                    sector = 'finance'
                elif ticker in ['JNJ', 'PFE', 'UNH', 'ABBV']:
                    sector = 'healthcare'
                elif ticker in ['XOM', 'CVX', 'COP']:
                    sector = 'energy'
                else:
                    sector = 'consumer'
                
                params = sector_esg_params[sector]
                
                # Generate ESG score
                esg_score = np.random.normal(params['mean'], params['std'])
                esg_score = np.clip(esg_score, 0, 1)  # Ensure score is between 0 and 1
                
                esg_scores[ticker] = round(esg_score, 3)
            
            logger.info(f"ESG scores generated: {esg_scores}")
            
            return esg_scores
            
        except Exception as e:
            raise DataProcessingError(f"ESG score simulation failed: {e}")
    
    def save_generated_data(self, price_data: pd.DataFrame, news_data: pd.DataFrame, 
                          esg_scores: Dict[str, float], output_dir: str = 'data'):
        """
        Save generated data to CSV files.
        
        Args:
            price_data: Generated price data
            news_data: Generated news data
            esg_scores: Generated ESG scores
            output_dir: Output directory
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Save price data
            price_file = os.path.join(output_dir, 'sample_prices.csv')
            price_data.to_csv(price_file)
            logger.info(f"Price data saved to {price_file}")
            
            # Save news data
            news_file = os.path.join(output_dir, 'sample_news.csv')
            news_data.to_csv(news_file, index=False)
            logger.info(f"News data saved to {news_file}")
            
            # Save ESG scores
            esg_file = os.path.join(output_dir, 'sample_esg_scores.csv')
            esg_df = pd.DataFrame(list(esg_scores.items()), columns=['ticker', 'esg_score'])
            esg_df.to_csv(esg_file, index=False)
            logger.info(f"ESG scores saved to {esg_file}")
            
        except Exception as e:
            raise DataProcessingError(f"Failed to save generated data: {e}")
    
    def generate_complete_dataset(self, save_to_disk: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Generate complete dataset with prices, news, and ESG scores.
        
        Args:
            save_to_disk: Whether to save data to disk
            
        Returns:
            Tuple of (price_data, news_data, esg_scores)
        """
        try:
            logger.info("Generating complete dataset")
            
            # Calculate periods from date range
            start_date = pd.to_datetime(self.config.start_date)
            end_date = pd.to_datetime(self.config.end_date)
            periods = (end_date - start_date).days + 1
            
            # Generate price data
            price_data = self.generate_sample_data(
                tickers=self.config.tickers,
                periods=periods,
                start_date=self.config.start_date
            )
            
            # Generate news data
            dates = price_data.index.strftime('%Y-%m-%d').tolist()
            news_data = self.simulate_news_data(
                tickers=self.config.tickers,
                dates=dates
            )
            
            # Generate ESG scores
            esg_scores = self.simulate_esg_scores(self.config.tickers)
            
            if save_to_disk:
                self.save_generated_data(price_data, news_data, esg_scores, self.config.output_dir)
            
            logger.info("Complete dataset generation finished")
            
            return price_data, news_data, esg_scores
            
        except Exception as e:
            raise DataProcessingError(f"Complete dataset generation failed: {e}")


# Unit tests for DataSimulator
def test_data_simulator():
    """Unit tests for DataSimulator functionality."""
    
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL', 'MSFT'],
        start_date='2020-01-01',
        end_date='2020-03-31',
        random_seed=42
    )
    
    simulator = DataSimulator(config)
    
    try:
        # Test price data generation
        price_data = simulator.generate_sample_data(['AAPL', 'GOOGL'], 90)
        assert not price_data.empty, "Price data should not be empty"
        assert price_data.shape == (90, 2), f"Expected shape (90, 2), got {price_data.shape}"
        
        # Test news data generation
        dates = ['2020-01-01', '2020-01-02', '2020-01-03']
        news_data = simulator.simulate_news_data(['AAPL'], dates)
        assert not news_data.empty, "News data should not be empty"
        assert 'sentiment_score' in news_data.columns, "News data should have sentiment scores"
        
        # Test ESG scores generation
        esg_scores = simulator.simulate_esg_scores(['AAPL', 'GOOGL'])
        assert len(esg_scores) == 2, "Should generate ESG scores for all tickers"
        assert all(0 <= score <= 1 for score in esg_scores.values()), "ESG scores should be between 0 and 1"
        
        # Test complete dataset generation
        price_data, news_data, esg_scores = simulator.generate_complete_dataset(save_to_disk=False)
        assert not price_data.empty, "Complete price data should not be empty"
        assert not news_data.empty, "Complete news data should not be empty"
        assert len(esg_scores) > 0, "Complete ESG scores should not be empty"
        
        print("All DataSimulator tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_data_simulator()