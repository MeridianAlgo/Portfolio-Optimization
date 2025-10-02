"""
DeFi Integration for Portfolio Optimization

Integrates decentralized finance protocols for yield farming, liquidity mining,
and automated market making strategies in portfolio optimization.
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class DeFiPool:
    """DeFi liquidity pool information"""
    protocol: str
    pool_address: str
    token0: str
    token1: str
    apy: float
    tvl: float  # Total Value Locked
    volume_24h: float
    fees_24h: float
    impermanent_loss_risk: float
    smart_contract_risk: float


@dataclass
class YieldFarmingOpportunity:
    """Yield farming opportunity"""
    protocol: str
    pool_id: str
    tokens: List[str]
    apy: float
    rewards_token: str
    lock_period: int  # days
    minimum_deposit: float
    risk_score: float


class DeFiIntegration:
    """
    DeFi Integration for Advanced Portfolio Optimization
    
    Integrates with major DeFi protocols for yield optimization,
    liquidity provision, and automated market making.
    """
    
    def __init__(self, 
                 web3_provider: str = None,
                 api_keys: Dict[str, str] = None):
        
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys or {}
        
        # Web3 connection
        self.w3 = None
        if WEB3_AVAILABLE and web3_provider:
            try:
                self.w3 = Web3(Web3.HTTPProvider(web3_provider))
                if self.w3.isConnected():
                    self.logger.info("✓ Web3 connected")
                else:
                    self.logger.warning("Web3 connection failed")
            except Exception as e:
                self.logger.warning(f"Web3 initialization failed: {e}")
        
        # DeFi protocol APIs
        self.protocol_apis = {
            'uniswap_v3': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            'sushiswap': 'https://api.thegraph.com/subgraphs/name/sushiswap/exchange',
            'compound': 'https://api.compound.finance/api/v2',
            'aave': 'https://aave-api-v2.aave.com',
            'yearn': 'https://api.yearn.finance/v1/chains/1/vaults/all',
            'curve': 'https://api.curve.fi/api/getPools/ethereum/main'
        }
        
        # CEX integration for arbitrage
        self.exchanges = {}
        if CCXT_AVAILABLE:
            self._initialize_exchanges()
        
        # DeFi data cache
        self.pools_cache = {}
        self.yield_opportunities_cache = {}
        self.last_update = {}
    
    def _initialize_exchanges(self):
        """Initialize centralized exchanges for arbitrage detection"""
        
        exchange_configs = {
            'binance': {'apiKey': self.api_keys.get('binance_api'), 'secret': self.api_keys.get('binance_secret')},
            'coinbase': {'apiKey': self.api_keys.get('coinbase_api'), 'secret': self.api_keys.get('coinbase_secret')},
            'kraken': {'apiKey': self.api_keys.get('kraken_api'), 'secret': self.api_keys.get('kraken_secret')}
        }
        
        for exchange_name, config in exchange_configs.items():
            if config['apiKey'] and config['secret']:
                try:
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class(config)
                    self.logger.info(f"✓ {exchange_name} exchange initialized")
                except Exception as e:
                    self.logger.warning(f"{exchange_name} initialization failed: {e}")
    
    async def fetch_defi_pools(self, protocols: List[str] = None) -> List[DeFiPool]:
        """Fetch DeFi liquidity pools from multiple protocols"""
        
        if protocols is None:
            protocols = ['uniswap_v3', 'sushiswap', 'curve']
        
        all_pools = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for protocol in protocols:
                if protocol in self.protocol_apis:
                    tasks.append(self._fetch_protocol_pools(session, protocol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_pools.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Protocol fetch failed: {result}")
        
        # Cache results
        self.pools_cache = {pool.pool_address: pool for pool in all_pools}
        self.last_update['pools'] = datetime.now()
        
        return all_pools
    
    async def _fetch_protocol_pools(self, session: aiohttp.ClientSession, protocol: str) -> List[DeFiPool]:
        """Fetch pools from specific protocol"""
        
        pools = []
        
        try:
            if protocol == 'uniswap_v3':
                pools = await self._fetch_uniswap_pools(session)
            elif protocol == 'sushiswap':
                pools = await self._fetch_sushiswap_pools(session)
            elif protocol == 'curve':
                pools = await self._fetch_curve_pools(session)
            elif protocol == 'yearn':
                pools = await self._fetch_yearn_vaults(session)
        
        except Exception as e:
            self.logger.error(f"Failed to fetch {protocol} pools: {e}")
        
        return pools
    
    async def _fetch_uniswap_pools(self, session: aiohttp.ClientSession) -> List[DeFiPool]:
        """Fetch Uniswap V3 pools"""
        
        query = """
        {
          pools(first: 100, orderBy: totalValueLockedUSD, orderDirection: desc) {
            id
            token0 {
              symbol
            }
            token1 {
              symbol
            }
            totalValueLockedUSD
            volumeUSD
            feeTier
            liquidity
          }
        }
        """
        
        async with session.post(
            self.protocol_apis['uniswap_v3'],
            json={'query': query}
        ) as response:
            data = await response.json()
            
            pools = []
            for pool_data in data.get('data', {}).get('pools', []):
                # Calculate estimated APY (simplified)
                tvl = float(pool_data.get('totalValueLockedUSD', 0))
                volume = float(pool_data.get('volumeUSD', 0))
                fee_tier = float(pool_data.get('feeTier', 3000)) / 1000000  # Convert to percentage
                
                if tvl > 0:
                    daily_fees = volume * fee_tier
                    apy = (daily_fees * 365 / tvl) * 100
                else:
                    apy = 0
                
                pool = DeFiPool(
                    protocol='uniswap_v3',
                    pool_address=pool_data['id'],
                    token0=pool_data['token0']['symbol'],
                    token1=pool_data['token1']['symbol'],
                    apy=min(apy, 1000),  # Cap at 1000% APY
                    tvl=tvl,
                    volume_24h=volume,
                    fees_24h=daily_fees,
                    impermanent_loss_risk=self._calculate_il_risk(pool_data['token0']['symbol'], pool_data['token1']['symbol']),
                    smart_contract_risk=0.1  # Low risk for Uniswap
                )
                
                pools.append(pool)
            
            return pools
    
    async def _fetch_sushiswap_pools(self, session: aiohttp.ClientSession) -> List[DeFiPool]:
        """Fetch SushiSwap pools"""
        
        query = """
        {
          pairs(first: 100, orderBy: reserveUSD, orderDirection: desc) {
            id
            token0 {
              symbol
            }
            token1 {
              symbol
            }
            reserveUSD
            volumeUSD
            untrackedVolumeUSD
          }
        }
        """
        
        async with session.post(
            self.protocol_apis['sushiswap'],
            json={'query': query}
        ) as response:
            data = await response.json()
            
            pools = []
            for pool_data in data.get('data', {}).get('pairs', []):
                tvl = float(pool_data.get('reserveUSD', 0))
                volume = float(pool_data.get('volumeUSD', 0))
                
                # SushiSwap has 0.3% fee
                if tvl > 0:
                    daily_fees = volume * 0.003
                    apy = (daily_fees * 365 / tvl) * 100
                else:
                    apy = 0
                
                pool = DeFiPool(
                    protocol='sushiswap',
                    pool_address=pool_data['id'],
                    token0=pool_data['token0']['symbol'],
                    token1=pool_data['token1']['symbol'],
                    apy=min(apy, 1000),
                    tvl=tvl,
                    volume_24h=volume,
                    fees_24h=daily_fees,
                    impermanent_loss_risk=self._calculate_il_risk(pool_data['token0']['symbol'], pool_data['token1']['symbol']),
                    smart_contract_risk=0.15  # Slightly higher risk
                )
                
                pools.append(pool)
            
            return pools
    
    async def _fetch_curve_pools(self, session: aiohttp.ClientSession) -> List[DeFiPool]:
        """Fetch Curve Finance pools"""
        
        async with session.get(self.protocol_apis['curve']) as response:
            data = await response.json()
            
            pools = []
            for pool_data in data.get('data', {}).get('poolData', []):
                pool = DeFiPool(
                    protocol='curve',
                    pool_address=pool_data.get('address', ''),
                    token0=pool_data.get('coins', [{}])[0].get('symbol', ''),
                    token1=pool_data.get('coins', [{}])[1].get('symbol', '') if len(pool_data.get('coins', [])) > 1 else '',
                    apy=float(pool_data.get('apy', 0)),
                    tvl=float(pool_data.get('usdTotal', 0)),
                    volume_24h=float(pool_data.get('volumeUSD', 0)),
                    fees_24h=float(pool_data.get('volumeUSD', 0)) * 0.0004,  # 0.04% fee
                    impermanent_loss_risk=0.05,  # Lower IL risk for stablecoins
                    smart_contract_risk=0.1
                )
                
                pools.append(pool)
            
            return pools
    
    async def _fetch_yearn_vaults(self, session: aiohttp.ClientSession) -> List[DeFiPool]:
        """Fetch Yearn Finance vaults"""
        
        async with session.get(self.protocol_apis['yearn']) as response:
            data = await response.json()
            
            pools = []
            for vault_data in data:
                apy = vault_data.get('apy', {})
                net_apy = apy.get('net_apy', 0) * 100 if apy else 0
                
                pool = DeFiPool(
                    protocol='yearn',
                    pool_address=vault_data.get('address', ''),
                    token0=vault_data.get('token', {}).get('symbol', ''),
                    token1='',  # Single asset vaults
                    apy=net_apy,
                    tvl=float(vault_data.get('tvl', {}).get('totalAssets', 0)),
                    volume_24h=0,  # Not applicable for vaults
                    fees_24h=0,
                    impermanent_loss_risk=0,  # No IL for single asset
                    smart_contract_risk=0.2  # Higher complexity risk
                )
                
                pools.append(pool)
            
            return pools
    
    def _calculate_il_risk(self, token0: str, token1: str) -> float:
        """Calculate impermanent loss risk based on token correlation"""
        
        # Simplified IL risk calculation
        stablecoins = {'USDC', 'USDT', 'DAI', 'BUSD', 'FRAX'}
        
        if token0 in stablecoins and token1 in stablecoins:
            return 0.01  # Very low IL risk
        elif token0 in stablecoins or token1 in stablecoins:
            return 0.15  # Medium IL risk
        else:
            return 0.3  # High IL risk for volatile pairs
    
    async def find_yield_farming_opportunities(self, min_apy: float = 5.0) -> List[YieldFarmingOpportunity]:
        """Find high-yield farming opportunities"""
        
        opportunities = []
        
        # Fetch current pools if not cached
        if not self.pools_cache or self._cache_expired('pools'):
            await self.fetch_defi_pools()
        
        for pool in self.pools_cache.values():
            if pool.apy >= min_apy:
                # Calculate risk score
                risk_score = (
                    pool.impermanent_loss_risk * 0.4 +
                    pool.smart_contract_risk * 0.3 +
                    min(pool.apy / 100, 1.0) * 0.3  # Higher APY = higher risk
                )
                
                opportunity = YieldFarmingOpportunity(
                    protocol=pool.protocol,
                    pool_id=pool.pool_address,
                    tokens=[pool.token0, pool.token1] if pool.token1 else [pool.token0],
                    apy=pool.apy,
                    rewards_token=pool.token0,  # Simplified
                    lock_period=0,  # Most pools have no lock
                    minimum_deposit=100,  # Estimated minimum
                    risk_score=risk_score
                )
                
                opportunities.append(opportunity)
        
        # Sort by risk-adjusted yield
        opportunities.sort(key=lambda x: x.apy / (1 + x.risk_score), reverse=True)
        
        return opportunities
    
    async def detect_arbitrage_opportunities(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities between DEX and CEX"""
        
        arbitrage_ops = []
        
        if not self.exchanges:
            self.logger.warning("No CEX exchanges configured for arbitrage detection")
            return arbitrage_ops
        
        # Get DEX prices
        dex_prices = await self._get_dex_prices(tokens)
        
        # Get CEX prices
        cex_prices = await self._get_cex_prices(tokens)
        
        # Find arbitrage opportunities
        for token in tokens:
            if token in dex_prices and token in cex_prices:
                for dex, dex_price in dex_prices[token].items():
                    for cex, cex_price in cex_prices[token].items():
                        if dex_price and cex_price:
                            # Calculate price difference
                            price_diff = abs(dex_price - cex_price) / min(dex_price, cex_price)
                            
                            if price_diff > 0.005:  # 0.5% minimum arbitrage
                                arbitrage_ops.append({
                                    'token': token,
                                    'buy_venue': dex if dex_price < cex_price else cex,
                                    'sell_venue': cex if dex_price < cex_price else dex,
                                    'buy_price': min(dex_price, cex_price),
                                    'sell_price': max(dex_price, cex_price),
                                    'profit_percentage': price_diff * 100,
                                    'estimated_profit': price_diff * 1000,  # Assuming $1000 trade
                                    'timestamp': datetime.now()
                                })
        
        return sorted(arbitrage_ops, key=lambda x: x['profit_percentage'], reverse=True)
    
    async def _get_dex_prices(self, tokens: List[str]) -> Dict[str, Dict[str, float]]:
        """Get token prices from DEXes"""
        
        prices = {token: {} for token in tokens}
        
        # Use cached pool data to estimate prices
        for pool in self.pools_cache.values():
            if pool.token0 in tokens or pool.token1 in tokens:
                # Simplified price calculation (would need more sophisticated logic)
                if pool.tvl > 0:
                    # Estimate price based on pool reserves (simplified)
                    estimated_price = 100 + np.random.normal(0, 10)  # Mock price
                    
                    if pool.token0 in tokens:
                        prices[pool.token0][pool.protocol] = estimated_price
                    if pool.token1 in tokens:
                        prices[pool.token1][pool.protocol] = estimated_price * 1.1
        
        return prices
    
    async def _get_cex_prices(self, tokens: List[str]) -> Dict[str, Dict[str, float]]:
        """Get token prices from centralized exchanges"""
        
        prices = {token: {} for token in tokens}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                tickers = await asyncio.get_event_loop().run_in_executor(
                    None, exchange.fetch_tickers
                )
                
                for token in tokens:
                    symbol = f"{token}/USDT"
                    if symbol in tickers:
                        prices[token][exchange_name] = tickers[symbol]['last']
                        
            except Exception as e:
                self.logger.error(f"Failed to fetch prices from {exchange_name}: {e}")
        
        return prices
    
    def optimize_defi_allocation(self, 
                                portfolio_value: float,
                                risk_tolerance: float,
                                target_apy: float) -> Dict[str, Any]:
        """Optimize DeFi allocation based on risk and return preferences"""
        
        if not self.yield_opportunities_cache:
            self.logger.warning("No yield opportunities cached. Run find_yield_farming_opportunities first.")
            return {}
        
        # Filter opportunities by risk tolerance
        suitable_opportunities = [
            opp for opp in self.yield_opportunities_cache
            if opp.risk_score <= risk_tolerance and opp.apy >= target_apy
        ]
        
        if not suitable_opportunities:
            return {'error': 'No suitable opportunities found'}
        
        # Simple allocation optimization (equal weight for now)
        num_opportunities = min(5, len(suitable_opportunities))  # Max 5 positions
        allocation_per_position = portfolio_value / num_opportunities
        
        allocations = []
        total_expected_yield = 0
        total_risk_score = 0
        
        for i, opp in enumerate(suitable_opportunities[:num_opportunities]):
            allocation = {
                'protocol': opp.protocol,
                'tokens': opp.tokens,
                'allocation_amount': allocation_per_position,
                'allocation_percentage': 100 / num_opportunities,
                'expected_apy': opp.apy,
                'risk_score': opp.risk_score,
                'expected_annual_yield': allocation_per_position * opp.apy / 100
            }
            
            allocations.append(allocation)
            total_expected_yield += allocation['expected_annual_yield']
            total_risk_score += opp.risk_score
        
        return {
            'allocations': allocations,
            'total_allocation': portfolio_value,
            'expected_annual_yield': total_expected_yield,
            'expected_apy': (total_expected_yield / portfolio_value) * 100,
            'average_risk_score': total_risk_score / num_opportunities,
            'diversification_score': num_opportunities / 5.0,  # Max 5 positions
            'timestamp': datetime.now()
        }
    
    def _cache_expired(self, cache_type: str, expiry_minutes: int = 15) -> bool:
        """Check if cache has expired"""
        
        if cache_type not in self.last_update:
            return True
        
        return datetime.now() - self.last_update[cache_type] > timedelta(minutes=expiry_minutes)
    
    def get_defi_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of DeFi portfolio opportunities"""
        
        summary = {
            'timestamp': datetime.now(),
            'total_pools_tracked': len(self.pools_cache),
            'total_opportunities': len(self.yield_opportunities_cache),
            'protocols': list(set(pool.protocol for pool in self.pools_cache.values())),
            'avg_apy': np.mean([pool.apy for pool in self.pools_cache.values()]) if self.pools_cache else 0,
            'total_tvl': sum(pool.tvl for pool in self.pools_cache.values()),
            'cache_status': {
                'pools_last_update': self.last_update.get('pools'),
                'opportunities_last_update': self.last_update.get('opportunities')
            }
        }
        
        return summary