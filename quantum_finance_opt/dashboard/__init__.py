"""
Real-time Dashboard Module

Provides interactive web dashboard for portfolio monitoring and optimization
with real-time updates and advanced visualizations.
"""

from .dashboard_app import DashboardApp
from .components import PortfolioComponents
from .real_time_updater import RealTimeUpdater

__all__ = [
    'DashboardApp',
    'PortfolioComponents', 
    'RealTimeUpdater'
]