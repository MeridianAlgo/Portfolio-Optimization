"""
Real-time Portfolio Dashboard

Interactive Streamlit dashboard for real-time portfolio monitoring,
optimization, and analysis with live market data integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import asyncio
import threading

# Dashboard imports with fallbacks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from ..core.optimizer import QuantumFinanceOptimizer
from ..core.config import OptimizationConfig
from ..realtime.streaming_optimizer import StreamingOptimizer
from ..realtime.data_stream_manager import DataStreamManager, StreamConfig


class DashboardApp:
    """
    Real-time portfolio dashboard application
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.optimizer = None
        self.streaming_optimizer = None
        self.data_stream_manager = None
        
        # Dashboard state
        self.last_update = datetime.now()
        self.update_frequency = 5  # seconds
        self.auto_refresh = True
        
        # Performance tracking
        self.performance_history = []
        self.portfolio_history = []
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize optimization and streaming services"""
        try:
            # Core optimizer
            self.optimizer = QuantumFinanceOptimizer(self.config)
            
            # Data stream manager
            self.data_stream_manager = DataStreamManager()
            
            # Streaming optimizer
            self.streaming_optimizer = StreamingOptimizer(
                config=self.config,
                data_stream_manager=self.data_stream_manager,
                update_frequency="1m"
            )
            
            # Setup data streams
            self._setup_data_streams()
            
            self.logger.info("Dashboard services initialized")
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
    
    def _setup_data_streams(self):
        """Setup real-time data streams"""
        try:
            # Add mock data stream for demo
            stream_config = StreamConfig(
                provider="mock",
                symbols=self.config.tickers,
                data_types=["price", "volume"],
                update_frequency="5s"
            )
            
            self.data_stream_manager.add_stream("main_stream", stream_config)
            self.data_stream_manager.start_stream("main_stream")
            
            # Start streaming optimization
            self.streaming_optimizer.start_streaming_optimization()
            
        except Exception as e:
            self.logger.error(f"Data stream setup failed: {e}")
    
    def run_streamlit_dashboard(self, port: int = 8501):
        """Run Streamlit dashboard"""
        
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not available")
        
        st.set_page_config(
            page_title="QuantumFinanceOpt Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<h1 class="main-header">🚀 QuantumFinanceOpt Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
        
        # Auto-refresh
        if self.auto_refresh:
            time.sleep(self.update_frequency)
            st.experimental_rerun()
    
    def _render_sidebar(self):
        """Render dashboard sidebar"""
        
        st.sidebar.header("⚙️ Controls")
        
        # Auto-refresh toggle
        self.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=self.auto_refresh)
        
        # Update frequency
        self.update_frequency = st.sidebar.slider(
            "Update Frequency (seconds)", 
            min_value=1, 
            max_value=60, 
            value=self.update_frequency
        )
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.experimental_rerun()
        
        # Portfolio controls
        st.sidebar.header("📊 Portfolio")
        
        # Force rebalance
        if st.sidebar.button("⚡ Force Rebalance"):
            if self.streaming_optimizer:
                self.streaming_optimizer.force_rebalance("manual_dashboard")
                st.sidebar.success("Rebalancing triggered!")
        
        # Optimization method selection
        optimization_methods = st.sidebar.multiselect(
            "Optimization Methods",
            ["classical", "quantum", "ml", "ensemble"],
            default=["classical", "quantum"]
        )
        
        # Risk parameters
        st.sidebar.header("⚠️ Risk Settings")
        
        risk_aversion = st.sidebar.slider(
            "Risk Aversion", 
            min_value=0.1, 
            max_value=5.0, 
            value=1.0, 
            step=0.1
        )
        
        esg_threshold = st.sidebar.slider(
            "ESG Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1
        )
        
        # System status
        st.sidebar.header("🔧 System Status")
        self._render_system_status()
    
    def _render_system_status(self):
        """Render system status in sidebar"""
        
        try:
            # Data stream status
            if self.data_stream_manager:
                stream_status = self.data_stream_manager.get_stream_status()
                for stream_id, status in stream_status.items():
                    if status['active']:
                        st.sidebar.markdown(f"🟢 {stream_id}: Active")
                    else:
                        st.sidebar.markdown(f"🔴 {stream_id}: Inactive")
            
            # Optimization status
            if self.streaming_optimizer:
                perf_summary = self.streaming_optimizer.get_performance_summary()
                success_rate = perf_summary.get('success_rate', 0)
                
                if success_rate > 0.9:
                    status_color = "🟢"
                elif success_rate > 0.7:
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                
                st.sidebar.markdown(f"{status_color} Optimization: {success_rate:.1%}")
            
            # Last update time
            st.sidebar.markdown(f"🕒 Last Update: {self.last_update.strftime('%H:%M:%S')}")
            
        except Exception as e:
            st.sidebar.error(f"Status error: {e}")
    
    def _render_main_content(self):
        """Render main dashboard content"""
        
        # Key metrics row
        self._render_key_metrics()
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_portfolio_allocation()
            self._render_performance_chart()
        
        with col2:
            self._render_risk_metrics()
            self._render_correlation_heatmap()
        
        # Additional sections
        self._render_optimization_history()
        self._render_market_data()
    
    def _render_key_metrics(self):
        """Render key portfolio metrics"""
        
        st.header("📊 Key Metrics")
        
        try:
            # Get current portfolio data
            if self.streaming_optimizer:
                portfolio_data = self.streaming_optimizer.get_current_portfolio()
                perf_summary = self.streaming_optimizer.get_performance_summary()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    total_value = portfolio_data.get('total_value', 0)
                    st.metric(
                        label="Portfolio Value",
                        value=f"${total_value:,.0f}",
                        delta=f"{np.random.uniform(-2, 2):.1f}%"  # Mock delta
                    )
                
                with col2:
                    sharpe_ratio = perf_summary.get('avg_sharpe_ratio', 0)
                    st.metric(
                        label="Sharpe Ratio",
                        value=f"{sharpe_ratio:.3f}",
                        delta=f"{np.random.uniform(-0.1, 0.1):.3f}"
                    )
                
                with col3:
                    success_rate = perf_summary.get('success_rate', 0)
                    st.metric(
                        label="Success Rate",
                        value=f"{success_rate:.1%}",
                        delta=f"{np.random.uniform(-5, 5):.1f}%"
                    )
                
                with col4:
                    avg_turnover = perf_summary.get('avg_turnover', 0)
                    st.metric(
                        label="Avg Turnover",
                        value=f"{avg_turnover:.2%}",
                        delta=f"{np.random.uniform(-1, 1):.2f}%"
                    )
                
                with col5:
                    total_costs = perf_summary.get('total_transaction_costs', 0)
                    st.metric(
                        label="Transaction Costs",
                        value=f"${total_costs:.0f}",
                        delta=f"${np.random.uniform(-50, 50):.0f}"
                    )
            
            else:
                st.warning("Streaming optimizer not available")
                
        except Exception as e:
            st.error(f"Error rendering metrics: {e}")
    
    def _render_portfolio_allocation(self):
        """Render portfolio allocation chart"""
        
        st.subheader("🥧 Portfolio Allocation")
        
        try:
            if self.streaming_optimizer:
                portfolio_data = self.streaming_optimizer.get_current_portfolio()
                positions = portfolio_data.get('positions', {})
                
                if positions:
                    # Create pie chart
                    symbols = list(positions.keys())
                    weights = [pos['weight'] for pos in positions.values()]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=symbols,
                        values=weights,
                        hole=0.3,
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    
                    fig.update_layout(
                        title="Current Portfolio Weights",
                        showlegend=True,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No portfolio data available")
            else:
                # Mock data for demo
                symbols = self.config.tickers
                weights = np.random.dirichlet(np.ones(len(symbols)))
                
                fig = go.Figure(data=[go.Pie(
                    labels=symbols,
                    values=weights,
                    hole=0.3
                )])
                
                fig.update_layout(title="Portfolio Allocation", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering allocation: {e}")
    
    def _render_performance_chart(self):
        """Render portfolio performance chart"""
        
        st.subheader("📈 Performance")
        
        try:
            # Generate mock performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            
            # Simulate portfolio performance
            returns = np.random.normal(0.001, 0.02, len(dates))
            cumulative_returns = np.cumprod(1 + returns)
            
            # Create benchmark
            benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))
            benchmark_cumulative = np.cumprod(1 + benchmark_returns)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_cumulative,
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Cumulative Returns (30 Days)",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering performance: {e}")    
   
 def _render_risk_metrics(self):
        """Render risk metrics display"""
        
        st.subheader("⚠️ Risk Metrics")
        
        try:
            # Mock risk metrics
            risk_metrics = {
                'VaR (95%)': np.random.uniform(-0.05, -0.02),
                'VaR (99%)': np.random.uniform(-0.08, -0.05),
                'Max Drawdown': np.random.uniform(-0.15, -0.08),
                'Volatility': np.random.uniform(0.12, 0.25),
                'Beta': np.random.uniform(0.8, 1.2),
                'Correlation': np.random.uniform(0.6, 0.9)
            }
            
            # Create gauge charts
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=list(risk_metrics.keys()),
                specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
            )
            
            positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
            
            for i, (metric, value) in enumerate(risk_metrics.items()):
                row, col = positions[i]
                
                # Determine color based on metric type
                if 'VaR' in metric or 'Drawdown' in metric:
                    color = "red" if abs(value) > 0.05 else "orange" if abs(value) > 0.03 else "green"
                else:
                    color = "green" if value < 0.2 else "orange" if value < 0.3 else "red"
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=abs(value) if 'VaR' in metric or 'Drawdown' in metric else value,
                        title={'text': metric},
                        gauge={'axis': {'range': [None, 1]},
                               'bar': {'color': color},
                               'steps': [{'range': [0, 0.5], 'color': "lightgray"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 0.8}}
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering risk metrics: {e}")
    
    def _render_correlation_heatmap(self):
        """Render asset correlation heatmap"""
        
        st.subheader("🔥 Asset Correlations")
        
        try:
            # Generate mock correlation matrix
            n_assets = len(self.config.tickers)
            correlation_matrix = np.random.uniform(0.3, 0.9, (n_assets, n_assets))
            
            # Make symmetric and set diagonal to 1
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=self.config.tickers,
                y=self.config.tickers,
                colorscale='RdYlBu_r',
                zmid=0,
                text=correlation_matrix,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Asset Correlation Matrix",
                height=400,
                xaxis_title="Assets",
                yaxis_title="Assets"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering correlation heatmap: {e}")
    
    def _render_optimization_history(self):
        """Render optimization history table"""
        
        st.header("📋 Optimization History")
        
        try:
            if self.streaming_optimizer:
                history = self.streaming_optimizer.get_optimization_history(limit=20)
                
                if history:
                    # Convert to DataFrame
                    df = pd.DataFrame(history)
                    
                    # Select relevant columns
                    display_columns = [
                        'timestamp', 'optimization_method', 'sharpe_ratio', 
                        'expected_return', 'volatility', 'turnover', 'success'
                    ]
                    
                    df_display = df[display_columns].copy()
                    
                    # Format columns
                    df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%H:%M:%S')
                    df_display['sharpe_ratio'] = df_display['sharpe_ratio'].round(3)
                    df_display['expected_return'] = (df_display['expected_return'] * 100).round(2)
                    df_display['volatility'] = (df_display['volatility'] * 100).round(2)
                    df_display['turnover'] = (df_display['turnover'] * 100).round(2)
                    
                    # Rename columns
                    df_display.columns = [
                        'Time', 'Method', 'Sharpe', 'Return (%)', 'Vol (%)', 'Turnover (%)', 'Success'
                    ]
                    
                    # Style the dataframe
                    def style_success(val):
                        return 'color: green' if val else 'color: red'
                    
                    styled_df = df_display.style.applymap(style_success, subset=['Success'])
                    
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.info("No optimization history available")
            else:
                st.warning("Streaming optimizer not available")
                
        except Exception as e:
            st.error(f"Error rendering optimization history: {e}")
    
    def _render_market_data(self):
        """Render live market data"""
        
        st.header("📊 Live Market Data")
        
        try:
            if self.data_stream_manager:
                # Get latest data for each symbol
                market_data = []
                
                for symbol in self.config.tickers:
                    latest_data = self.data_stream_manager.get_latest_data(symbol, limit=1)
                    
                    if latest_data:
                        data_point = latest_data[0]
                        market_data.append({
                            'Symbol': symbol,
                            'Price': f"${data_point.get('price', 0):.2f}",
                            'Volume': f"{data_point.get('volume', 0):,}",
                            'Source': data_point.get('source', 'unknown'),
                            'Quality': f"{data_point.get('quality_score', 1.0):.2f}",
                            'Timestamp': data_point.get('timestamp', 'N/A')
                        })
                
                if market_data:
                    df = pd.DataFrame(market_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No market data available")
            else:
                # Mock market data
                mock_data = []
                for symbol in self.config.tickers:
                    mock_data.append({
                        'Symbol': symbol,
                        'Price': f"${np.random.uniform(50, 200):.2f}",
                        'Volume': f"{np.random.randint(1000, 100000):,}",
                        'Change': f"{np.random.uniform(-5, 5):.2f}%",
                        'Last Update': datetime.now().strftime('%H:%M:%S')
                    })
                
                df = pd.DataFrame(mock_data)
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering market data: {e}")
    
    def run_dash_dashboard(self, port: int = 8050):
        """Run Dash dashboard (alternative to Streamlit)"""
        
        if not DASH_AVAILABLE:
            raise ImportError("Dash not available")
        
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("QuantumFinanceOpt Dashboard", className="text-center mb-4"),
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Portfolio Value", className="card-title"),
                            html.H2(id="portfolio-value", children="$0", className="text-primary")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sharpe Ratio", className="card-title"),
                            html.H2(id="sharpe-ratio", children="0.000", className="text-success")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Success Rate", className="card-title"),
                            html.H2(id="success-rate", children="0%", className="text-info")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Turnover", className="card-title"),
                            html.H2(id="turnover", children="0%", className="text-warning")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="portfolio-allocation")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="performance-chart")
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="risk-metrics")
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="correlation-heatmap")
                ], width=6)
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ], fluid=True)
        
        # Callbacks for updating charts
        @app.callback(
            [Output('portfolio-value', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('success-rate', 'children'),
             Output('turnover', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            # Mock data updates
            portfolio_value = f"${np.random.uniform(90000, 110000):,.0f}"
            sharpe_ratio = f"{np.random.uniform(0.5, 2.0):.3f}"
            success_rate = f"{np.random.uniform(70, 95):.0f}%"
            turnover = f"{np.random.uniform(1, 10):.1f}%"
            
            return portfolio_value, sharpe_ratio, success_rate, turnover
        
        @app.callback(
            Output('portfolio-allocation', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_allocation(n):
            symbols = self.config.tickers
            weights = np.random.dirichlet(np.ones(len(symbols)))
            
            fig = go.Figure(data=[go.Pie(
                labels=symbols,
                values=weights,
                hole=0.3
            )])
            
            fig.update_layout(title="Portfolio Allocation")
            return fig
        
        # Run the app
        app.run_server(debug=False, host='0.0.0.0', port=port)
    
    def cleanup(self):
        """Cleanup dashboard resources"""
        
        try:
            if self.streaming_optimizer:
                self.streaming_optimizer.stop_streaming_optimization()
            
            if self.data_stream_manager:
                self.data_stream_manager.stop_all_streams()
            
            self.logger.info("Dashboard cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Dashboard cleanup failed: {e}")


def main():
    """Main function to run dashboard"""
    
    # Default configuration
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        budget=100000.0,
        optimization_methods=['classical', 'quantum']
    )
    
    # Create and run dashboard
    dashboard = DashboardApp(config)
    
    try:
        if STREAMLIT_AVAILABLE:
            dashboard.run_streamlit_dashboard()
        elif DASH_AVAILABLE:
            dashboard.run_dash_dashboard()
        else:
            print("No dashboard framework available. Install streamlit or dash.")
    
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        dashboard.cleanup()
    
    except Exception as e:
        print(f"Dashboard error: {e}")
        dashboard.cleanup()


if __name__ == "__main__":
    main()