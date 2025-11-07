import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from data_loader import DataLoader
from models import MarkovRegimeModel, MonteCarloModel, CorrelationAnalyzer, LotkaVolterraModel
from llm_analyzer import FinancialAnalyzer
from utils import VisualizationUtils

# Configure page with modern styling
st.set_page_config(
    page_title="Probabilistic Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --dark-bg: #0e1117;
        --card-bg: #262730;
    }
    
    /* Hide default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
    }
    
    /* Modern card styling */
    .metric-card {
        background: linear-gradient(145deg, #262730 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #3d4147;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 0.5rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 119, 180, 0.2);
    }
    
    /* Animated gradient headers */
    .gradient-header {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(38, 39, 48, 0.8);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #262730 0%, #1e1e2e 100%);
        border: 1px solid #3d4147;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Success/Error/Warning boxes */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 15px;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading spinner custom */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Sidebar enhancements */
    .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

class ProbabilisticMarketApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.markov_model = MarkovRegimeModel()
        self.mc_model = MonteCarloModel()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.lv_model = LotkaVolterraModel()
        self.llm_analyzer = FinancialAnalyzer()
        self.viz_utils = VisualizationUtils()
        
        # Initialize session state
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'regime_data' not in st.session_state:
            st.session_state.regime_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'lv_results' not in st.session_state:
            st.session_state.lv_results = {}
    
    def create_animated_header(self):
        """Create animated header with modern styling"""
        st.markdown("""
        <div class="gradient-header">
             Probabilistic Market Ecosystem Analyzer
        </div>
        """, unsafe_allow_html=True)
        
        # Add subtitle with typing effect
        st.markdown("""
        <div style="text-align: center; margin: -1rem 0 2rem 0; opacity: 0.8;">
            <span style="font-size: 1.2rem; font-style: italic;">
                Advanced Financial Analysis • Real-time Insights • Predictive Modeling
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    def create_enhanced_sidebar(self):
        """Enhanced sidebar with modern design"""
        with st.sidebar:
            # Header
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: white; margin: 0;"> Control Center</h2>
                <p style="opacity: 0.8; margin: 0.5rem 0;">Configure Your Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Analysis Mode Selection with enhanced styling
            st.markdown("###  Analysis Mode")
            analysis_type = st.selectbox(
                "",
                ["Quick Analysis", "⚙️ Advanced Configuration"],
                label_visibility="collapsed"
            )
            
            st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
            
            # Stock selection with modern multiselect
            st.markdown("### Select Assets")
            available_stocks = ['AAPL', 'MSFT', 'GOOGL', '^GSPC']
            stock_options = {
                'AAPL': 'Apple Inc.',
                'MSFT': 'Microsoft Corp.',
                'GOOGL': 'Alphabet Inc.',
                '^GSPC': 'S&P500'
            }
            
            selected_stocks = []
            cols = st.columns(2)
            for i, (symbol, name) in enumerate(stock_options.items()):
                col = cols[i % 2]
                if col.checkbox(name, value=True, key=f"stock_{symbol}"):
                    selected_stocks.append(symbol)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Date Range Selection
            st.markdown("### 📅 Time Period")
            col1, col2 = st.columns(2)
            
            min_date = pd.to_datetime("2015-01-01")
            max_date = pd.to_datetime("2024-01-01")
            
            with col1:
                start_date = st.date_input(
                    "Start", 
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
            with col2:
                end_date = st.date_input(
                    "End", 
                    value=max_date,
                    min_value=start_date,
                    max_value=max_date
                )
            
            # Analysis Parameters
            st.markdown("###  Analysis Parameters")
            
            if "Quick" in analysis_type:
                mc_simulations = 500
                forecast_days = 60
                st.info("Quick mode: Optimized for speed")
            else:
                mc_simulations = st.slider(" Monte Carlo Simulations", 100, 5000, 1000, step=100)
                forecast_days = st.slider("Forecast Horizon (Days)", 30, 365, 90, step=15)
            
            # Ecology Analysis
            st.markdown("###  Market Ecology")
            include_lv = st.toggle("Enable Ecosystem Analysis", value=True)
            
            if include_lv:
                lv_lookback = st.slider("Historical Window (Days)", 30, 180, 60, step=10)
            else:
                lv_lookback = 60
            
            # Analysis Summary
            st.markdown("---")
            st.markdown("### 📋 Current Configuration")
            
            summary_data = {
                "Assets": len(selected_stocks),
                "Period": f"{(end_date - start_date).days} days",
                "Simulations": f"{mc_simulations:,}",
                "Forecast": f"{forecast_days} days",
                "Ecology": "✅" if include_lv else "❌"
            }
            
            for key, value in summary_data.items():
                st.markdown(f"{key}:{value}")
            
            # Action Button with enhanced styling
            st.markdown("---")
            run_analysis = st.button(
                "Launch Analysis", 
                type="primary", 
                use_container_width=True,
                help="Run comprehensive market analysis"
            )
            
            return (run_analysis, selected_stocks, start_date, end_date, 
                   mc_simulations, forecast_days, include_lv, lv_lookback)
    
    def create_status_dashboard(self, selected_stocks):
        """Create a real-time status dashboard"""
        if st.session_state.df is not None:
            latest_data = st.session_state.df.iloc[-1]
            
            st.markdown("### Market Status Board")
            
# Create status cards
        cols = st.columns(len(selected_stocks))

        for i, stock in enumerate(selected_stocks):
            if stock in st.session_state.df.columns:
                with cols[i]:
                    current_price = float(latest_data[stock])
                    prev_price = float(st.session_state.df[stock].iloc[-2]) if len(st.session_state.df) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100

            # Determine status color and arrow
                    if change_pct > 0.75:
                            status = "🟢"
                            trend_color = "#2ca02c"
                            arrow = "⬆️"
                    elif change_pct < -0.75:
                            status = "🔴"
                            trend_color = "#d62728"
                            arrow = "⬇️"
                    else:
                            status = "🟡"
                            trend_color = "#ff7f0e"
                            arrow = "➡️"
            
            # Enhanced metric card with arrow
                    st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{status}</div>
                        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">{stock}</div>
                        <div style="font-size: 1.3rem; font-weight: bold; color: white;">${current_price:.2f}</div>
                        <div style="color: {trend_color}; font-weight: 500;">
                            {arrow} {change_pct:+.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    
    def create_enhanced_tabs(self, selected_stocks):
        """Create enhanced tab interface"""
        # Create tabs with modern styling
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Live Dashboard", 
            "Regime Analysis", 
            " Monte Carlo", 
            " Ecosystem Analysis", 
            "Insights"
        ])
        
        with tab1:
            self.display_enhanced_live_dashboard(selected_stocks)
            
        with tab2:
            self.display_enhanced_regime_analysis(selected_stocks)
            
        with tab3:
            self.display_enhanced_monte_carlo(selected_stocks)
            
        with tab4:
            self.display_enhanced_lv_analysis(selected_stocks)
            
        with tab5:
            self.display_enhanced_ai_insights_wrapper(selected_stocks)
    
    def display_enhanced_live_dashboard(self, selected_stocks):
        """Enhanced live dashboard with modern visuals"""
        st.markdown("## Real-Time Market Dashboard")
        
        if st.session_state.df is None:
            st.info("Configure settings in the sidebar and launch analysis to see live data")
            return
        
        # Status cards at top
        self.create_status_dashboard(selected_stocks)

        st.markdown("### Price Evolution")
        price_chart = self.viz_utils.create_price_plot(st.session_state.df, selected_stocks)
            
            # Enhance chart styling
        price_chart.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=20, color='white'),
            legend=dict(
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1
            )            )
        st.plotly_chart(price_chart, use_container_width=True)
        
        st.markdown("### 🔗 Correlations")
        correlation_chart = self.viz_utils.create_correlation_heatmap(st.session_state.df, selected_stocks)
        correlation_chart.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10)
            )
        st.plotly_chart(correlation_chart, use_container_width=True)
        
    
    def display_enhanced_regime_analysis(self, selected_stocks):
        """Enhanced regime analysis with modern visuals"""
        st.markdown("## Market Regime Analysis")
        
        if st.session_state.regime_data is None:
            st.info("Run analysis to see regime data")
            return
        
        # Overview section
        st.markdown("###  Regime Overview")
        
        # Create regime summary
        regime_cols = st.columns(len(selected_stocks))
        
        for i, stock in enumerate(selected_stocks):
            if stock in st.session_state.regime_data.columns:
                with regime_cols[i]:
                    current_regime = st.session_state.regime_data[stock].iloc[-1]
                    duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
                    
                    # Regime-specific styling
                    if current_regime == "Bull":
                        regime_color = "#2ca02c"
                        regime_icon = "🐂"
                    elif current_regime == "Bear":
                        regime_color = "#d62728"
                        regime_icon = "🐻"
                    else:
                        regime_color = "#ff7f0e"
                        regime_icon = "⚖️"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; border-left: 4px solid {regime_color};">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{regime_icon}</div>
                        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">{stock}</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: {regime_color};">{current_regime}</div>
                        <div style="opacity: 0.8; font-size: 0.9rem;">{duration} days</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed analysis for each stock
        st.markdown("### Detailed Regime Analysis")
        
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                with st.expander(f"{stock} Regime Details", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        timeline = self.viz_utils.create_regime_timeline(st.session_state.regime_data, stock)
                        timeline.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(timeline, use_container_width=True)
                    
                    with col2:
    # Current regime
                        current_regime = st.session_state.regime_data[stock].iloc[-1]
    
    # Calculate transition matrix
                        transition_matrix = self.calculate_stock_transition_matrix(st.session_state.regime_data[stock])
    
    # Ensure the matrix has all regimes in the correct order
                        regimes = ['Bull', 'Bear', 'Stable']
                        transition_matrix = transition_matrix.reindex(index=regimes, columns=regimes, fill_value=0)
    
                        st.markdown("### Transition Probability Matrix")
                        st.dataframe(transition_matrix.style.format("{:.1%}"))

    
    def display_enhanced_monte_carlo(self, selected_stocks):
        """Enhanced Monte Carlo analysis"""
        st.markdown("##  Monte Carlo Simulation")
        
        if 'monte_carlo' not in st.session_state.analysis_results:
            st.info("Run analysis to see Monte Carlo results")
            return
        
        mc_results = st.session_state.analysis_results['monte_carlo']
        
        for stock in selected_stocks:
            if stock in mc_results:
                with st.expander(f"{stock} - Price Simulation", expanded=True):
                    results = mc_results[stock]
                    
                    # Validate results
                    current_price = results['current_price']
                    if results['percentile_95'] > current_price * 10:
                        st.warning("Simulation results appear unrealistic")
                        continue
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Enhanced Monte Carlo plot
                        mc_plot = self.viz_utils.create_monte_carlo_plot(results, stock)
                        mc_plot.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            title_font=dict(size=18, color='white')
                        )
                        st.plotly_chart(mc_plot, use_container_width=True)
                    
                    with col2:
                        st.markdown("Risk Metrics")
                        
                        # Risk metrics cards
                        final_prices = results['paths'][-1, :]
                        returns = (final_prices / current_price) - 1
                        var_5 = np.percentile(returns, 5)
                        cvar_5 = returns[returns <= var_5].mean()
                        
                        metrics_data = [
                            ("Current Price", f"${current_price:.2f}", "💰"),
                            ("VaR (95%)", f"{var_5:.1%}", "⚠️"),
                            ("CVaR (95%)", f"{cvar_5:.1%}", "🚨"),
                            ("Expected Return", f"{results['mu']:.1%}", "📈"),
                            ("Volatility", f"{results['sigma']:.1%}", "📊")
                        ]
                        
                        for label, value, icon in metrics_data:
                            st.markdown(f"""
                            <div style="padding: 0.5rem; margin: 0.2rem 0; background: rgba(255,255,255,0.05); border-radius: 8px;">
                                {icon} {label}:{value}
                            </div>
                            """, unsafe_allow_html=True)
    
    def display_enhanced_lv_analysis(self, selected_stocks):
        """Enhanced Lotka-Volterra analysis with modern visuals"""
        st.markdown("##  Market Ecosystem Analysis")
        
        if not st.session_state.lv_results:
            st.info("Enable ecosystem analysis and run to see market ecology insights")
            return
        
        # Ecosystem overview
        st.markdown("### Ecosystem Health Dashboard")
        
        ecosystem_cols = st.columns(len(selected_stocks))
        
        for i, stock in enumerate(selected_stocks):
            if stock in st.session_state.lv_results:
                with ecosystem_cols[i]:
                    lv_data = st.session_state.lv_results[stock]
                    prey = lv_data['current_prey']
                    predator = lv_data['current_predator']
                    dominance = prey / (predator + 1e-8)
                    
                    # Determine ecosystem status
                    if dominance > 1.2:
                        status = "🟢 Thriving"
                        status_color = "#2ca02c"
                    elif dominance < 0.8:
                        status = "🔴 Stressed"
                        status_color = "#d62728"
                    else:
                        status = "🟡 Balanced"
                        status_color = "#ff7f0e"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; border-left: 4px solid {status_color};">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;"></div>
                        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">{stock}</div>
                        <div style="font-size: 1rem; font-weight: bold; color: {status_color};">{status}</div>
                        <div style="opacity: 0.8; font-size: 0.8rem;">Ratio: {dominance:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed ecosystem analysis
        st.markdown("### Detailed Ecosystem Analysis")
        
        for stock in selected_stocks:
            if stock in st.session_state.lv_results:
                with st.expander(f" {stock} - Ecosystem Dynamics", expanded=True):
                    self.display_enhanced_stock_lv_analysis(stock)
    
    def display_enhanced_stock_lv_analysis(self, stock):
        """Enhanced individual stock LV analysis"""
        lv_data = st.session_state.lv_results[stock]
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Enhanced ecology plot
            fig = self.viz_utils.create_ecology_plot(lv_data, stock)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16, color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("Current State")
            
            # Ecosystem metrics
            metrics = [
                ("🐰 Buying Pressure", f"{lv_data['current_prey']:.3f}"),
                ("🦅 Selling Pressure", f"{lv_data['current_predator']:.3f}"),
                ("📈 Buy Trend", f"{lv_data['prey_trend']:+.4f}"),
                ("📉 Sell Trend", f"{lv_data['predator_trend']:+.4f}")
            ]
            
            for label, value in metrics:
                st.markdown(f"{label}:{value}")
        
        
        # Phase space analysis
        col4, col5 = st.columns(2)
        
        with col4:
            phase_fig = self.viz_utils.create_phase_plot(lv_data, stock)
            phase_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=14)
            )
            st.plotly_chart(phase_fig, use_container_width=True)
        
        with col5:
        
            # Model parameters
            st.markdown("⚙️ Model Parameters:")
            params = [
                ("α (Growth)", f"{lv_data['alpha']:.4f}"),
                ("β (Predation)", f"{lv_data['beta']:.4f}"),
                ("δ (Efficiency)", f"{lv_data['delta']:.4f}"),
                ("γ (Decline)", f"{lv_data['gamma']:.4f}")
            ]
            
            for param, value in params:
                st.text(f"{param}: {value}")
    
    

    def _display_stock_quantitative_analysis(self, stock):
        """Display quantitative analysis for individual stock"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Regime Analysis")
            current_regime = st.session_state.regime_data[stock].iloc[-1]
            regime_duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
            
            # Regime statistics
            regime_counts = st.session_state.regime_data[stock].value_counts()
            total_periods = len(st.session_state.regime_data)
            
            st.metric("Current Regime", f"{current_regime} ({regime_duration} days)")
            
            for regime, count in regime_counts.items():
                percentage = (count / total_periods) * 100
                st.progress(percentage/100, text=f"{regime}: {percentage:.1f}%")
        
        with col2:
            st.subheader("📊 Risk Metrics")
            
            if stock in st.session_state.df.columns and f"{stock}_log_return" in st.session_state.df.columns:
                returns = st.session_state.df[f"{stock}_log_return"].dropna()
                
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    max_drawdown = (returns.cumsum().expanding().max() - returns.cumsum()).max()
                    
                    st.metric("Annual Volatility", f"{volatility:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        
        # Monte Carlo insights if available
        if 'monte_carlo' in st.session_state.analysis_results and stock in st.session_state.analysis_results['monte_carlo']:
            st.subheader(" Monte Carlo Projections")
            mc_results = st.session_state.analysis_results['monte_carlo'][stock]
            
            final_prices = mc_results['paths'][-1, :]
            expected_price = np.mean(final_prices)
            current_price = mc_results['current_price']
            expected_return = (expected_price / current_price - 1) * 100
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Expected Price", f"${expected_price:.2f}")
            with col4:
                st.metric("Projected Return", f"{expected_return:.1f}%")

    # Core logic methods (unchanged from original)
    def run_complete_analysis(self, selected_stocks, start_date, end_date, 
                            mc_simulations, forecast_days, include_lv, lv_lookback):
        """Run the complete analysis pipeline"""
        with st.spinner("Loading data and running analysis..."):
            # Load data
            st.session_state.df = self.data_loader.download_data(
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d")
            )
            st.session_state.df = self.data_loader.calculate_technical_indicators(
                st.session_state.df
            )
            
            # Calculate regimes
            regime_data = pd.DataFrame()
            for stock in selected_stocks:
                if f"{stock}_log_return" in st.session_state.df.columns:
                    returns = st.session_state.df[f"{stock}_log_return"]
                    regime_data[stock] = self.markov_model.classify_regime(returns)
                    self.markov_model.calculate_transition_matrix(regime_data[stock])
            
            st.session_state.regime_data = regime_data
            
            # Run Monte Carlo simulations
            st.session_state.analysis_results['monte_carlo'] = {}
            for stock in selected_stocks:
                if stock in st.session_state.df.columns:
                    prices = st.session_state.df[stock].dropna()
                    if len(prices) > 30:
                        self.mc_model.n_simulations = mc_simulations
                        st.session_state.analysis_results['monte_carlo'][stock] = (
                            self.mc_model.simulate_stock_paths(prices, forecast_days)
                        )
            
            # Run Lotka-Volterra Analysis
            if include_lv and len(selected_stocks) > 0:
                with st.spinner("Analyzing market ecology..."):
                    st.session_state.lv_results = self.lv_model.analyze_multiple_stocks(
                        st.session_state.df, selected_stocks, lookback_days=lv_lookback
                    )
    
    def calculate_regime_duration(self, regime_series):
        """Calculate how long current regime has lasted"""
        current_regime = regime_series.iloc[-1]
        duration = 0
        for i in range(len(regime_series)-1, -1, -1):
            if regime_series.iloc[i] == current_regime:
                duration += 1
            else:
                break
        return duration

    def calculate_stock_transition_matrix(self, stock_regime_series):
        """Calculate transition matrix for individual stock"""
        states = ['Bull', 'Bear', 'Stable']
        transition_matrix = pd.DataFrame(0, index=states, columns=states)
    
        for i in range(len(stock_regime_series)-1):
            from_state = stock_regime_series.iloc[i]
            to_state = stock_regime_series.iloc[i+1]
            if from_state in states and to_state in states:
                transition_matrix.loc[from_state, to_state] += 1
    
        # Normalize
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
        return transition_matrix.fillna(0)
    

    def display_enhanced_ai_insights(self, selected_stocks):
        """Simplified AI insights - just show meaningful analysis"""
        st.markdown("## AI Market Insights")
        
        if st.button("Generate Insights", type="primary"):
            with st.spinner("Analyzing market conditions..."):
                try:
                    # Prepare data
                    market_data = {}
                    mathematical_results = {
                        'monte_carlo': st.session_state.analysis_results.get('monte_carlo', {}),
                        'regime_data': st.session_state.regime_data
                    }
                    
                    latest_data = st.session_state.df.iloc[-1]
                    for stock in selected_stocks:
                        if stock in latest_data and stock in st.session_state.regime_data.columns:
                            market_data[stock] = {
                                'price': float(latest_data[stock]),
                                'regime': st.session_state.regime_data[stock].iloc[-1]
                            }
                    
                    # Get AI analysis
                    insights = self.llm_analyzer.generate_structured_insights(market_data, mathematical_results)
                    
                    # Display simple, clean insights
                    st.markdown("### Key Findings")
                    st.write(insights)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    # Fallback to basic analysis
                    self._display_simple_fallback(selected_stocks)
        else:
            st.info("Click the button to generate AI insights")

    def _display_simple_fallback(self, selected_stocks):
        """Simple fallback analysis"""
        st.write("### Market Summary")
        
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                regime = st.session_state.regime_data[stock].iloc[-1]
                duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
                
                st.write(f"**{stock}**: {regime} regime ({duration} days)")
                
                if regime == "Bull":
                    st.write("  - Positive momentum detected")
                elif regime == "Bear":
                    st.write("  - Caution advised")
                else:
                    st.write("  - Market consolidating")

    def _display_combined_insights(self, insights_text, selected_stocks):
        """Display both raw AI output and manual interpretation side by side"""
        
        # Create tabs for different views
        ai_tab, manual_tab = st.tabs([
            "AI Analysis", 
            "Manual Interpretation", 
        ])
        
        with ai_tab:
            self._display_raw_ai_analysis(insights_text)
        
        with manual_tab:
            self._display_manual_interpretation(selected_stocks)
        
    def _display_raw_ai_analysis(self, insights_text):
        """Display the raw AI output in a structured format"""
        st.header("AI-Generated Analysis")
        
        # Check if this is a proper AI response or fallback
        is_ai_response = not ("AI ANALYSIS UNAVAILABLE" in insights_text or "fallback" in insights_text.lower())
        
        if is_ai_response:
            st.success("✅ AI Analysis Generated Successfully")
            
            # Display the raw AI output in a scrollable box
            with st.expander("📋 Raw AI Output", expanded=True):
                st.text_area(
                    "AI Response",
                    insights_text,
                    height=300,
                    key="raw_ai_output",
                    label_visibility="collapsed"
                )
            
            # Try to parse and format the AI response
            try:
                self._parse_and_display_ai_response(insights_text)
            except Exception as e:
                st.warning("AI response format unexpected. Showing raw analysis.")
                st.markdown(f"**AI Analysis:**\n\n{insights_text}")
        else:
            st.warning("⚠️ AI response not available in expected format")
            st.info("Showing raw output from analysis engine:")
            st.text_area(
                "Analysis Output",
                insights_text,
                height=200,
                key="fallback_output",
                label_visibility="collapsed"
            )

    def _display_manual_interpretation(self, selected_stocks):
        """Display manual interpretation based on mathematical models"""
        st.header("Manual Interpretation")
        st.info("Based on quantitative analysis of mathematical models")
        
        # Overall market assessment
        st.subheader("📈 Overall Market Assessment")
        
        if st.session_state.regime_data is not None:
            bull_count = 0
            bear_count = 0
            stable_count = 0
            
            for stock in selected_stocks:
                if stock in st.session_state.regime_data.columns:
                    regime = st.session_state.regime_data[stock].iloc[-1]
                    if regime == "Bull":
                        bull_count += 1
                    elif regime == "Bear":
                        bear_count += 1
                    else:
                        stable_count += 1
            
            total = len(selected_stocks)
            st.write(f"**Market Sentiment:** {bull_count}/{total} Bullish, {bear_count}/{total} Bearish, {stable_count}/{total} Stable")
            
            if bull_count > bear_count and bull_count > stable_count:
                st.success("**Overall Bias:** 🟢 Bullish")
            elif bear_count > bull_count and bear_count > stable_count:
                st.error("**Overall Bias:** 🔴 Bearish")
            else:
                st.warning("**Overall Bias:** 🟡 Neutral")
        
        # Individual stock analysis
        st.subheader("📊 Stock-by-Stock Analysis")
        
        for stock in selected_stocks:
            with st.expander(f"{stock} - Technical Analysis", expanded=True):
                self._display_stock_technical_analysis(stock)

    def _display_integrated_view(self, insights_text, selected_stocks):
        """Display integrated view combining AI and manual analysis"""
        st.header("📊 Integrated Market View")
        st.success("Combining AI insights with quantitative analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("AI Perspectives")
            if "AI ANALYSIS UNAVAILABLE" not in insights_text:
                # Extract key points from AI analysis
                ai_key_points = self._extract_ai_key_points(insights_text)
                for point in ai_key_points:
                    st.write(f"• {point}")
            else:
                st.info("AI insights currently unavailable")
        
        with col2:
            st.subheader("Quantitative Signals")
            self._display_quantitative_signals(selected_stocks)
        
        # Recommendations section
        st.markdown("---")
        st.subheader("Combined Recommendations")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.write("**Short-term (1-4 weeks)**")
            st.info("Monitor regime transitions and volume patterns")
        
        with rec_col2:
            st.write("**Medium-term (1-3 months)**")
            st.info("Watch for correlation breakdowns between assets")
        
        with rec_col3:
            st.write("**Long-term (3+ months)**")
            st.info("Focus on fundamental drivers and macro trends")

    def _parse_and_display_ai_response(self, insights_text):
        """Parse and display the AI response in a structured format"""
        lines = [line.strip() for line in insights_text.split('\n') if line.strip()]
        
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('STOCK:') or line.startswith('PORTFOLIO') or line.startswith('RECOMMENDATION'):
                # Process previous section
                if current_section and current_content:
                    self._display_ai_section(current_section, '\n'.join(current_content))
                
                # Start new section
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Process final section
        if current_section and current_content:
            self._display_ai_section(current_section, '\n'.join(current_content))

    # Color coding based on recommendation
    def _get_recommendation_color(self, recommendation):
        colors = {
            'buy': 'green',
            'strong buy': 'darkgreen',
            'hold': 'blue',
            'sell': 'red',
            'strong sell': 'darkred'
        }
        return colors.get(recommendation.lower(), 'blue')

    # Progress bars for risk levels
    def _display_risk_level(self, risk_text):
        risk_map = {'low': 25, 'medium': 50, 'high': 75, 'very high': 95}
        level = risk_map.get(risk_text.lower(), 50)
        st.progress(level/100)
        st.write(f"**Risk Level:** {risk_text}")

    def _display_ai_section(self, section_header, content):
        """Display a section of AI analysis with enhanced parsing and error handling"""
        
        if not content or not content.strip():
            st.warning(f"No content available for {section_header}")
            return
        
        try:
            if section_header.startswith('STOCK:'):
                self._display_stock_section(section_header, content)
            
            elif section_header.startswith('PORTFOLIO'):
                self._display_portfolio_section(content)
            
            elif section_header.startswith('RECOMMENDATION'):
                self._display_recommendation_section(content)
            
            else:
                st.subheader(section_header)
                st.write(content)
                
        except Exception as e:
            st.error(f"Error displaying section: {str(e)}")
            st.text_area("Raw Content", content, height=150)

    def _display_stock_section(self, section_header, content):
        """Display stock-specific analysis with enhanced parsing"""
        stock_name = section_header.replace('STOCK:', '').strip()
        st.subheader(f"📈 {stock_name}")
        
        # Parse key-value pairs more robustly
        parsed_data = {}
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                parsed_data[key] = value
        
        # Display in a consistent order with better formatting
        display_order = [
            ('current price', 'Current Price', 'metric'),
            ('expected return', 'Expected Return', 'metric'),
            ('market regime', 'Market Regime', 'metric'),
            ('ai recommendation', 'AI Recommendation', 'info'),
            ('risk level', 'Risk Level', 'warning'),
            ('key insight', 'Key Insight', 'success')
        ]
        
        for key, display_name, display_type in display_order:
            if key in parsed_data:
                if display_type == 'metric':
                    st.metric(display_name, parsed_data[key])
                elif display_type == 'info':
                    st.info(f"**{display_name}:** {parsed_data[key]}")
                elif display_type == 'warning':
                    st.warning(f"**{display_name}:** {parsed_data[key]}")
                elif display_type == 'success':
                    st.success(f"**{display_name}:** {parsed_data[key]}")

    def _display_portfolio_section(self, content):
        """Display portfolio overview"""
        st.subheader("Portfolio Overview")
        
        # Add potential for structured portfolio data
        if any(keyword in content.lower() for keyword in ['diversification', 'allocation', 'risk']):
            st.info("**Portfolio Analysis:**")
            st.write(content)
        else:
            st.write(content)

    def _display_recommendation_section(self, content):
        """Display strategic recommendations"""
        st.subheader("Strategic Recommendations")
        
        # Handle bullet points or numbered lists
        if '\n' in content:
            for line in content.split('\n'):
                if line.strip():
                    st.success(f"• {line.strip()}")
        else:
            st.success(content)

    def _extract_ai_key_points(self, insights_text):
        """Extract key points from AI analysis"""
        key_points = []
        lines = [line.strip() for line in insights_text.split('\n') if line.strip()]
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'advise', 'opportunity', 'risk', 'warning']):
                if len(line) > 20:  # Substantive points only
                    key_points.append(line)
        
        return key_points[:5]  # Return top 5 points

    def _display_quantitative_signals(self, selected_stocks):
        """Display quantitative trading signals"""
        signals = []
        
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                regime = st.session_state.regime_data[stock].iloc[-1]
                duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
                
                # Generate signal based on regime and duration
                if regime == "Bull" and duration > 10:
                    signals.append(f"🟢 {stock}: Strong bullish trend ({duration} days)")
                elif regime == "Bear" and duration > 10:
                    signals.append(f"🔴 {stock}: Strong bearish trend ({duration} days)")
                elif regime == "Stable":
                    signals.append(f"🟡 {stock}: Consolidating phase")
                else:
                    signals.append(f"⚪ {stock}: Regime transition detected")
        
        for signal in signals:
            st.write(signal)

    def _display_stock_technical_analysis(self, stock):
        """Display technical analysis for individual stock"""
        if stock not in st.session_state.df.columns:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price analysis
            current_price = st.session_state.df[stock].iloc[-1]
            prev_price = st.session_state.df[stock].iloc[-2] if len(st.session_state.df) > 1 else current_price
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            st.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
            
            # Volatility
            if f"{stock}_log_return" in st.session_state.df.columns:
                volatility = st.session_state.df[f"{stock}_log_return"].std() * np.sqrt(252)
                st.metric("Annual Volatility", f"{volatility:.2%}")
        
        with col2:
            # Regime analysis
            if stock in st.session_state.regime_data.columns:
                regime = st.session_state.regime_data[stock].iloc[-1]
                duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
                
                st.metric("Current Regime", regime)
                st.metric("Regime Duration", f"{duration} days")
                
                # Recommendation based on regime
                if regime == "Bull":
                    st.success("**Manual Recommendation:** Consider buying on dips")
                elif regime == "Bear":
                    st.error("**Manual Recommendation:** Consider reducing exposure")
                else:
                    st.warning("**Manual Recommendation:** Wait for clearer direction")

    # Update the main method call in the tabs section
    def display_enhanced_ai_insights_wrapper(self, selected_stocks):
        """Wrapper method for the enhanced AI insights with separate manual insights tab"""
        # Create tabs for AI Insights and Manual Insights
        ai_tab, manual_tab = st.tabs([
            "🤖 AI Raw Insights", 
            "📊 Formatted Insights"
        ])
        
        with ai_tab:
            self.display_enhanced_ai_insights(selected_stocks)
        
        with manual_tab:
            self.display_manual_insights(selected_stocks)

    def display_manual_insights(self, selected_stocks):
        """Display comprehensive manual insights based on quantitative analysis"""
        st.markdown("## Formatted Market Insights")
        
        if st.session_state.df is None or st.session_state.regime_data is None:
            st.info("Run analysis first to generate manual insights")
            return
        
        # Overall Market Summary
        st.markdown("### Overall Market Summary")
        self._display_overall_market_summary(selected_stocks)
        
        # Individual Stock Analysis
        st.markdown("### 📈 Individual Stock Analysis")
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                with st.expander(f"{stock} - Detailed Analysis", expanded=True):
                    self._display_stock_manual_analysis(stock)
        
        # Risk Assessment
        st.markdown("### ⚠️ Risk Assessment")
        self._display_risk_assessment(selected_stocks)
        
        # Trading Recommendations
        st.markdown("### Trading Recommendations")
        self._display_trading_recommendations(selected_stocks)

    def _display_overall_market_summary(self, selected_stocks):
        """Display overall market summary"""
        if st.session_state.regime_data is None:
            return
        
        # Calculate regime distribution
        regime_counts = {'Bull': 0, 'Bear': 0, 'Stable': 0}
        regime_durations = []
        
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                current_regime = st.session_state.regime_data[stock].iloc[-1]
                regime_counts[current_regime] += 1
                duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
                regime_durations.append(duration)
        
        total_stocks = len(selected_stocks)
        avg_duration = np.mean(regime_durations) if regime_durations else 0
        
        # Market sentiment indicator
        bull_ratio = regime_counts['Bull'] / total_stocks
        bear_ratio = regime_counts['Bear'] / total_stocks
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bullish Stocks", f"{regime_counts['Bull']}/{total_stocks}", 
                     f"{bull_ratio:.1%}")
        
        with col2:
            st.metric("Bearish Stocks", f"{regime_counts['Bear']}/{total_stocks}",
                     f"{bear_ratio:.1%}")
        
        with col3:
            st.metric("Avg Regime Duration", f"{avg_duration:.1f} days")
        
        # Market sentiment assessment
        if bull_ratio > 0.6:
            st.success("**Market Sentiment:** 🟢 Strongly Bullish - Favorable conditions for long positions")
        elif bear_ratio > 0.6:
            st.error("**Market Sentiment:** 🔴 Strongly Bearish - Consider defensive positioning")
        elif abs(bull_ratio - bear_ratio) < 0.2:
            st.warning("**Market Sentiment:** 🟡 Mixed/Neutral - Market is indecisive")
        elif bull_ratio > bear_ratio:
            st.info("**Market Sentiment:** 🔵 Moderately Bullish - Cautious optimism recommended")
        else:
            st.info("**Market Sentiment:** 🔵 Moderately Bearish - Risk management crucial")

    def _display_stock_manual_analysis(self, stock):
        """Display detailed manual analysis for individual stock"""
        if stock not in st.session_state.df.columns:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Analysis")
            
            # Current price and recent performance
            current_price = st.session_state.df[stock].iloc[-1]
            week_ago_price = st.session_state.df[stock].iloc[-5] if len(st.session_state.df) > 5 else current_price
            month_ago_price = st.session_state.df[stock].iloc[-20] if len(st.session_state.df) > 20 else current_price
            
            weekly_change = ((current_price - week_ago_price) / week_ago_price) * 100
            monthly_change = ((current_price - month_ago_price) / month_ago_price) * 100
            
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Weekly Change", f"{weekly_change:+.2f}%")
            st.metric("Monthly Change", f"{monthly_change:+.2f}%")
            
            # Volatility analysis
            if f"{stock}_log_return" in st.session_state.df.columns:
                returns = st.session_state.df[f"{stock}_log_return"].dropna()
                volatility_30d = returns.tail(30).std() * np.sqrt(252) * 100
                volatility_90d = returns.tail(90).std() * np.sqrt(252) * 100
                
                st.metric("30D Volatility", f"{volatility_30d:.1f}%")
                st.metric("90D Volatility", f"{volatility_90d:.1f}%")
        
        with col2:
            st.subheader("Regime Analysis")
            
            if stock in st.session_state.regime_data.columns:
                current_regime = st.session_state.regime_data[stock].iloc[-1]
                duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
                
                # Regime statistics
                regime_series = st.session_state.regime_data[stock]
                total_periods = len(regime_series)
                regime_frequency = regime_series.value_counts() / total_periods
                
                st.metric("Current Regime", current_regime)
                st.metric("Duration", f"{duration} days")
                
                # Regime frequency
                st.write("**Regime Frequency:**")
                for regime, freq in regime_frequency.items():
                    st.progress(freq, text=f"{regime}: {freq:.1%}")
                
                # Regime strength assessment
                if duration > 30:
                    st.success("**Strong Trend** - Regime has persisted significantly")
                elif duration > 10:
                    st.info("**Moderate Trend** - Established direction")
                else:
                    st.warning("**Early Stage** - Monitor for confirmation")
        
        # Monte Carlo Insights if available
        if 'monte_carlo' in st.session_state.analysis_results and stock in st.session_state.analysis_results['monte_carlo']:
            st.subheader(" Monte Carlo Projections")
            mc_results = st.session_state.analysis_results['monte_carlo'][stock]
            
            final_prices = mc_results['paths'][-1, :]
            expected_price = np.mean(final_prices)
            current_price = mc_results['current_price']
            expected_return = (expected_price / current_price - 1) * 100
            
            # Risk metrics
            var_5 = np.percentile(final_prices, 5)
            var_1 = np.percentile(final_prices, 1)
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Expected Price", f"${expected_price:.2f}")
                st.metric("Projected Return", f"{expected_return:.1f}%")
            
            with col4:
                st.metric("VaR (95%)", f"${var_5:.2f}")
                st.metric("Drawdown Risk", f"{(current_price - var_5)/current_price:.1%}")
            
            with col5:
                st.metric("VaR (99%)", f"${var_1:.2f}")
                st.metric("Extreme Risk", f"{(current_price - var_1)/current_price:.1%}")

    def _display_risk_assessment(self, selected_stocks):
        """Display comprehensive risk assessment"""
        st.info("### Portfolio Risk Analysis")
        
        risk_metrics = []
        
        for stock in selected_stocks:
            if stock in st.session_state.df.columns and f"{stock}_log_return" in st.session_state.df.columns:
                returns = st.session_state.df[f"{stock}_log_return"].dropna()
                
                if len(returns) > 0:
                    # Calculate various risk metrics
                    volatility = returns.std() * np.sqrt(252)
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    max_drawdown = (returns.cumsum().expanding().max() - returns.cumsum()).max()
                    
                    # Current regime risk
                    current_regime = st.session_state.regime_data[stock].iloc[-1] if stock in st.session_state.regime_data.columns else "Unknown"
                    regime_duration = self.calculate_regime_duration(st.session_state.regime_data[stock]) if stock in st.session_state.regime_data.columns else 0
                    
                    risk_metrics.append({
                        'stock': stock,
                        'volatility': volatility,
                        'sharpe': sharpe,
                        'max_drawdown': max_drawdown,
                        'regime': current_regime,
                        'regime_duration': regime_duration
                    })
        
        if risk_metrics:
            # Display risk metrics in a table
            risk_df = pd.DataFrame(risk_metrics)
            st.dataframe(risk_df.style.format({
                'volatility': '{:.2%}',
                'sharpe': '{:.2f}',
                'max_drawdown': '{:.2%}'
            }), use_container_width=True)
            
            # Overall risk assessment
            avg_volatility = risk_df['volatility'].mean()
            avg_drawdown = risk_df['max_drawdown'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if avg_volatility > 0.4:
                    st.error("**High Volatility Environment** - Elevated risk levels")
                elif avg_volatility > 0.25:
                    st.warning("**Moderate Volatility** - Standard risk management required")
                else:
                    st.success("**Low Volatility** - Relatively stable conditions")
            
            with col2:
                if avg_drawdown > 0.3:
                    st.error("**High Drawdown Risk** - Significant loss potential")
                elif avg_drawdown > 0.15:
                    st.warning("**Moderate Drawdown Risk** - Standard precautions needed")
                else:
                    st.success("**Low Drawdown Risk** - Favorable risk profile")

    def _display_trading_recommendations(self, selected_stocks):
        """Display trading recommendations based on analysis"""
        st.success("### Trading Strategy Recommendations")
        
        recommendations = []
        
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                current_regime = st.session_state.regime_data[stock].iloc[-1]
                duration = self.calculate_regime_duration(st.session_state.regime_data[stock])
                
                # Generate recommendation based on regime and duration
                if current_regime == "Bull":
                    if duration > 30:
                        rec = "Consider taking profits - extended bullish phase"
                        confidence = "Medium"
                    elif duration > 10:
                        rec = "Buy on dips - strong bullish trend"
                        confidence = "High"
                    else:
                        rec = "Monitor for confirmation - early bullish phase"
                        confidence = "Low"
                
                elif current_regime == "Bear":
                    if duration > 30:
                        rec = "Consider bottom fishing - extended bearish phase"
                        confidence = "Medium"
                    elif duration > 10:
                        rec = "Reduce exposure - strong bearish trend"
                        confidence = "High"
                    else:
                        rec = "Wait for confirmation - early bearish phase"
                        confidence = "Low"
                
                else:  # Stable regime
                    rec = "Range-bound trading - wait for breakout"
                    confidence = "Medium"
                
                recommendations.append({
                    'Stock': stock,
                    'Regime': current_regime,
                    'Duration': f"{duration} days",
                    'Recommendation': rec,
                    'Confidence': confidence
                })
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True)
            
            # Overall portfolio recommendation
            bull_count = len([r for r in recommendations if r['Regime'] == 'Bull'])
            bear_count = len([r for r in recommendations if r['Regime'] == 'Bear'])
            
            if bull_count > bear_count:
                st.success("**Overall Portfolio Bias:** 🟢 Bullish - Favor long positions")
            elif bear_count > bull_count:
                st.error("**Overall Portfolio Bias:** 🔴 Bearish - Favor short positions or hedging")
            else:
                st.warning("**Overall Portfolio Bias:** 🟡 Neutral - Balanced approach recommended")


    

def main():
    # Initialize enhanced app
    app = ProbabilisticMarketApp()
    
    # Create animated header
    app.create_animated_header()
    
    # Setup enhanced sidebar
    (run_analysis, selected_stocks, start_date, end_date, 
     mc_simulations, forecast_days, include_lv, lv_lookback) = app.create_enhanced_sidebar()
    
    # Run analysis if button clicked
    if run_analysis:
        if selected_stocks:
            app.run_complete_analysis(selected_stocks, start_date, end_date, 
                                    mc_simulations, forecast_days, include_lv, lv_lookback)
            st.success("✅ Analysis completed successfully!")
            st.rerun()  # Refresh to show results
        else:
            st.error("Please select at least one stock to analyze")
    
    # Display main dashboard
    if st.session_state.df is not None:
        app.create_enhanced_tabs(selected_stocks)
    else:
        # Welcome screen when no data
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; margin: 2rem 0;">
            <div style="font-size: 6rem; margin-bottom: 2rem;"></div>
            <h2 style="margin-bottom: 1rem; color: #667eea;">Welcome to Market Analysis</h2>
            <p style="font-size: 1.2rem; opacity: 0.8; max-width: 600px; margin: 0 auto;">
                Configure your analysis parameters in the sidebar and click "Launch Analysis" 
                to begin exploring market dynamics with advanced probabilistic models.
            </p>
            <div style="margin-top: 2rem; opacity: 0.6;">
                <p>Real-time Analysis • Regime Detection •  Monte Carlo •  Market Ecology • AI Insights</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()