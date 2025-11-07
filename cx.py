import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from data_loader import DataLoader
from models import MarkovRegimeModel, MonteCarloModel, CorrelationAnalyzer, MarketForceAnalyzer, LotkaVolterraModel
from llm_analyzer import FinancialAnalyzer
from utils import VisualizationUtils

# Configure page with modern styling
st.set_page_config(
    page_title="Probabilistic Market Analysis",
    page_icon="🌊",
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
            🌊 Probabilistic Market Ecosystem Analyzer
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
                <h2 style="color: white; margin: 0;">🎮 Control Center</h2>
                <p style="opacity: 0.8; margin: 0.5rem 0;">Configure Your Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Analysis Mode Selection with enhanced styling
            st.markdown("### 🎯 Analysis Mode")
            analysis_type = st.selectbox(
                "",
                ["🚀 Quick Analysis", "⚙️ Advanced Configuration"],
                label_visibility="collapsed"
            )
            
            st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
            
            # Stock selection with modern multiselect
            st.markdown("### 📈 Select Assets")
            available_stocks = ['AAPL', 'MSFT', 'GOOGL', '^GSPC']
            stock_options = {
                'AAPL': '🍎 Apple Inc.',
                'MSFT': '💻 Microsoft Corp.',
                'GOOGL': '🔍 Alphabet Inc.',
                '^GSPC': '📊 S&P 500'
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
            st.markdown("### ⚡ Analysis Parameters")
            
            if "Quick" in analysis_type:
                mc_simulations = 500
                forecast_days = 60
                st.info("🚀 Quick mode: Optimized for speed")
            else:
                mc_simulations = st.slider("🎲 Monte Carlo Simulations", 100, 5000, 1000, step=100)
                forecast_days = st.slider("🔮 Forecast Horizon (Days)", 30, 365, 90, step=15)
            
            # Market Force Analysis
            st.markdown("### ⚖️ Market Forces")
            include_lv = st.toggle("Enable Force Analysis", value=True)
            
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
                "Force Analysis": "✅" if include_lv else "❌"
            }
            
            for key, value in summary_data.items():
                st.markdown(f"**{key}:** {value}")
            
            # Action Button with enhanced styling
            st.markdown("---")
            run_analysis = st.button(
                "🚀 Launch Analysis", 
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
            
            st.markdown("### 📊 Market Status Board")
            
            # Create status cards
            cols = st.columns(len(selected_stocks))
            
            for i, stock in enumerate(selected_stocks):
                if stock in st.session_state.df.columns:
                    with cols[i]:
                        current_price = latest_data[stock]
                        prev_price = st.session_state.df[stock].iloc[-2] if len(st.session_state.df) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        # Determine status color
                        if change_pct > 1:
                            status = "🟢"
                            trend_color = "#2ca02c"
                        elif change_pct < -1:
                            status = "🔴"
                            trend_color = "#d62728"
                        else:
                            status = "🟡"
                            trend_color = "#ff7f0e"
                        
                        # Create enhanced metric card
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{status}</div>
                            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">{stock}</div>
                            <div style="font-size: 1.3rem; font-weight: bold; color: white;">${current_price:.2f}</div>
                            <div style="color: {trend_color}; font-weight: 500;">
                                {change_pct:+.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    def create_enhanced_tabs(self, selected_stocks):
        """Create enhanced tab interface"""
        # Create tabs with modern styling
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Dashboard", 
            "🔄 Regime Analysis", 
            "🎲 Monte Carlo", 
            "⚖️ Force Analysis", 
            "🤖 AI Insights"
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
            self.display_enhanced_ai_insights(selected_stocks)
    
    def display_enhanced_live_dashboard(self, selected_stocks):
        """Enhanced live dashboard with modern visuals"""
        st.markdown("## 📊 Real-Time Market Dashboard")
        
        if st.session_state.df is None:
            st.info("💡 Configure settings in the sidebar and launch analysis to see live data")
            return
        
        # Status cards at top
        self.create_status_dashboard(selected_stocks)
        
        # Charts section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Price Evolution")
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
                )
            )
            st.plotly_chart(price_chart, use_container_width=True)
        
        with col2:
            st.markdown("### 🔗 Correlations")
            correlation_chart = self.viz_utils.create_correlation_heatmap(st.session_state.df, selected_stocks)
            correlation_chart.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=10)
            )
            st.plotly_chart(correlation_chart, use_container_width=True)
        
        # Market Summary
        st.markdown("### 📋 Market Summary")
        if st.session_state.df is not None:
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                volatility = st.session_state.df[f'{selected_stocks[0]}_volatility_30d'].iloc[-1] if len(selected_stocks) > 0 else 0
                st.metric("Avg Volatility", f"{volatility:.1%}")
            
            with summary_cols[1]:
                correlation = st.session_state.df[[s for s in selected_stocks if s in st.session_state.df.columns]].corr().mean().mean()
                st.metric("Avg Correlation", f"{correlation:.2f}")
            
            with summary_cols[2]:
                data_points = len(st.session_state.df)
                st.metric("Data Points", f"{data_points:,}")
            
            with summary_cols[3]:
                last_update = st.session_state.df.index[-1].strftime("%Y-%m-%d")
                st.metric("Last Update", last_update)
    
    def display_enhanced_regime_analysis(self, selected_stocks):
        """Enhanced regime analysis with modern visuals"""
        st.markdown("## 🔄 Market Regime Analysis")
        
        if st.session_state.regime_data is None:
            st.info("Run analysis to see regime data")
            return
        
        # Overview section
        st.markdown("### 🎯 Regime Overview")
        
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
        st.markdown("### 📈 Detailed Regime Analysis")
        
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                with st.expander(f"🔍 {stock} Regime Details", expanded=True):
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
                        # Transition probabilities
                        current_regime = st.session_state.regime_data[stock].iloc[-1]
                        transition_matrix = self.calculate_stock_transition_matrix(st.session_state.regime_data[stock])
                        
                        st.markdown("**Transition Probabilities:**")
                        for regime in ['Bull', 'Bear', 'Stable']:
                            if regime in transition_matrix.index and current_regime in transition_matrix.index:
                                prob = transition_matrix.loc[current_regime, regime]
                                st.markdown(f"→ {regime}: {prob:.1%}")
    
    def display_enhanced_monte_carlo(self, selected_stocks):
        """Enhanced Monte Carlo analysis"""
        st.markdown("## 🎲 Monte Carlo Simulation")
        
        if 'monte_carlo' not in st.session_state.analysis_results:
            st.info("Run analysis to see Monte Carlo results")
            return
        
        mc_results = st.session_state.analysis_results['monte_carlo']
        
        for stock in selected_stocks:
            if stock in mc_results:
                with st.expander(f"📊 {stock} - Price Simulation", expanded=True):
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
                        st.markdown("**📊 Risk Metrics**")
                        
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
                                {icon} **{label}:** {value}
                            </div>
                            """, unsafe_allow_html=True)
    
    def display_enhanced_lv_analysis(self, selected_stocks):
        """Enhanced Market Force Analysis with modern visuals"""
        st.markdown("## ⚖️ Market Force Analysis")
        
        if not st.session_state.lv_results:
            st.info("Enable force analysis and run to see market dynamics insights")
            return
        
        # Force analysis overview
        st.markdown("### 📊 Force Balance Dashboard")
        
        force_cols = st.columns(len(selected_stocks))
        
        for i, stock in enumerate(selected_stocks):
            if stock in st.session_state.lv_results:
                with force_cols[i]:
                    force_data = st.session_state.lv_results[stock]
                    momentum = force_data['current_momentum']
                    contrarian = force_data['current_contrarian']
                    dominance = force_data['force_dominance']
                    
                    # Determine market state
                    if dominance > 1.5:
                        status = "🔵 Momentum"
                        status_color = "#1f77b4"
                    elif dominance < 0.67:
                        status = "🟠 Contrarian"
                        status_color = "#ff7f0e"
                    else:
                        status = "⚖️ Balanced"
                        status_color = "#2ca02c"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; border-left: 4px solid {status_color};">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚖️</div>
                        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">{stock}</div>
                        <div style="font-size: 1rem; font-weight: bold; color: {status_color};">{status}</div>
                        <div style="opacity: 0.8; font-size: 0.8rem;">Ratio: {dominance:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed force analysis
        st.markdown("### 🔬 Detailed Force Analysis")
        
        for stock in selected_stocks:
            if stock in st.session_state.lv_results:
                with st.expander(f"⚖️ {stock} - Market Forces", expanded=True):
                    self.display_enhanced_stock_force_analysis(stock)
    
    def display_enhanced_stock_force_analysis(self, stock):
        """Enhanced individual stock force analysis"""
        force_data = st.session_state.lv_results[stock]
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Enhanced force dynamics plot
            fig = self.viz_utils.create_force_dynamics_plot(force_data, stock)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16, color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**🎯 Current Forces**")
            
            # Force metrics
            metrics = [
                ("🔵 Momentum Force", f"{force_data['current_momentum']:.3f}"),
                ("🟠 Contrarian Force", f"{force_data['current_contrarian']:.3f}"),
                ("📈 Momentum Trend", f"{force_data['momentum_trend']:+.4f}"),
                ("📉 Contrarian Trend", f"{force_data['contrarian_trend']:+.4f}")
            ]
            
            for label, value in metrics:
                st.markdown(f"**{label}:** {value}")
        
        with col3:
            st.markdown("**🎯 Trading Signals**")
            
            for signal in force_data['signals']:
                if "MOMENTUM" in signal.upper() and ("STRONG" in signal.upper() or "BUILDING" in signal.upper()):
                    st.success(signal)
                elif "CONTRARIAN" in signal.upper() or "REVERSION" in signal.upper():
                    st.error(signal)
                elif "TENSION" in signal.upper() or "PEAK" in signal.upper():
                    st.warning(signal)
                else:
                    st.info(signal)
        
        # Phase space analysis
        col4, col5 = st.columns(2)
        
        with col4:
            phase_fig = self.viz_utils.create_force_phase_plot(force_data, stock)
            phase_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=14)
            )
            st.plotly_chart(phase_fig, use_container_width=True)
        
        with col5:
            st.markdown("**🎭 Market State Interpretation**")
            
            # Market state analysis
            momentum = force_data['current_momentum']
            contrarian = force_data['current_contrarian']
            tension = force_data['market_tension']
            
            if momentum > 0.6 and contrarian < 0.4:
                state_desc = "🔵 **Momentum Phase**: Trend-following dominates"
            elif momentum < 0.4 and contrarian > 0.6:
                state_desc = "🟠 **Contrarian Phase**: Mean reversion active"
            elif tension > 0.6:
                state_desc = "🔴 **High Tension**: Major move expected"
            elif tension < 0.2:
                state_desc = "🟢 **Low Conviction**: Range-bound market"
            else:
                state_desc = "⚖️ **Balanced Phase**: Forces in equilibrium"
            
            st.markdown(state_desc)
            
            # Model parameters
            st.markdown("**⚙️ Economic Parameters:**")
            params = [
                ("Self-Reinforcement", f"{force_data['self_reinforcement']:.4f}"),
                ("Contrarian Damping", f"{force_data['contrarian_damping']:.4f}"),
                ("Opportunity Response", f"{force_data['opportunity_response']:.4f}"),
                ("Decay Rate", f"{force_data['decay_rate']:.4f}")
            ]
            
            for param, value in params:
                st.text(f"{param}: {value}")
    
    def display_enhanced_ai_insights(self, selected_stocks):
        """Enhanced AI insights with modern presentation"""
        st.markdown("## 🤖 AI-Powered Market Intelligence")
        
        if st.session_state.df is None or st.session_state.regime_data is None:
            st.info("Complete the analysis to generate AI insights")
            return
        
        # Prepare data for AI analysis
        market_data = {}
        mathematical_results = {}
        
        latest = st.session_state.df.iloc[-1]
        for stock in selected_stocks:
            if stock in st.session_state.regime_data.columns:
                market_data[stock] = {
                    'price': latest[stock],
                    'regime': st.session_state.regime_data[stock].iloc[-1],
                    'volatility': latest.get(f'{stock}_volatility_30d', 0)
                }
        
        if 'monte_carlo' in st.session_state.analysis_results:
            mathematical_results['monte_carlo'] = st.session_state.analysis_results['monte_carlo']
        
        if st.session_state.lv_results:
            mathematical_results['ecology'] = st.session_state.lv_results
        
        # AI Analysis section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Analysis Control")
            
            # Enhanced button with loading state
            if st.button("🚀 Generate AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 AI analyzing market conditions..."):
                    # Progress bar for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    insights = self.llm_analyzer.generate_insights(market_data, mathematical_results)
                    st.session_state.ai_insights = insights
                    progress_bar.empty()
            
            # Analysis options
            st.markdown("### ⚙️ AI Options")
            analysis_depth = st.selectbox("Analysis Depth", ["Quick Overview", "Detailed Analysis", "Deep Dive"])
            include_predictions = st.checkbox("Include Predictions", value=True)
            include_risks = st.checkbox("Risk Assessment", value=True)
        
        with col2:
            st.markdown("### 📊 AI Market Analysis")
            
            if hasattr(st.session_state, 'ai_insights'):
                # Display insights in enhanced format
                insights = st.session_state.ai_insights
                
                # Parse and format the insights
                lines = insights.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('STOCK:'):
                        current_section = "stock"
                        stock_name = line.replace('STOCK:', '').strip()
                        st.markdown(f"#### 📈 {stock_name}")
                        
                    elif line.startswith('Current Price:'):
                        price = line.replace('Current Price:', '').strip()
                        st.markdown(f"**💰 Price:** {price}")
                        
                    elif line.startswith('Recommendation:'):
                        rec = line.replace('Recommendation:', '').strip()
                        if 'Buy' in rec:
                            st.success(f"🟢 **Recommendation:** {rec}")
                        elif 'Sell' in rec:
                            st.error(f"🔴 **Recommendation:** {rec}")
                        else:
                            st.warning(f"🟡 **Recommendation:** {rec}")
                            
                    elif line.startswith('Risk Level:'):
                        risk = line.replace('Risk Level:', '').strip()
                        risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        risk_icon = risk_colors.get(risk, "⚪")
                        st.markdown(f"**{risk_icon} Risk Level:** {risk}")
                        
                    elif line.startswith('Key Insight:'):
                        insight = line.replace('Key Insight:', '').strip()
                        st.info(f"💡 {insight}")
                        
                    elif line.startswith('PORTFOLIO SUMMARY:'):
                        st.markdown("---")
                        st.markdown("### 📋 Portfolio Summary")
                        
                    elif line.startswith('- Overall:'):
                        overall = line.replace('- Overall:', '').strip()
                        st.markdown(f"**🎯 Overall:** {overall}")
                        
                    elif line.startswith('- Opportunities:'):
                        opp = line.replace('- Opportunities:', '').strip()
                        st.markdown(f"**🚀 Opportunities:** {opp}")
                        
                    elif line.startswith('- Risks:'):
                        risks = line.replace('- Risks:', '').strip()
                        st.markdown(f"**⚠️ Risks:** {risks}")
                        
                    elif line.startswith('- Strategy:'):
                        strategy = line.replace('- Strategy:', '').strip()
                        st.markdown(f"**📋 Strategy:** {strategy}")
                        
                    elif line.startswith('ANALYSIS DATE:'):
                        date = line.replace('ANALYSIS DATE:', '').strip()
                        st.markdown(f"**📅 Generated:** {date}")
                
                # Additional insights visualization
                st.markdown("---")
                st.markdown("### 📊 Insights Summary")
                
                # Create summary metrics
                summary_cols = st.columns(4)
                
                # Count recommendations
                buy_count = insights.count('Buy')
                sell_count = insights.count('Sell')
                hold_count = insights.count('Hold')
                
                with summary_cols[0]:
                    st.metric("🟢 Buy Signals", buy_count)
                
                with summary_cols[1]:
                    st.metric("🔴 Sell Signals", sell_count)
                
                with summary_cols[2]:
                    st.metric("🟡 Hold Signals", hold_count)
                
                with summary_cols[3]:
                    total_analyzed = buy_count + sell_count + hold_count
                    confidence = min(100, (total_analyzed / len(selected_stocks)) * 100) if selected_stocks else 0
                    st.metric("🎯 Analysis Coverage", f"{confidence:.0f}%")
            
            else:
                # Placeholder content
                st.markdown("""
                <div style="text-align: center; padding: 3rem; opacity: 0.6;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                    <h3>AI Analysis Ready</h3>
                    <p>Click "Generate AI Analysis" to get intelligent market insights powered by advanced language models.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Core logic methods (unchanged from original)
    def run_complete_analysis(self, selected_stocks, start_date, end_date, 
                            mc_simulations, forecast_days, include_lv, lv_lookback):
        """Run the complete analysis pipeline"""
        with st.spinner("📊 Loading data and running analysis..."):
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
            
            # Run Market Force Analysis
            if include_lv and len(selected_stocks) > 0:
                with st.spinner("⚖️ Analyzing market forces..."):
                    try:
                        st.session_state.lv_results = self.lv_model.analyze_multiple_stocks(
                            st.session_state.df, selected_stocks, lookback_days=lv_lookback
                        )
                        print(f"✓ Force analysis completed. Results keys: {list(st.session_state.lv_results.keys())}")
                        
                        # Debug: Show what we got
                        if st.session_state.lv_results:
                            st.success(f"Market Force Analysis completed for {len(st.session_state.lv_results)} stocks")
                        else:
                            st.warning("Market Force Analysis returned no results")
                            
                    except Exception as e:
                        st.error(f"Market Force Analysis failed: {str(e)}")
                        print(f"Force analysis error: {e}")
                        st.session_state.lv_results = {}
            elif include_lv:
                st.warning("No stocks selected for force analysis")
            else:
                st.session_state.lv_results = {}
    
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
    
    def calculate_raw_transition_counts(self, stock_regime_series):
        """Calculate raw transition counts before normalization"""
        states = ['Bull', 'Bear', 'Stable']
        count_matrix = pd.DataFrame(0, index=states, columns=states)
        
        for i in range(len(stock_regime_series)-1):
            from_state = stock_regime_series.iloc[i]
            to_state = stock_regime_series.iloc[i+1]
            if from_state in states and to_state in states:
                count_matrix.loc[from_state, to_state] += 1
        
        return count_matrix
    
    def calculate_regime_statistics(self, stock_regime_series):
        """Calculate comprehensive regime statistics"""
        stats = {}
        
        # Count total days in each regime
        regime_counts = stock_regime_series.value_counts()
        total_days = len(stock_regime_series)
        
        # Calculate percentages
        for regime in ['Bull', 'Bear', 'Stable']:
            count = regime_counts.get(regime, 0)
            percentage = (count / total_days) * 100 if total_days > 0 else 0
            stats[f'{regime}_Days'] = count
            stats[f'{regime}_Percentage'] = f"{percentage:.1f}%"
        
        # Calculate average regime durations
        regime_durations = self.calculate_all_regime_durations(stock_regime_series)
        for regime in ['Bull', 'Bear', 'Stable']:
            durations = regime_durations.get(regime, [])
            avg_duration = np.mean(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            stats[f'{regime}_Avg_Duration'] = f"{avg_duration:.1f}"
            stats[f'{regime}_Max_Duration'] = max_duration
        
        # Convert to DataFrame for display
        stats_df = pd.DataFrame([stats])
        return stats_df.T.rename(columns={0: 'Value'})
    
    def calculate_all_regime_durations(self, regime_series):
        """Calculate durations of all regime periods"""
        durations = {'Bull': [], 'Bear': [], 'Stable': []}
        
        if len(regime_series) == 0:
            return durations
        
        current_regime = regime_series.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regime_series)):
            if regime_series.iloc[i] == current_regime:
                current_duration += 1
            else:
                # Regime change - record duration
                if current_regime in durations:
                    durations[current_regime].append(current_duration)
                current_regime = regime_series.iloc[i]
                current_duration = 1
        
        # Don't forget the last regime
        if current_regime in durations:
            durations[current_regime].append(current_duration)
        
        return durations

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
            <div style="font-size: 6rem; margin-bottom: 2rem;">🌊</div>
            <h2 style="margin-bottom: 1rem; color: #667eea;">Welcome to Market Analysis</h2>
            <p style="font-size: 1.2rem; opacity: 0.8; max-width: 600px; margin: 0 auto;">
                Configure your analysis parameters in the sidebar and click "Launch Analysis" 
                to begin exploring market dynamics with advanced probabilistic models.
            </p>
            <div style="margin-top: 2rem; opacity: 0.6;">
                <p>📊 Real-time Analysis • 🔄 Regime Detection • 🎲 Monte Carlo • ⚖️ Market Forces • 🤖 AI Insights</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()