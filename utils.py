import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class VisualizationUtils:
    def __init__(self):
        self.color_scheme = {
            'Bull': '#00FF00',
            'Bear': '#FF0000', 
            'Stable': '#FFFF00',
            'AAPL': '#1d77b4',
            'MSFT': '#ff7f0e',
            'GOOGL': '#2ca02c',
            '^GSPC': '#d62728'
        }
    
    def create_price_plot(self, df, selected_stocks):
        """FIXED: Normalize prices for proper comparison"""
        fig = go.Figure()
    
    # Normalize prices to percentage scale
        for stock in selected_stocks:
            if stock in df.columns:
            # Calculate percentage change from first value
                base_price = df[stock].iloc[0]
                normalized_prices = (df[stock] / base_price - 1) * 100
            
                fig.add_trace(go.Scatter(
                    x=df.index, y=normalized_prices,
                    name=stock,
                    line=dict(width=2),
                    hovertemplate=f'{stock}<br>Date: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>'
                ))
    
        fig.update_layout(
            title='Stock Performance Comparison (Normalized)',
            xaxis_title='Date',
            yaxis_title='Percentage Return (%)',
            hovermode='x unified',
            height=400
        )
    
        return fig

    def create_individual_price_plots(self, df, selected_stocks):
        """Create separate subplots for each stock"""
        fig = make_subplots(
            rows=len(selected_stocks), cols=1,
            subplot_titles=[f'{stock} Price' for stock in selected_stocks],
            vertical_spacing=0.05
        )
    
        for i, stock in enumerate(selected_stocks):
            if stock in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[stock], name=stock, line=dict(width=2)),
                    row=i+1, col=1
                )
    
        fig.update_layout(height=200*len(selected_stocks), showlegend=False)
        return fig
    
    def create_regime_timeline(self, regime_data, stock):
        """FIXED: Colorful regime timeline with distinct colors"""
        if stock not in regime_data.columns:
            return go.Figure()
    
        # Create color mapping with distinct colors
        regime_colors = {
            'Bull': '#00FF00',  # Bright Green
            'Bear': '#FF0000',  # Bright Red
            'Stable': '#FFFF00' # Bright Yellow
        }
    
        fig = go.Figure()
    
    # Create segments for each regime with proper colors
        current_regime = None
        start_date = regime_data.index[0]
        segments = []
    
        for date, regime in regime_data[stock].items():
            if regime != current_regime:
                if current_regime is not None:
                    segments.append({
                        'start': start_date,
                        'end': date,
                        'regime': current_regime,
                        'color': regime_colors.get(current_regime, '#000000')
                    })
                current_regime = regime
                start_date = date
    
    # Add the last segment
        if current_regime is not None:
            segments.append({
                'start': start_date,
                'end': regime_data.index[-1],
                'regime': current_regime,
                'color': regime_colors.get(current_regime, '#000000')
            })
    
    # Add segments to plot
        for seg in segments:
            fig.add_trace(go.Scatter(
                x=[seg['start'], seg['end']],
                y=[stock, stock],
                mode='lines',
                line=dict(color=seg['color'], width=20),
                name=seg['regime'],
                showlegend=False,
                hoverinfo='text',
                text=f"{seg['regime']} regime: {seg['start'].strftime('%Y-%m-%d')} to {seg['end'].strftime('%Y-%m-%d')}"
            ))
    
    # Customize layout
        fig.update_layout(
            title=f'{stock} - Market Regime Timeline',
            xaxis_title='Date',
            yaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
            height=150,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    
    # Add regime legend manually
        for regime, color in regime_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=regime
            ))
    
            return fig

    def create_monte_carlo_plot(self, simulation_results, stock):
        """Create Monte Carlo simulation visualization"""
        paths = simulation_results['paths']
        time = simulation_results['time']
        
        fig = go.Figure()
        
        # Plot individual paths (sample)
        n_paths_to_show = min(50, paths.shape[1])
        for i in range(n_paths_to_show):
            fig.add_trace(go.Scatter(
                x=time, y=paths[:, i],
                line=dict(width=1, color='lightgray'),
                showlegend=False,
                opacity=0.3
            ))
        
        # Plot percentiles
        percentile_5 = np.percentile(paths, 5, axis=1)
        percentile_50 = np.percentile(paths, 50, axis=1)
        percentile_95 = np.percentile(paths, 95, axis=1)
        
        fig.add_trace(go.Scatter(
            x=time, y=percentile_5,
            line=dict(color='red', dash='dash'),
            name='5th Percentile'
        ))
        
        fig.add_trace(go.Scatter(
            x=time, y=percentile_50,
            line=dict(color='blue', width=3),
            name='Median'
        ))
        
        fig.add_trace(go.Scatter(
            x=time, y=percentile_95,
            line=dict(color='green', dash='dash'),
            name='95th Percentile'
        ))
        
        fig.update_layout(
            title=f'{stock} - Monte Carlo Simulation (90% Confidence Interval)',
            xaxis_title='Days',
            yaxis_title='Price ($)',
            height=400
        )
        
        return fig
    
    def create_correlation_heatmap(self, df, selected_stocks):
        """FIXED: Better color scale and formatting"""
    # Filter for selected stocks
        price_data = df[[stock for stock in selected_stocks if stock in df.columns]]
    
    # Calculate returns correlation
        returns_data = price_data.pct_change().dropna()
        correlation_matrix = returns_data.corr()
    
    # Fix any NaN values
        correlation_matrix = correlation_matrix.fillna(0)
    
    # Create custom colorscale focused on the 0.5 to 1.0 range
        colorscale = [
            [0.0, 'blue'],      # -1 to -0.5: Blue
            [0.25, 'lightblue'], # -0.5 to 0: Light Blue  
            [0.5, 'white'],     # 0: White
            [0.75, 'lightcoral'], # 0 to 0.5: Light Red
            [1.0, 'red']        # 0.5 to 1: Red
        ]
    
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale=colorscale,
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
            )
        ))
    
    # Add annotations with proper formatting
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.index)):
                corr_value = correlation_matrix.iloc[j, i]
        # Ensure we're using the rounded value for display
                display_value = round(corr_value, 3)
        
                fig.add_annotation(
                    x=correlation_matrix.columns[i],
                    y=correlation_matrix.index[j],
                    text=f"{display_value:.3f}",
                    showarrow=False,
                    font=dict(
                        color="white" if abs(corr_value) > 0.3 else "black",
                        size=12
                    )
                )
    
        fig.update_layout(
            title='Stock Returns Correlation Matrix',
            height=500,
            xaxis_title="",
            yaxis_title="",
            font=dict(size=10)
        )
    
        return fig

    # NEW: LV Analysis Visualization Methods
    def create_ecology_plot(self, lv_data, stock):
        """Create ecology dynamics plot for LV analysis"""
        historical_data = lv_data.get('historical_data', pd.DataFrame())
        
        if len(historical_data) == 0:
            # Create a simple plot with current state only
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0], y=[lv_data['current_prey']],
                mode='markers', name='Current Buying',
                marker=dict(size=15, color='green')
            ))
            fig.add_trace(go.Scatter(
                x=[0], y=[lv_data['current_predator']],
                mode='markers', name='Current Selling',
                marker=dict(size=15, color='red')
            ))
        else:
            # Plot historical dynamics
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Prey'],
                name='Buying Pressure (Prey)',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Predator'],
                name='Selling Pressure (Predator)',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title=f'{stock} - Market Ecology Dynamics',
            xaxis_title='Date',
            yaxis_title='Pressure Level',
            height=300
        )
        return fig

    def create_phase_plot(self, lv_data, stock):
        """Create phase space plot for LV analysis"""
        historical_data = lv_data.get('historical_data', pd.DataFrame())
        
        fig = go.Figure()
        
        if len(historical_data) > 1:
            # Plot phase trajectory
            fig.add_trace(go.Scatter(
                x=historical_data['Prey'],
                y=historical_data['Predator'],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                name='Phase Trajectory'
            ))
        
        # Mark current state
        fig.add_trace(go.Scatter(
            x=[lv_data['current_prey']],
            y=[lv_data['current_predator']],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Current State'
        ))
        
        fig.update_layout(
            title=f'{stock} - Phase Space Analysis',
            xaxis_title='Buying Pressure',
            yaxis_title='Selling Pressure',
            height=300
        )
        return fig

    def create_lv_forecast_plot(self, lv_data, stock):
        """Create forecast plot for LV analysis"""
        future_path = lv_data.get('future_path', np.array([]))
        
        if len(future_path) == 0:
            return go.Figure()
        
        fig = go.Figure()
        days = len(future_path)
        
        fig.add_trace(go.Scatter(
            x=list(range(days)),
            y=future_path[:, 0],
            name='Projected Buying',
            line=dict(color='green', dash='dash', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(days)),
            y=future_path[:, 1],
            name='Projected Selling',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f'{stock} - 5-Day Ecology Forecast',
            xaxis_title='Days Ahead',
            yaxis_title='Pressure Level',
            height=300
        )
        return fig