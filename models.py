import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from datetime import datetime, timedelta

class MarkovRegimeModel:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.states = ['Bull', 'Bear', 'Stable']
        self.transition_matrix = None
    
    def classify_regime(self, returns_series, window=20):
        """Classify market regime based on rolling z-score"""
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        z_scores = (returns_series - rolling_mean) / rolling_std
        
        # Define regime thresholds
        regimes = pd.Series(index=returns_series.index, dtype=object)
        regimes[z_scores > 0.5] = 'Bull'
        regimes[z_scores < -0.5] = 'Bear'
        regimes[(z_scores >= -0.5) & (z_scores <= 0.5)] = 'Stable'
        
        return regimes.fillna('Stable')
    
    def calculate_transition_matrix(self, regime_series):
        """Calculate Markov transition probabilities"""
        transitions = []
        for i in range(len(regime_series) - 1):
            transitions.append((regime_series.iloc[i], regime_series.iloc[i + 1]))
        
        # Initialize transition matrix
        transition_matrix = pd.DataFrame(0, index=self.states, columns=self.states)
        
        # Count transitions
        for from_state, to_state in transitions:
            if from_state in self.states and to_state in self.states:
                transition_matrix.loc[from_state, to_state] += 1
        
        # Normalize to probabilities
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
        self.transition_matrix = transition_matrix.fillna(0)
        
        return self.transition_matrix
    
    def predict_next_regime(self, current_regime):
        """Predict probabilities for next regime"""
        if self.transition_matrix is not None and current_regime in self.states:
            return self.transition_matrix.loc[current_regime]
        return pd.Series([0.33, 0.33, 0.33], index=self.states)

class MonteCarloModel:
    def __init__(self, n_simulations=1000):
        self.n_simulations = n_simulations
    
    def geometric_brownian_motion(self, S0, mu, sigma, T=252, dt=1):
        """FIXED: Simulate stock price using GBM"""
        n_steps = int(T / dt)
    
        # Calculate daily parameters
        mu_daily = mu / 252
        sigma_daily = sigma / np.sqrt(252)
    
    # Generate random shocks
        shocks = np.random.standard_normal(size=(n_steps, self.n_simulations))
    
    # Calculate cumulative returns
        daily_returns = mu_daily * dt + sigma_daily * np.sqrt(dt) * shocks
        cumulative_returns = np.cumsum(daily_returns, axis=0)
    
    # Calculate prices
        S = S0 * np.exp(cumulative_returns)
    
        return np.arange(n_steps), S

    def simulate_stock_paths(self, historical_prices, days_forward=90):
        """FIXED: Run proper Monte Carlo simulation"""
    # Use simple returns instead of log returns for stability
        returns = historical_prices.pct_change().dropna()
    
    # More robust parameter estimation
        mu = np.mean(returns) * 252  # Annualized return
        sigma = np.std(returns) * np.sqrt(252)  # Annualized volatility
        S0 = historical_prices.iloc[-1]  # Current price
    
    # Ensure reasonable parameters
        mu = np.clip(mu, -0.5, 0.5)  # Limit to ±50% annual return
        sigma = np.clip(sigma, 0.05, 0.8)  # Limit volatility between 5% and 80%
    
        t, paths = self.geometric_brownian_motion(S0, mu, sigma, T=days_forward)
    
    # Calculate percentiles properly
        final_prices = paths[-1, :]
    
        return {
            'time': t,
            'paths': paths,
            'current_price': S0,
            'mu': mu,
            'sigma': sigma,
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_50': np.percentile(final_prices, 50),
            'percentile_95': np.percentile(final_prices, 95)
        }
    
    def calculate_var(self, paths, confidence=0.05):
        """Calculate Value at Risk"""
        final_returns = (paths[-1] / paths[0]) - 1
        var = np.percentile(final_returns, confidence * 100)
        return var
    
    def calculate_cvar(self, paths, confidence=0.05):
        """Calculate Conditional Value at Risk"""
        final_returns = (paths[-1] / paths[0]) - 1
        var = self.calculate_var(paths, confidence)
        cvar = final_returns[final_returns <= var].mean()
        return cvar

class CorrelationAnalyzer:
    def __init__(self):
        pass
    
    def calculate_rolling_correlations(self, df, window=30):
        """Calculate rolling correlations between stocks"""
        stocks = [col for col in df.columns if col in ['AAPL', 'MSFT', 'GOOGL', '^GSPC']]
        correlations = {}
        
        for i, stock1 in enumerate(stocks):
            for stock2 in stocks[i+1:]:
                col_name = f"{stock1}_{stock2}_corr"
                correlations[col_name] = df[stock1].rolling(window).corr(df[stock2])
        
        return pd.DataFrame(correlations)

class LotkaVolterraModel:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def calculate_improved_proxies(self, df, stock_column):
        """Calculate LV proxies for a specific stock - FIXED VERSION"""
        data = df[[stock_column]].copy()
        
        # Ensure we have enough data
        if len(data) < 30:
            return pd.DataFrame()
            
        data['Returns'] = data[stock_column].pct_change()
        
        # Calculate High/Low proxies using rolling windows
        data['High_Proxy'] = data[stock_column].rolling(window=5, min_periods=1).max()
        data['Low_Proxy'] = data[stock_column].rolling(window=5, min_periods=1).min()
        
        # Use volume from dataframe if available, otherwise create proxy
        if 'Volume' in df.columns:
            data['Volume'] = df['Volume']
        else:
            # Create volume proxy based on price movement magnitude
            data['Volume'] = data['Returns'].abs().rolling(window=5).mean() * 1000000  # Scale appropriately
        
        # **IMPROVED PREY (BULLISH) INDICATORS**
        
        # 1. Positive momentum with volume confirmation
        data['Positive_Returns'] = np.where(data['Returns'] > 0, data['Returns'], 0)
        data['Volume_Momentum'] = data['Positive_Returns'] * data['Volume']
        data['VM_MA'] = data['Volume_Momentum'].rolling(window=5, min_periods=1).mean()
        
        # 2. Support level strength (how well price holds above recent lows)
        data['Recent_Low'] = data['Low_Proxy'].rolling(window=10, min_periods=1).min()
        data['Above_Support'] = (data[stock_column] - data['Recent_Low']) / (data['Recent_Low'] + 1e-8)
        
        # 3. Bullish trend strength
        data['Price_Trend_5'] = data[stock_column].pct_change(5)
        data['Up_Days'] = data['Returns'].rolling(window=5, min_periods=1).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) == 5 else 0.5
        )
        
        # **IMPROVED PREDATOR (BEARISH) INDICATORS**
        
        # 1. Negative momentum with volume confirmation
        data['Negative_Returns'] = np.where(data['Returns'] < 0, abs(data['Returns']), 0)
        data['Negative_Volume_Momentum'] = data['Negative_Returns'] * data['Volume']
        data['NVM_MA'] = data['Negative_Volume_Momentum'].rolling(window=5, min_periods=1).mean()
        
        # 2. Resistance level pressure
        data['Recent_High'] = data['High_Proxy'].rolling(window=10, min_periods=1).max()
        data['Below_Resistance'] = (data['Recent_High'] - data[stock_column]) / (data['Recent_High'] + 1e-8)
        
        # 3. Bearish trend strength
        data['Down_Days'] = data['Returns'].rolling(window=5, min_periods=1).apply(
            lambda x: (x < 0).sum() / len(x) if len(x) == 5 else 0.5
        )
        
        # 4. Volatility expansion (often precedes declines)
        data['Volatility'] = data['Returns'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # **NORMALIZE INDICATORS PROPERLY**
        indicators_to_normalize = [
            'VM_MA', 'Above_Support', 'Up_Days', 
            'NVM_MA', 'Below_Resistance', 'Down_Days', 'Volatility'
        ]
        
        # Apply individual normalization to each indicator first
        for indicator in indicators_to_normalize:
            if indicator in data.columns:
                # Handle cases where all values might be the same
                if data[indicator].std() > 1e-8:  # Only normalize if there's variation
                    data[f'{indicator}_norm'] = (data[indicator] - data[indicator].min()) / (data[indicator].max() - data[indicator].min() + 1e-8)
                else:
                    data[f'{indicator}_norm'] = 0.5  # Neutral value if no variation
        
        # **COMPOSITE SCORES WITH PROPER WEIGHTING**
        data['Prey_Raw'] = (
            0.4 * data.get('VM_MA_norm', 0.5) +
            0.3 * data.get('Above_Support_norm', 0.5) + 
            0.3 * data.get('Up_Days_norm', 0.5)
        )
        
        data['Predator_Raw'] = (
            0.3 * data.get('NVM_MA_norm', 0.5) +
            0.3 * data.get('Below_Resistance_norm', 0.5) +
            0.2 * data.get('Down_Days_norm', 0.5) +
            0.2 * data.get('Volatility_norm', 0.5)
        )
        
        # **FINAL SMOOTHING AND NORMALIZATION**
        data['Prey'] = data['Prey_Raw'].rolling(window=3, min_periods=1, center=True).mean()
        data['Predator'] = data['Predator_Raw'].rolling(window=3, min_periods=1, center=True).mean()
        
        # Ensure values are between 0 and 1
        data['Prey'] = np.clip(data['Prey'], 0, 1)
        data['Predator'] = np.clip(data['Predator'], 0, 1)
        
        result = data[['Prey', 'Predator']].dropna()
        
        # Final check to ensure we have valid data
        if len(result) < 10:
            return pd.DataFrame()
            
        return result
    
    def estimate_robust_parameters(self, data):
        """Estimate LV parameters with better stability"""
        try:
            if len(data) < 20:
                return None, "Insufficient data"
            
            # Use larger window for better estimation
            window_size = min(40, len(data) // 2)
            recent_data = data.tail(window_size).copy()
            
            if len(recent_data) < 15:
                return None, "Not enough recent data"
            
            # Calculate differences with better handling
            recent_data['dPrey'] = recent_data['Prey'].diff().fillna(0)
            recent_data['dPredator'] = recent_data['Predator'].diff().fillna(0)
            
            # Remove extreme outliers more aggressively
            for col in ['dPrey', 'dPredator']:
                if recent_data[col].std() > 1e-8:
                    z_scores = np.abs((recent_data[col] - recent_data[col].mean()) / (recent_data[col].std() + 1e-8))
                    recent_data = recent_data[z_scores < 2]  # Remove beyond 2 std dev
            
            if len(recent_data) < 10:
                return None, "Too many outliers after filtering"
            
            # **IMPROVED PARAMETER ESTIMATION**
            # Use constrained optimization instead of linear regression
            
            def lotka_volterra_residuals(params):
                alpha, beta, delta, gamma = params
                residuals = []
                
                for i in range(1, len(recent_data)):
                    x = recent_data['Prey'].iloc[i-1]
                    y = recent_data['Predator'].iloc[i-1]
                    
                    # Predicted changes
                    dx_pred = alpha * x - beta * x * y
                    dy_pred = delta * x * y - gamma * y
                    
                    # Actual changes
                    dx_actual = recent_data['dPrey'].iloc[i]
                    dy_actual = recent_data['dPredator'].iloc[i]
                    
                    residuals.extend([dx_pred - dx_actual, dy_pred - dy_actual])
                
                return np.array(residuals)
            
            # Initial guess with reasonable constraints
            x0 = [0.1, 0.1, 0.1, 0.1]  # All parameters positive initially
            
            # Bounds to ensure positive parameters (except gamma can be slightly negative)
            bounds = [(0.001, 1.0), (0.001, 1.0), (0.001, 1.0), (-0.5, 1.0)]
            
            try:
                result = minimize(lambda params: np.sum(lotka_volterra_residuals(params)**2), 
                                x0, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    alpha, beta, delta, gamma = result.x
                    
                    # Validate parameters
                    if all(0.001 <= p <= 1.0 for p in [alpha, beta, delta]) and -0.5 <= gamma <= 1.0:
                        return (alpha, beta, delta, gamma), "Success"
                    else:
                        return None, "Parameters outside valid range"
                else:
                    return None, "Optimization failed"
                    
            except Exception as opt_error:
                # Fallback to Ridge regression if optimization fails
                return self._ridge_fallback(recent_data)
            
        except Exception as e:
            return None, f"Parameter estimation error: {e}"
    
    def _ridge_fallback(self, recent_data):
        """Fallback method using Ridge regression"""
        try:
            X_prey = np.column_stack([
                recent_data['Prey'].values,
                -recent_data['Prey'].values * recent_data['Predator'].values
            ])
            y_prey = recent_data['dPrey'].values
            
            model_prey = Ridge(alpha=1.0, fit_intercept=False, positive=True)
            model_prey.fit(X_prey, y_prey)
            alpha, beta = np.clip(model_prey.coef_, 0.001, 1.0)
            
            X_pred = np.column_stack([
                recent_data['Prey'].values * recent_data['Predator'].values,
                -recent_data['Predator'].values
            ])
            y_pred = recent_data['dPredator'].values
            
            model_pred = Ridge(alpha=1.0, fit_intercept=False)
            model_pred.fit(X_pred, y_pred)
            delta, gamma = model_pred.coef_
            delta = np.clip(delta, 0.001, 1.0)
            gamma = np.clip(gamma, -0.5, 1.0)
            
            return (alpha, beta, delta, gamma), "Ridge fallback success"
            
        except Exception as e:
            return None, f"Ridge fallback failed: {e}"
    
    def analyze_current_regime(self, data, stock_prices):
        """Analyze current market regime with better trend calculation"""
        if len(data) < 5:
            return None
            
        current = data.iloc[-1]
        recent_5 = data.tail(5)
        
        # Calculate trends with proper handling
        if len(recent_5) >= 3:
            prey_trend = np.polyfit(range(len(recent_5)), recent_5['Prey'].values, 1)[0]
            predator_trend = np.polyfit(range(len(recent_5)), recent_5['Predator'].values, 1)[0]
        else:
            prey_trend = 0
            predator_trend = 0
        
        # Calculate price trend if available
        if len(stock_prices) >= 5:
            price_trend = np.polyfit(range(5), stock_prices.tail(5).values, 1)[0] / stock_prices.iloc[-5] if stock_prices.iloc[-5] != 0 else 0
        else:
            price_trend = 0
        
        analysis = {
            'current_prey': float(current['Prey']),
            'current_predator': float(current['Predator']),
            'prey_trend': float(prey_trend),
            'predator_trend': float(predator_trend),
            'price_trend': float(price_trend),
            'dominance_ratio': float(current['Prey'] / (current['Predator'] + 1e-8)),
        }
        
        return analysis
    
    def generate_signals(self, analysis):
        """Generate trading signals with improved logic"""
        if analysis is None:
            return [], 0
            
        signals = []
        confidence = 0
        cp = analysis['current_prey']
        cpred = analysis['current_predator']
        pt = analysis['prey_trend']
        pdt = analysis['predator_trend']
        dominance = analysis['dominance_ratio']
        
        # Ensure we have reasonable values
        if cp == 0 and cpred == 0:
            return ["⚪ NEUTRAL: Insufficient data"], 0
        
        # Bullish signals
        if cp > 0.7 and pt > 0.01:
            signals.append("🟢 STRONG BULL: Very high buying pressure")
            confidence += 2
        elif cp > 0.6 and pt > 0.005 and cpred < 0.4:
            signals.append("🟢 BULLISH: Strong buying momentum")
            confidence += 1
        elif dominance > 1.5 and cp > cpred:
            signals.append("🟢 BULLISH: Buyers strongly dominating")
            confidence += 1
            
        # Bearish signals
        if cpred > 0.7 and pdt > 0.01:
            signals.append("🔴 STRONG BEAR: Very high selling pressure")
            confidence += 2
        elif cpred > 0.6 and pdt > 0.005 and cp < 0.4:
            signals.append("🔴 BEARISH: Strong selling momentum")
            confidence += 1
        elif dominance < 0.67 and cpred > cp:
            signals.append("🔴 BEARISH: Sellers strongly dominating")
            confidence += 1
            
        # Reversal signals
        if cp < 0.3 and pt > 0.01:
            signals.append("🔵 OVERSOLD: Buying pressure starting to recover")
            confidence += 1
        elif cpred < 0.3 and pdt < -0.01:
            signals.append("🔵 OVERBOUGHT: Selling pressure exhausting")
            confidence += 1
            
        # Neutral case
        if not signals and 0.4 <= cp <= 0.6 and 0.4 <= cpred <= 0.6:
            signals.append("⚪ NEUTRAL: Balanced market forces")
            confidence = 1
            
        return signals, min(max(confidence, 0), 5)
    
    def simulate_future(self, params, current_state, days_forward=5):
        """Simulate future dynamics with stability checks"""
        def lotka_volterra(z, t, alpha, beta, delta, gamma):
            x, y = z
            dxdt = alpha * x - beta * x * y
            dydt = delta * x * y - gamma * y
            return [dxdt, dydt]
        
        t = np.arange(days_forward)
        
        try:
            solution = odeint(lotka_volterra, current_state, t, args=params)
            # Ensure values stay reasonable
            solution = np.clip(solution, 0, 1)
            return solution
        except:
            # Return constant values if simulation fails
            return np.array([current_state] * days_forward)
    
    def analyze_single_stock(self, df, stock_column, lookback_days=200):
        """Complete LV analysis for a single stock - FIXED VERSION"""
        try:
            # Get recent data
            recent_data = df.tail(lookback_days).copy()
            if len(recent_data) < 30:
                return None
            
            # Calculate proxies
            lv_data = self.calculate_improved_proxies(recent_data, stock_column)
            if len(lv_data) < 10:
                return None
            
            # Estimate parameters
            params, message = self.estimate_robust_parameters(lv_data)
            if params is None:
                # Use default parameters if estimation fails
                params = (0.1, 0.1, 0.1, 0.1)
            
            alpha, beta, delta, gamma = params
            
            # Analyze current regime
            stock_prices = recent_data[stock_column]
            analysis = self.analyze_current_regime(lv_data, stock_prices)
            if analysis is None:
                return None
            
            # Generate signals
            signals, confidence = self.generate_signals(analysis)
            
            # Simulate future
            current_state = [analysis['current_prey'], analysis['current_predator']]
            future_path = self.simulate_future(params, current_state)
            
            return {
                'current_prey': analysis['current_prey'],
                'current_predator': analysis['current_predator'],
                'prey_trend': analysis['prey_trend'],
                'predator_trend': analysis['predator_trend'],
                'dominance_ratio': analysis['dominance_ratio'],
                'alpha': alpha,
                'beta': beta,
                'delta': delta,
                'gamma': gamma,
                'signals': signals,
                'confidence': confidence,
                'future_path': future_path,
                'historical_data': lv_data.tail(30)  # Last 30 days of LV data
            }
            
        except Exception as e:
            print(f"LV analysis error for {stock_column}: {e}")
            # Return a default analysis with neutral values
            return {
                'current_prey': 0.5,
                'current_predator': 0.5,
                'prey_trend': 0.0,
                'predator_trend': 0.0,
                'dominance_ratio': 1.0,
                'alpha': 0.1,
                'beta': 0.1,
                'delta': 0.1,
                'gamma': 0.1,
                'signals': ["⚪ NEUTRAL: Default analysis"],
                'confidence': 1,
                'future_path': np.array([[0.5, 0.5]] * 5),
                'historical_data': pd.DataFrame({'Prey': [0.5]*5, 'Predator': [0.5]*5})
            }
    
    def analyze_multiple_stocks(self, df, selected_stocks, lookback_days=200):
        """Analyze multiple stocks with LV model"""
        results = {}
        
        for stock in selected_stocks:
            if stock in df.columns:
                print(f"Analyzing {stock} with LV model...")
                result = self.analyze_single_stock(df, stock, lookback_days)
                if result is not None:
                    results[stock] = result
                    print(f"✓ {stock} analysis completed")
                else:
                    print(f"✗ {stock} analysis failed")
        
        return results