import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        self.tickers = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft', 
            'GOOGL': 'Google',
            '^GSPC': 'SP500'
        }
    
    def download_data(self, start_date="2015-01-01", end_date=None):
        """Download and prepare stock data"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print("Downloading stock data...")
        data = {}
        
        for ticker, name in self.tickers.items():
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                if not stock_data.empty:
                    data[name] = stock_data['Close']
                    print(f"✓ Downloaded {name} ({ticker})")
                else:
                    print(f"✗ Failed to download {name} ({ticker})")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        
        # Create DataFrame
        df = pd.concat(data, axis=1)
        df.columns = df.columns.droplevel(0) if df.columns.nlevels > 1 else df.columns
        df = df.dropna()
        
        # Calculate returns and volatilities
        for stock in df.columns:
            df[f'{stock}_return'] = df[stock].pct_change()
            df[f'{stock}_log_return'] = np.log(df[stock] / df[stock].shift(1))
            df[f'{stock}_volatility_30d'] = df[f'{stock}_log_return'].rolling(window=30).std() * np.sqrt(252)
        
        print(f"✅ Data loaded: {df.shape[0]} days, {df.shape[1]} columns")
        return df

    def calculate_technical_indicators(self, df):
        """Calculate additional technical indicators"""
        for stock in [col for col in df.columns if '_return' not in col and '_volatility' not in col]:
            if stock in ['AAPL', 'MSFT', 'GOOGL', '^GSPC']:
                # Rolling statistics
                df[f'{stock}_rolling_mean_20'] = df[stock].rolling(window=20).mean()
                df[f'{stock}_rolling_std_20'] = df[stock].rolling(window=20).std()
                
                # RSI approximation
                delta = df[stock].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df[f'{stock}_rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def validate_data(self, df):
        """Validate and clean data"""
    # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill missing values (limited)
        df = df.ffill(limit=5)
    
    # Drop remaining NaN values
        df = df.dropna()
    
    # Validate price data
        for stock in ['AAPL', 'MSFT', 'GOOGL', '^GSPC']:
            if stock in df.columns:
            # Ensure prices are positive and reasonable
                if (df[stock] <= 0).any():
                    print(f"Warning: {stock} has non-positive prices")
                if (df[stock] > 1e6).any():  # Unrealistically high prices
                    print(f"Warning: {stock} has unrealistic prices")
    
        return df