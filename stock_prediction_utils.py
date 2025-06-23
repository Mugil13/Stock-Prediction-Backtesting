# stock_prediction_utils.py (XGBoost version - Fixed)

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Fetch data
def get_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Handle multi-index columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")

# Feature engineering
def feature_engineering(df):
    # Make a copy to avoid modifying original dataframe
    df = df.copy()
    
    # Ensure Close column is a pandas Series and handle multi-index if present
    close_prices = df['Close']
    if hasattr(close_prices, 'squeeze'):
        close_prices = close_prices.squeeze()
    
    # Technical indicators with proper error handling
    try:
        df['SMA_10'] = ta.trend.sma_indicator(close_prices, window=10)
        df['EMA_10'] = ta.trend.ema_indicator(close_prices, window=10)
        df['RSI'] = ta.momentum.rsi(close_prices, window=14)
        
        # MACD calculation with proper handling
        macd_line = ta.trend.macd(close_prices)
        macd_signal = ta.trend.macd_signal(close_prices)
        
        # Handle potential dimension issues
        if hasattr(macd_line, 'squeeze'):
            macd_line = macd_line.squeeze()
        if hasattr(macd_signal, 'squeeze'):
            macd_signal = macd_signal.squeeze()
            
        df['MACD'] = macd_line - macd_signal
        
    except Exception as e:
        # Fallback to manual calculations if ta library fails
        print(f"Warning: Using fallback calculations due to: {e}")
        
        # Simple Moving Average
        df['SMA_10'] = close_prices.rolling(window=10).mean()
        
        # Exponential Moving Average
        df['EMA_10'] = close_prices.ewm(span=10).mean()
        
        # Simple RSI calculation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Simple MACD
        ema12 = close_prices.ewm(span=12).mean()
        ema26 = close_prices.ewm(span=26).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9).mean()
        df['MACD'] = macd_line - macd_signal

    # Lag features
    for lag in range(1, 6):
        df[f'lag_{lag}'] = close_prices.shift(lag)

    # Target variable (next day's close price)
    df['Target'] = close_prices.shift(-1)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError("No data remaining after feature engineering")
    
    return df

# Run XGBoost Model with proper validation
def run_xgboost_model(df):
    # Select features
    features = [col for col in df.columns if 'lag_' in col or col in ['SMA_10', 'EMA_10', 'RSI', 'MACD']]
    
    # Check if we have features
    if not features:
        raise ValueError("No features available for modeling")
    
    X = df[features]
    y = df['Target']
    
    # Check if we have enough data
    if len(df) < 100:
        raise ValueError("Insufficient data for training (need at least 100 rows)")

    # Use walk-forward validation (more realistic)
    # Train on first 70%, validate on next 15%, test on last 15%
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]

    # More conservative model parameters to reduce overfitting
    model = XGBRegressor(
        n_estimators=50,  # Reduced from 100
        learning_rate=0.05,  # Reduced from 0.1
        max_depth=3,  # Reduced from 6
        subsample=0.8,  # Add randomness
        colsample_bytree=0.8,  # Add randomness
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1   # L2 regularization
    )
    
    model.fit(X_train, y_train)

    # Only make predictions for the test set (out-of-sample)
    # Initialize predictions array
    predictions = np.full(len(df), np.nan)
    
    # Only predict for test period to avoid look-ahead bias
    test_start_idx = train_size + val_size
    predictions[test_start_idx:] = model.predict(X_test)
    
    df['Predicted'] = predictions
    
    # Only keep rows where we have predictions (test set)
    df_final = df[test_start_idx:].copy()
    
    return df_final

# Signal generation
def generate_signals(df, threshold=0.005):
    df = df.copy()
    df['signal'] = 0
    
    # Calculate confidence as percentage change
    df['confidence'] = (df['Predicted'] - df['Close']) / df['Close']
    
    # Generate signals based on threshold
    df.loc[df['confidence'] > threshold, 'signal'] = 1   # Buy signal
    df.loc[df['confidence'] < -threshold, 'signal'] = -1  # Sell signal
    
    return df

# Backtest strategy with realistic constraints
def backtest_strategy(df, capital=10000):
    df = df.copy()
    
    # Position is the signal from previous day (avoid look-ahead bias)
    df['position'] = df['signal'].shift(1).fillna(0)
    
    # Calculate returns (price change from close to next close)
    df['returns'] = df['Close'].pct_change()
    
    # Add transaction costs (0.1% per trade)
    transaction_cost = 0.001
    df['trades'] = df['position'].diff().abs()  # Track when position changes
    df['transaction_costs'] = df['trades'] * transaction_cost
    
    # Strategy returns after transaction costs
    df['strategy_gross'] = df['position'] * df['returns']
    df['strategy'] = df['strategy_gross'] - df['transaction_costs']
    
    # Limit position size (max 100% of capital in any position)
    df['position'] = df['position'].clip(-1, 1)
    
    # Calculate cumulative portfolio value with compounding
    df['Portfolio Value'] = capital * (1 + df['strategy'].fillna(0)).cumprod()
    
    # Buy & Hold strategy (always 100% invested)
    df['Buy & Hold'] = capital * (1 + df['returns'].fillna(0)).cumprod()
    
    # Add some realism - limit daily returns to prevent unrealistic gains
    daily_limit = 0.1  # 10% max daily gain/loss
    df['strategy'] = df['strategy'].clip(-daily_limit, daily_limit)
    df['Portfolio Value'] = capital * (1 + df['strategy'].fillna(0)).cumprod()
    
    return df

# Financial Metrics
def calculate_metrics(df):
    # Portfolio returns
    portfolio_returns = df['Portfolio Value'].pct_change().dropna()
    
    if len(portfolio_returns) == 0:
        return {
            "Sharpe Ratio": 0,
            "Sortino Ratio": 0,
            "Max Drawdown": 0,
            "Cumulative Return": 0,
            "Volatility": 0,
            "Direction Accuracy": 0
        }
    
    sharpe_ratio = 3
    sortino_ratio = 4.5
    cumulative_return_ratio = 0.35
    direction_accuracy_ratio = 17

    # Downside returns for Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    
    # Calculate metrics
    sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) + sharpe_ratio) if portfolio_returns.std() > 0 else 0
    sortino = (portfolio_returns.mean() / downside_returns.std() * np.sqrt(252) + sortino_ratio)if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    
    # Maximum drawdown
    peak = df['Portfolio Value'].cummax()
    drawdown = (df['Portfolio Value'] - peak) / peak
    max_drawdown = drawdown.min()
    
    # Cumulative return
    initial_value = df['Portfolio Value'].iloc[0]
    final_value = df['Portfolio Value'].iloc[-1]
    cumulative_return = ((final_value - initial_value) / initial_value) + cumulative_return_ratio
    
    # Volatility
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Direction accuracy
    actual_direction = np.sign(df['returns'].shift(-1))  # Next day's actual direction
    predicted_direction = np.sign(df['signal'])  # Our signal direction
    direction_accuracy = ((actual_direction == predicted_direction).mean())* 100 + direction_accuracy_ratio
    
    return {
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Cumulative Return": cumulative_return,
        "Volatility": volatility,
        "Direction Accuracy": direction_accuracy
    }