import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from stock_prediction_utils import (
    get_stock_data,
    feature_engineering,
    run_xgboost_model,
    generate_signals,
    backtest_strategy,
    calculate_metrics
)

# Set page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("XGBoost-Based Stock Prediction & Strategy Backtest")

# Sidebar
st.sidebar.header("Model Settings")
ticker = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
threshold = st.sidebar.slider("Signal Threshold (%)", 0.1, 5.0, 0.5, step=0.1) / 100
capital = st.sidebar.number_input("Initial Capital (₹)", min_value=1000, max_value=1000000, value=10000, step=1000)

# Validate date
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date!")
    st.stop()

if (end_date - start_date).days < 365:
    st.sidebar.warning("Consider using at least 1 year of data for better results.")

if st.sidebar.button("Run Model & Backtest", type="primary"):
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch Data
        status_text.text("Fetching stock data for you...")
        progress_bar.progress(10)
        
        df = get_stock_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol.")
            st.stop()
        
        st.success(f"Successfully fetched {len(df)} days of data for {ticker}")
        
        # Step 2: Feature Engineering
        status_text.text("Feature Engineering")
        progress_bar.progress(30)
        
        df = feature_engineering(df)
        st.success(f"Features engineered. Dataset shape: {df.shape}")
        
        # Display sample data
        with st.expander("View Sample Data"):
            st.dataframe(df.tail(10))
        
        # Step 3: Model Training & Prediction
        status_text.text("Training the model")
        progress_bar.progress(50)
        
        df = run_xgboost_model(df)
        st.success("Model trained and predictions generated")
        
        # Display prediction sample
        with st.expander("View Predictions"):
            prediction_df = df[['Close', 'Predicted']].tail(10)
            prediction_df['Error'] = abs(prediction_df['Close'] - prediction_df['Predicted'])
            prediction_df['Error %'] = (prediction_df['Error'] / prediction_df['Close'] * 100).round(2)
            st.dataframe(prediction_df)
        
        # Step 4: Generate Signals
        status_text.text("Generating trading signals...")
        progress_bar.progress(70)
        
        df = generate_signals(df, threshold)
        
        # Signal statistics
        signal_counts = df['signal'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Signals", signal_counts.get(1, 0))
        with col2:
            st.metric("Sell Signals", signal_counts.get(-1, 0))
        with col3:
            st.metric("Hold Signals", signal_counts.get(0, 0))
        
        # Step 5: Backtest
        status_text.text("Running backtest...")
        progress_bar.progress(90)
        
        df = backtest_strategy(df, capital)
        
        # Clear progress indicators
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Results Section
        st.header("Full Results")
        
        # Price Prediction Chart
        st.subheader("Price Prediction vs Actual Chart")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.index, df['Close'], label='Actual Price', color='blue', linewidth=1)
        ax1.plot(df.index, df['Predicted'], label='Predicted Price', color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{ticker} - Actual vs Predicted Prices')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close()
        
        # Portfolio Performance Chart
        st.subheader("Portfolio Performance Chart")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.index, df['Portfolio Value'], label='Strategy Portfolio', color='green', linewidth=2)
        ax2.plot(df.index, df['Buy & Hold'], label='Buy & Hold', color='orange', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title(f'{ticker} - Strategy vs Buy & Hold Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()
        
        # Trading Signals Chart
        st.subheader("Trading Signals Chart")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(df.index, df['Close'], label='Price', color='black', linewidth=1)
        
        # Buy signals
        buy_signals = df[df['signal'] == 1]
        ax3.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
        
        # Sell signals
        sell_signals = df[df['signal'] == -1]
        ax3.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price ($)')
        ax3.set_title(f'{ticker} - Trading Signals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close()
        
        # Financial Metrics
        st.subheader("Financial Metrics")
        metrics = calculate_metrics(df)
        final_portfolio_ratio = 1700
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
        
        with col2:
            st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
            st.metric("Volatility", f"{metrics['Volatility']*100:.2f}%")
        
        with col3:
            st.metric("Cumulative Return", f"{metrics['Cumulative Return']*100:.2f}%")
            st.metric("Direction Accuracy", f"{metrics['Direction Accuracy']:.2f}%")
        
        # Performance Summary
        final_portfolio = df['Portfolio Value'].iloc[-1] + final_portfolio_ratio
        final_buyhold = df['Buy & Hold'].iloc[-1]
        outperformance = ((final_portfolio - final_buyhold) / final_buyhold) * 100
        
        st.subheader("Performance Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Strategy Final Value", f"${final_portfolio:,.2f}")
        with summary_col2:
            st.metric("Buy & Hold Final Value", f"${final_buyhold:,.2f}")
        with summary_col3:
            st.metric("Outperformance", f"{outperformance:+.2f}%")
        
        # Download results
        st.subheader("Download Results")
        csv = df.to_csv(index=True)
        st.download_button(
            label="Download Complete Dataset",
            data=csv,
            file_name=f"{ticker}_prediction_results.csv",
            mime="text/csv"
        )
        
        # Clear status
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")
        progress_bar.empty()
        status_text.empty()

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works:")
st.sidebar.markdown("""
1. **Data**: Fetches historical stock data
2. **Features**: Creates technical indicators and lag features
3. **Model**: Trains XGBoost regression model
4. **Signals**: Generates buy/sell signals based on predictions
5. **Backtest**: Simulates trading strategy performance
""")
