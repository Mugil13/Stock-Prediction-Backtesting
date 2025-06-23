# Stock Price Prediction & Backtesting with Streamlit deployment

This project is a time-series stock prediction app that forecasts next-day closing prices, generates Buy/Sell signals, and evaluates the trading strategy using standard finance metrics. Built with Jupyter Notebook + Streamlit.

---

## What This Project Covers

The Jupyter notebook, 'Stock_Price_Prediction.ipynb' contains:
- the dataset, historical stock data pulled using `yfinance`
- Engineered features - SMA, EMA, RSI, MACD, lag features
- Trained and evaluated 3 models:
  - **ARIMA** – Statistical baseline
  - **LSTM** – Deep learning model 
  - **XGBoost** – Deployment model

```
NOTE:
Even though LSTM gave the best results, it has not been used for deplyment because it has TensorFlow issues with Python 3.12+. So, for this project to work for all environments, I have selected XGBoost, which gave the second best results
```

---

## Strategy Backtest Results

This for:
- **Stock Symbol:** AAPL
- **Date Range:** 2022/01/01 to 2023/12/31
- **Initial Capital:** ₹10,000
- **Signal Threshold:** 0.10%

| Metric              | Value        |
|---------------------|--------------|
| **Sharpe Ratio**    | 1.41         |
| **Sortino Ratio**   | 1.89         |
| **Max Drawdown**    | -15.69%      |
| **Volatility**      | 16.33%       |
| **Cumulative Return** | 27.71%    |
| **Direction Accuracy** | 62.07%   |
| **Final Strategy Value** | ₹10,971.19 |
| **Buy & Hold Value** | ₹10,824.54 |
| **Outperformance**   | +1.35%      |

---
## Finance Metrics Explained

This project evaluates the strategy using key trading metrics:

- Sharpe Ratio – Return per unit volatility
- Max Drawdown – Worst historical loss
- Sortino Ratio – Return per downside risk
- Cumulative Return – Net portfolio gain
- Volatility – Risk due to fluctuations
- Direction Accuracy – % of correct trend signals

---

## Run Locally

```
git clone https://github.com/Mugil13/Stock-Prediction-Backtesting.git
cd Stock-Prediction-Backtesting
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will open in your browser at: http://localhost:8501

An interface will pop up which allows you to:
- Select stock & date range
- View actual vs predicted prices
- See backtesting strategy plots
- Explore the evaluation metrics used
