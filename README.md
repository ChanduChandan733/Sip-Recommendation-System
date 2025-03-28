# Dataset
Nifty 50 intraday dataset contains intraday price data for the Nifty 50 index from the year 2010 to 2023. The Nifty 50 index is India's benchmark stock market index, representing the performance of 50 large-cap Indian stocks across various sectors. The dataset includes minute-by-minute pricing information for each trading day, capturing open, high, low, close prices, as well as trading volume and possibly additional indicators.

Attributes:

Timestamp: Date of the recorded intraday price.
Open: Opening price of the Nifty 50 index at that particular minute.
High: Highest price reached during the minute.
Low: Lowest price reached during the minute.
Close: Closing price at the end of the minute.
Volume: Total trading volume during that minute, where Trading volume is the number of shares of a certain stock that is traded over a given period
Adj Close: The adjusted closing price is attributed to anything that would affect the stock price after the market closes for the day

# data Agreegation
1. The data is already in a daily timeframe (not intraday).
2. do not have timestamps— but having Open ,High , and Close Values  
3. Volume is zero, which might not be useful for trading analysis.

Since the dataset is already in a daily format, we do not need to aggregate intraday data. Instead, we should focus on:
- Handling Missing Values
- Feature Engineering (Adding Technical Indicators like SMA,EMA,RSI, Bollinger Bands, etc.)
- Data Formatting (Ensuring Date column is correctly parsed)
- Checking for Data Issues (e.g., Zero Volume)

We already know that 0.78% values are missing, and we will remove weekends & apply Forward Fill (ffill).

Since we are building a real-time SIP & investment recommendation system, our model needs accurate trading volume data to make informed decisions.
Forward-filling (ffill) or rolling mean may introduce artificial trends that were not originally present., we were dealing with 759 rows of zero  volume. Since we are dropping the Volume column
Though further we wont be needing the Day column we are Dropping it

 The approaches we are following align well with the objectives of:

   Accurate forecasting of market trends and stock movements

We are applying ARIMA, XGBoost, and Clustering (K-Means, DBSCAN) for trend forecasting.
Using technical indicators (SMA, EMA, Bollinger Bands) enhances the model’s ability to detect price patterns.

# Real-time SIP and fund recommendations based on market conditions

Clustering techniques will segment funds based on risk-return profiles, helping users find suitable investments.
XGBoost will help in predictive modeling for stock movements, guiding SIP recommendations.
Volatility and trend analysis (Bollinger Bands & Moving Averages) will influence fund recommendations based on market conditions.

# Feature Engineering Plan (Without Volume Feature)
1️ Trend Indicators (Captures Market Trends)
 i  Simple Moving Average (SMA) – Short & Long-term trends (e.g., 20-day, 50-day)
 ii Exponential Moving Average (EMA) – Reacts quickly to recent price changes

2️ Volatility Indicators (Measures Market Fluctuations)
  i Rolling Standard Deviation – Measures price volatility over time
 ii Bollinger Bands – Identifies overbought/oversold conditions

3️ Momentum Indicators (Identifies Buy/Sell Signals)
  i Relative Strength Index (RSI) – Detects overbought/oversold conditions
 ii MACD (Moving Average Convergence Divergence) – Measures trend strength

4️ Price-Based Features (Captures Historical Price Trends)
  i Lag Features – Use previous closing prices as predictors

##  Machine Learning Models Used
1️ **XGBoost (Optimized):**
   - MAE: 68.76
   - MSE: 10072.22
   - R²: 0.99945
2️ **Hybrid ARIMA + XGBoost:**
   - MAE: 69.75
   - MSE: 10281.23
   - R²: 0.99944
3️ **K-Means Clustering:**
   - Identifies market phases (bullish, bearish, neutral).
4️ **DBSCAN:**
   - Tested but rejected (40% accuracy).

##  Features
. **Real-time Market Analysis** – Used recent Nifty 50 data for accurate predictions.
. **Multi-Model Approach** – Combined **XGBoost, K-Means, and Hybrid ARIMA+XGBoost** for better results.
. **Dynamic Risk Assessment** – Adjusts recommendations based on user’s risk appetite.
. **SHAP-based Feature Importance** – Shows which indicators drive predictions.
. **Deployment on Render** – Provides live investment suggestions via a web app.

##  Results & Insights
- The **Hybrid ARIMA + XGBoost** model provides **accurate predictions**.
- **SHAP analysis** highlights **SMA_20, EMA_20, and RSI_14** as key factors.
- Market trends impact investment recommendations dynamically.

##  Future Enhancements
🔹 Integration **deep learning (LSTM)** for better time-series forecasting.  
🔹 Improvement **real-time market data ingestion**.  
🔹 Implement **automated portfolio balancing**.