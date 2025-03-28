# app.py
from flask import Flask, request, render_template
import os
import joblib
import pandas as pd
import numpy as np
import ta  # For technical indicators
import shap  # For XGBoost feature importance

app = Flask(__name__)

# Loading Trained Models
arima_xgb_model = joblib.load("Arima_xgb_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")

# Dataset for analysis
DATA_FILE = "realtime_6months_nifty_data.csv"

# Initialize SHAP Explainer Once (For Performance)
explainer = shap.Explainer(xgb_model)

def get_latest_data():
    """Loads the latest available data from the dataset."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
        df = df.drop_duplicates().sort_values(by='Date')
        return df
    else:
        print("\u274C Error: Data file not found!")
        return None

# Preprocess Data for Model Predictions
def preprocess_data(df):
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["BB_Upper"], df["BB_Lower"] = ta.volatility.bollinger_hband(df["Close"]), ta.volatility.bollinger_lband(df["Close"])
    df.dropna(inplace=True)
    return df[["SMA_20", "EMA_20", "RSI_14", "MACD", "BB_Upper", "BB_Lower"]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    print("Received User Input:", data) 
    investment_amount_min = float(data['investment_amount_min'])
    investment_amount_max = float(data['investment_amount_max'])
    investment_duration = int(data['investment_duration'])
    risk_appetite = data['risk_appetite']
    market_sentiment = data['market_sentiment']

    df = get_latest_data()
    if df is None or df.empty:
        return "\u274C Error: No market data available.", 400
    
    if len(df) < 50:
        return "⚠ Not enough historical data for predictions. Please update dataset.", 400

    df_features = preprocess_data(df)
    if df_features.empty:
        return "\u274C Error: Not enough valid data for predictions.", 400
    
    # **Modify features based on risk and sentiment**
    latest_data = df_features.tail(1).copy()
    
    if risk_appetite == "low":
        latest_data *= 0.9  # Reduce risk exposure
    elif risk_appetite == "high":
        latest_data *= 1.1  # Increase risk exposure
    
    if market_sentiment == "bearish":
        latest_data *= 0.95  # Adjust for negative market trend
    elif market_sentiment == "bullish":
        latest_data *= 1.05  # Adjust for positive market trend

    # **Predictions**
    hybrid_prediction = arima_xgb_model.predict(latest_data)  # Now used for first recommendation
    market_cluster = kmeans_model.predict(latest_data)
    xgb_prediction = xgb_model.predict(latest_data)
    

    # **SHAP Feature Importance (Precomputed)**
    shap_values = explainer(latest_data)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = df_features.columns.tolist()
    feature_impact = {feature_names[i]: round(feature_importance[i], 2) for i in range(len(feature_names))}

    # Calculate recommended investment amount based on user input range and hybrid model prediction
    recommended_investment = (investment_amount_min + investment_amount_max) / 2
    adjusted_recommendation = recommended_investment * (xgb_prediction[0] / 100 + 1)

    # Calculate monthly investment amount based on investment duration
    monthly_investment = ((adjusted_recommendation / investment_duration) / 12)
    
    # Map market cluster to phase description
    market_cluster_map = {
        0: "Recovery Phase",
        1: "Growth Phase",
        2: "Maturity Phase",
        3: "Decline Phase"
    }

    result = {
        'investment_recommendation': round(float(adjusted_recommendation), 2),  # Now based on hybrid model
        'monthly_investment': round(float(monthly_investment), 2),
        'future_price_trend': round(float(hybrid_prediction[0]), 2),
        'market_cluster': int(market_cluster[0]),
        'market_phase': market_cluster_map[int(market_cluster[0])],
        'risk_analysis': risk_appetite,
        'market_sentiment': market_sentiment,
        'feature_impact': feature_impact
    }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
