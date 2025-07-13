# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 13:28:07 2025

@author: Hemal
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import uuid

# Function to calculate Klinger Volume Oscillator using pandas-ta
def klinger_oscillator(data, fast=34, slow=55, signal=13):
    kvo = ta.kvo(data['High'], data['Low'], data['Close'], data['Volume'], fast=fast, slow=slow, signal=signal)
    data['KVO'] = kvo[f'KVO_{fast}_{slow}_{signal}']
    data['KVO_Signal'] = kvo[f'KVOs_{fast}_{slow}_{signal}']
    return data['KVO'], data['KVO_Signal']

# Function to calculate Bollinger Bands %B using pandas-ta
def bollinger_b_percent(data, length=20, std=2):
    bbands = ta.bbands(data['Close'], length=length, std=std)
    bb_upper_col = [col for col in bbands.columns if col.startswith('BBU')][0]
    bb_lower_col = [col for col in bbands.columns if col.startswith('BBL')][0]
    data['BB_Upper'] = bbands[bb_upper_col]
    data['BB_Lower'] = bbands[bb_lower_col]
    data['BB_Percent'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    return data['BB_Percent']

# Function to detect Volume Spike
def volume_spike(data, lookback=20, threshold=2):
    data['Vol_MA'] = data['Volume'].rolling(window=lookback).mean()
    data['Vol_Std'] = data['Volume'].rolling(window=lookback).std()
    data['Vol_Spike'] = (data['Volume'] - data['Vol_MA']) / data['Vol_Std']
    return np.where(data['Vol_Spike'] > threshold, 1, 0)

# Function to create labels based on future returns
def create_labels(data, look_ahead=5, threshold=0.01):
    data['Future_Return'] = data['Close'].pct_change(look_ahead).shift(-look_ahead)
    labels = np.where(data['Future_Return'] > threshold, 1, np.where(data['Future_Return'] < -threshold, -1, 0))
    return labels

# Function to calculate trading performance metrics
def calculate_trading_metrics(data, predictions, look_ahead=5):
    data = data.copy()
    data['Predicted_Signal'] = predictions
    data['Future_Close'] = data['Close'].shift(-look_ahead)
    data['Trade_Return'] = np.where(
        data['Predicted_Signal'] == 1, (data['Future_Close'] - data['Close']) / data['Close'],
        np.where(data['Predicted_Signal'] == -1, (data['Close'] - data['Future_Close']) / data['Close'], 0)
    )
    
    cum_returns = (1 + data['Trade_Return']).cumprod().iloc[-1] - 1 if len(data['Trade_Return']) > 0 else 0
    returns = data['Trade_Return'].dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0
    cumprod = (1 + returns).cumprod()
    peak = cumprod.cummax()
    drawdown = (cumprod - peak) / peak
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    trades = returns[returns != 0]
    win_rate = len(trades[trades > 0]) / len(trades) if len(trades) > 0 else 0
    
    return {
        'Cumulative_Returns': cum_returns,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Win_Rate': win_rate
    }

# Function to generate signals and evaluate performance
def generate_signals(tickers, timeframes, start_date, end_date, look_ahead=5, retries=3):
    signal_results = []
    metrics_results = []
    
    for ticker in tickers:
        for timeframe in timeframes:
            st.write(f"Processing {ticker} on {timeframe} timeframe...")
            for attempt in range(retries):
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=start_date, end=end_date, interval=timeframe, threads=False)
                    if data.empty:
                        st.warning(f"No data for {ticker} on {timeframe}")
                        break
                    break  # Exit retry loop on success
                except Exception as e:
                    if attempt < retries - 1:
                        st.warning(f"Retry {attempt + 1} for {ticker} on {timeframe}: {e}")
                        continue
                    else:
                        st.error(f"Error fetching data for {ticker} on {timeframe}: {e}")
                        break
            else:
                continue  # Skip to next ticker if all retries fail
            
            data['KVO'], data['KVO_Signal'] = klinger_oscillator(data)
            data['BB_Percent'] = bollinger_b_percent(data)
            data['Vol_Spike'] = volume_spike(data)
            data['Signal'] = create_labels(data, look_ahead=look_ahead)
            
            features = ['KVO', 'KVO_Signal', 'BB_Percent', 'Vol_Spike']
            X = data[features].dropna()
            y = data['Signal'].loc[X.index]
            
            valid_idx = X.index.intersection(y.index)
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
            if len(X) < 50:
                st.warning(f"Insufficient data for {ticker} on {timeframe}")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            predictions = model.predict(scaler.transform(X))
            test_predictions = model.predict(X_test_scaled)
            
            class_report = classification_report(y_test, test_predictions, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_test, test_predictions)
            
            trading_metrics = calculate_trading_metrics(data.loc[X.index], predictions, look_ahead)
            
            result = data.loc[X.index].copy()
            result['Predicted_Signal'] = predictions
            result['Ticker'] = ticker
            result['Timeframe'] = timeframe
            signal_results.append(result[['Ticker', 'Timeframe', 'Close', 'KVO', 'KVO_Signal', 'BB_Percent', 'Vol_Spike', 'Signal', 'Predicted_Signal']])
            
            metrics = {
                'Ticker': ticker,
                'Timeframe': timeframe,
                'Accuracy': class_report['accuracy'],
                'Precision_Buy': class_report['1']['precision'],
                'Recall_Buy': class_report['1']['recall'],
                'F1_Buy': class_report['1']['f1-score'],
                'Precision_Sell': class_report['-1']['precision'],
                'Recall_Sell': class_report['-1']['recall'],
                'F1_Sell': class_report['-1']['f1-score'],
                'Precision_Hold': class_report['0']['precision'],
                'Recall_Hold': class_report['0']['recall'],
                'F1_Hold': class_report['0']['f1-score'],
                'Confusion_Matrix': str(conf_matrix.tolist()),
                'Cumulative_Returns': trading_metrics['Cumulative_Returns'],
                'Sharpe_Ratio': trading_metrics['Sharpe_Ratio'],
                'Max_Drawdown': trading_metrics['Max_Drawdown'],
                'Win_Rate': trading_metrics['Win_Rate']
            }
            metrics_results.append(metrics)
    
    return signal_results, metrics_results

# Streamlit app
st.title("Stock Trading Signal Generator")

# User inputs
st.header("Input Parameters")
tickers_input = st.text_input("Enter Tickers (comma-separated, e.g., ^NSEI,^NSEBANK)", "^NSEI,^NSEBANK,CNXFINANCE.NS,^BSESN")
timeframes = st.multiselect("Select Timeframes", ["1d", "1h", "15m"], default=["1d"])
start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.now())
look_ahead = st.slider("Look Ahead Periods", 1, 20, 5)

# Convert inputs
tickers = [ticker.strip() for ticker in tickers_input.split(",")]

# Run analysis
if st.button("Generate Signals"):
    with st.spinner("Generating signals..."):
        signal_results, metrics_results = generate_signals(tickers, timeframes, start_date, end_date, look_ahead)
    
    # Display results
    if signal_results:
        signals_df = pd.concat(signal_results)
        st.header("Trading Signals")
        st.dataframe(signals_df)
        
        # Download signals
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signals_file = f"trading_signals_{timestamp}.csv"
        signals_df.to_csv(signals_file)
        with open(signals_file, "rb") as file:
            st.download_button("Download Signals CSV", file, signals_file)
        
        # Plot signals for each ticker and timeframe
        for ticker in tickers:
            for timeframe in timeframes:
                df = signals_df[(signals_df['Ticker'] == ticker) & (signals_df['Timeframe'] == timeframe)]
                if not df.empty:
                    st.subheader(f"{ticker} - {timeframe}")
                    
                    # Price and signal plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')))
                    buy_signals = df[df['Predicted_Signal'] == 1]
                    sell_signals = df[df['Predicted_Signal'] == -1]
                    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], name='Buy Signal', mode='markers', marker=dict(symbol='triangle-up', size=10, color='green')))
                    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], name='Sell Signal', mode='markers', marker=dict(symbol='triangle-down', size=10, color='red')))
                    fig.update_layout(title=f"{ticker} {timeframe} - Price and Signals", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig)
    
    if metrics_results:
        metrics_df = pd.DataFrame(metrics_results)
        st.header("Performance Metrics")
        st.dataframe(metrics_df)
        
        # Download metrics
        metrics_file = f"performance_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        with open(metrics_file, "rb") as file:
            st.download_button("Download Metrics CSV", file, metrics_file)
        
        # Plot performance metrics
        st.subheader("Performance Metrics Visualization")
        fig = px.bar(metrics_df, x='Ticker', y=['Cumulative_Returns', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate'],
                     facet_col='Timeframe', barmode='group', title="Performance Metrics by Ticker and Timeframe")
        st.plotly_chart(fig)
    
    if not signal_results and not metrics_results:
        st.error("No signals or metrics generated. Please check inputs and try again.")
