import os 
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import yfinance as yf


def load_data(ticker="AAPL", start_date = "2020-01-01", end_date = "2026-01-01"):

    #Step 1: Download stock market data from Yahoo Finance
    print(f"Downdloading data for {ticker}....")
    df = yf.download(ticker, start=start_date, end=end_date)

    return df

print(load_data(ticker="AAPL", start_date = "2020-01-01", end_date = "2026-01-01").head(10))


def engineer_features(df):

    #Step 2: Transform raw prices into stationary technical indicators

    #Calculate the 20d and 50d Simple Moving Averages (SMA)
    df["Returns"] = df["Close"].pct_change()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    #Volatility --> 20day window of rolling standard deviation from Daily Returns
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    return df

def prepare_target_and_data(df):
    #Step 3. Target Definition & Data Cleaning

    #Binary direction target and drops resulting null values
    #Today's price - Tomorrow's price 
    # if todays > tomorrow = True (1) | today's < tomorrow's = False (0)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace= True)

    feature_columns = ["MA20", "MA_50", "Volatility", "Returns"]
    X = df["feature_columns"]
    y = df["Target"]

    return X,y
