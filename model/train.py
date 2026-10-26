import os 
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
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

    feature_columns = ["MA_20", "MA_50", "Volatility", "Returns"]
    X = df[feature_columns]
    y = df["Target"]

    return X,y

def data_split(X, y, test_size = 0.2):
    #Shuffle= False because we want the data intact 
     data_split = train_test_split(X, y, test_size = test_size, shuffle = False)
     return data_split



def train_and_evaluate(X_train, y_train, X_test, y_test):
    print(f"Training model on {len(X_test)} samples, testing on {len(X_test)} samples....")

    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    class_report = classification_report(y_test, prediction)
    print(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")
    print(f"Model's stats on test set:\n{class_report}")
    return model

def save_model(model, path = "model/model.joblib"):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved successfully to {path}")


def main():
    """Execution Workflow"""

    df = load_data()
    df = engineer_features(df)
    X, y = prepare_target_and_data(df)
    X_train, X_test, y_train, y_test = data_split(X, y)
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()