import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "Walmart.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")
    return df

@st.cache_data
def run_prophet(df_store, periods=26):
    ts = df_store[["Date", "Weekly_Sales"]].rename(
        columns={"Date": "ds", "Weekly_Sales": "y"}
    )
    train_size = int(len(ts) * 0.8)
    train = ts.iloc[:train_size]
    test = ts.iloc[train_size:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=periods, freq="W")
    forecast = model.predict(future)

    forecast_test = forecast.iloc[-len(test):]
    mae = mean_absolute_error(test["y"], forecast_test["yhat"])
    rmse = np.sqrt(mean_squared_error(test["y"], forecast_test["yhat"]))

    return model, train, test, forecast, mae, rmse

@st.cache_data
def run_sarimax(df_store):
    features = ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    df_model = df_store.set_index("Date")[["Weekly_Sales"] + features]

    split_index = int(len(df_model) * 0.8)
    train_s = df_model.iloc[:split_index]
    test_s = df_model.iloc[split_index:]

    y_train = train_s["Weekly_Sales"]
    y_test = test_s["Weekly_Sales"]
    X_train = train_s[features]
    X_test = test_s[features]

    model = SARIMAX(y_train, exog=X_train, order=(2,1,2), seasonal_order=(1,0,1,52))
    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=len(test_s), exog=X_test)

    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))

    return y_train, y_test, forecast, mae, rmse

def main():
    st.title("📊 Walmart Sales Forecasting App")

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error("⚠ CSV file nahi mili. Ensure Walmart.csv is in same folder.")
        st.stop()

    stores = sorted(df["Store"].unique())
    store_id = st.sidebar.selectbox("Store Select Karo", stores)

    df_store = df[df["Store"] == store_id]

    st.line_chart(df_store[["Date", "Weekly_Sales"]].set_index("Date"))

    model_choice = st.radio("Model Choose Karo:", ["Prophet", "SARIMAX"])

    if model_choice == "Prophet":
        model, train, test, forecast, mae, rmse = run_prophet(df_store)
        fig = model.plot(forecast)
        st.pyplot(fig)
        st.write(f"MAE: {mae}, RMSE: {rmse}")

    else:
        y_train, y_test, forecast_s, mae, rmse = run_sarimax(df_store)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y_train.index, y_train, label="Train")
        ax.plot(y_test.index, y_test, label="Test")
        ax.plot(forecast_s.index, forecast_s, label="Forecast")
        ax.legend()
        st.pyplot(fig)
        st.write(f"MAE: {mae}, RMSE: {rmse}")

if __name__ == "__main__":
    main()
