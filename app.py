from flask import Flask, jsonify, Response
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Load GRU model 
model = tf.keras.models.load_model("best_lstm_model.h5", compile=False)

#  Load and preprocess dataset 
df = pd.read_csv(r"C:\Users\HP\Desktop\DAM__IND.csv")
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df.set_index('TimeStamp', inplace=True)
df['MCP (Rs/MWh) *'] = df['MCP (Rs/MWh) *'].interpolate()

scaler = MinMaxScaler()
df['MCP_scaled'] = scaler.fit_transform(df[['MCP (Rs/MWh) *']])

# Forecast Function 
def forecast_next_month(df, window_size=24, steps=96*31):
    data = df['MCP_scaled'].values
    last_window = data[-window_size:]
    predictions = []

    std_residual = 0.03
    z_score = 1.645  # for 90% CI

    for _ in range(steps):
        input_seq = last_window[-window_size:].reshape(1, window_size, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        last_window = np.append(last_window, pred)

    pred_scaled = np.array(predictions).reshape(-1, 1)
    forecast = scaler.inverse_transform(pred_scaled).flatten()

    lower = forecast - z_score * std_residual * scaler.data_range_[0]
    upper = forecast + z_score * std_residual * scaler.data_range_[0]

    future_dates = pd.date_range(start="2025-01-01", periods=steps, freq="15min")
    return pd.DataFrame({
        "datetime": future_dates,
        "forecast": forecast,
        "lower_90": lower,
        "upper_90": upper
    })

# Routes

@app.route("/")
def index():
    return " Flask API is running. Go to /forecast for JSON or /plot for chart."

@app.route("/forecast", methods=["GET"])
def get_forecast():
    result_df = forecast_next_month(df)
    return jsonify(result_df.to_dict(orient='records'))

@app.route("/plot", methods=["GET"])
def plot_forecast():
    result_df = forecast_next_month(df)

    plt.figure(figsize=(16, 6))
    plt.plot(result_df['datetime'], result_df['forecast'], label='Forecast (GRU)', color='dodgerblue', linewidth=2)
    plt.fill_between(result_df['datetime'], result_df['lower_90'], result_df['upper_90'],
                     color='lightblue', alpha=0.4, label='90% Confidence Interval')

    plt.title(" MCP Forecast for January 2025 (GRU Model)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=13)
    plt.ylabel("MCP (Rs/MWh)", fontsize=13)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Return as PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plt.close()
    return Response(img.getvalue(), mimetype='image/png')

# Start Server 
if __name__ == "__main__":
    app.run(debug=False)
