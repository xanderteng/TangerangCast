import os

import pandas as pd
from flask import Flask, jsonify, render_template

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CURRENT_CSV = os.path.join(DATA_DIR, "current.csv")
FORECAST_CSV = os.path.join(DATA_DIR, "forecast.csv")
HISTORIC_CSV = os.path.join(DATA_DIR, "historic.csv")

CURRENT_COLUMNS = [
    "Fetch_Time",
    "Data_Time",
    "Latitude",
    "Longitude",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Cloud_Cover",
    "Pressure",
    "Rain",
]

FORECAST_COLUMNS = [
    "Fetch_Time",
    "Forecast_Target_Time",
    "Latitude",
    "Longitude",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Cloud_Cover",
    "Pressure",
    "Rain",
]

DATASETS = {
    "current": {
        "path": CURRENT_CSV,
        "timestamp": "Data_Time",
        "fallback_timestamp": "Fetch_Time",
        "columns": CURRENT_COLUMNS,
    },
    "forecast": {
        "path": FORECAST_CSV,
        "timestamp": "Forecast_Target_Time",
        "fallback_timestamp": None,
        "columns": FORECAST_COLUMNS,
    },
    "historic": {
        "path": HISTORIC_CSV,
        "timestamp": "Data_Time",
        "fallback_timestamp": "Fetch_Time",
        "columns": CURRENT_COLUMNS,
    },
}


def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def pick_timestamp_column(df, config):
    if config["timestamp"] in df.columns:
        return config["timestamp"]
    fallback = config.get("fallback_timestamp")
    if fallback and fallback in df.columns:
        return fallback
    return config["timestamp"]


def unique_value(df, column):
    if column not in df.columns or df.empty:
        return None
    values = df[column].dropna().astype(str).unique().tolist()
    return values[0] if len(values) == 1 else None


def normalize_weather_rows(df, timestamp_column):
    points = []
    for _, row in df.iterrows():
        rain_value = row.get("Rain")
        rain_label = None if pd.isna(rain_value) else int(rain_value)
        points.append({
            "latitude": clean_value(row.get("Latitude")),
            "longitude": clean_value(row.get("Longitude")),
            "rain_label": rain_label,
            "timestamp": clean_value(row.get(timestamp_column)),
            "fetch_time": clean_value(row.get("Fetch_Time")),
            "temperature": clean_value(row.get("Temperature")),
            "humidity": clean_value(row.get("Humidity")),
            "wind_speed": clean_value(row.get("Wind_Speed")),
            "cloud_cover": clean_value(row.get("Cloud_Cover")),
            "pressure": clean_value(row.get("Pressure")),
        })
    return points


def load_dataset(dataset_name, target_time=None):
    config = DATASETS[dataset_name]
    csv_path = config["path"]
    if not os.path.exists(csv_path):
        return {
            "mode": dataset_name,
            "fetch_time": None,
            "timestamp": None,
            "points": [],
        }

    df = pd.read_csv(csv_path, usecols=lambda col: col in config["columns"] or col == "Fetch_Time")
    timestamp_column = pick_timestamp_column(df, config)

    if target_time and timestamp_column in df.columns:
        df = df[df[timestamp_column].astype(str) == target_time]

    return {
        "mode": dataset_name,
        "fetch_time": unique_value(df, "Fetch_Time"),
        "timestamp": unique_value(df, timestamp_column),
        "points": normalize_weather_rows(df, timestamp_column),
    }


def load_dataset_times(dataset_name):
    config = DATASETS[dataset_name]
    csv_path = config["path"]
    if not os.path.exists(csv_path):
        return []

    df = pd.read_csv(csv_path, usecols=lambda col: col in {config["timestamp"], config.get("fallback_timestamp")})
    timestamp_column = pick_timestamp_column(df, config)
    return sorted(df[timestamp_column].dropna().astype(str).unique().tolist())


@app.route("/")
def index():
    return render_template("map.html")


@app.route("/api/current")
def api_current():
    return jsonify(load_dataset("current"))


@app.route("/api/forecast/times")
def api_forecast_times():
    return jsonify({"times": load_dataset_times("forecast")})


@app.route("/api/forecast/<path:target_time>")
def api_forecast(target_time):
    data = load_dataset("forecast", target_time)
    data["target_time"] = target_time
    return jsonify(data)


@app.route("/api/historic")
def api_historic():
    return jsonify(load_dataset("historic"))


@app.route("/api/historic/times")
def api_historic_times():
    return jsonify({"times": load_dataset_times("historic")})


@app.route("/api/historic/<path:target_time>")
def api_historic_at(target_time):
    data = load_dataset("historic", target_time)
    data["target_time"] = target_time
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
