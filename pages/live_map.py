"""
pages/live_map.py — Streamlit page that renders the Leaflet map.

Reads the latest current weather CSV from data/raw/current/,
normalizes columns to match the JavaScript expectations,
injects the data as window.INJECTED_CURRENT_DATA into the
live HTML, and renders it via st.components.v1.html.
"""

import glob
import json
import os

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CURRENT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "current")
_FUTURE_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "future")
_FORECAST_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "forecast")
_LIVE_MAP_HTML = os.path.join(_PROJECT_ROOT, "assets", "live_map.html")
_BORDER_GEOJSON = os.path.join(_PROJECT_ROOT, "assets", "tangerang_border.geojson")

# ---------------------------------------------------------------------------
# Column mapping: api_fetcher PascalCase → JS lowercase keys
# ---------------------------------------------------------------------------
_COLUMN_MAP = {
    "Fetch_Time": "fetch_time",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind_Speed": "wind_speed",
    "Cloud_Cover": "cloud_cover",
    "Pressure": "pressure",
    "Rain": "rain_label",
}


def _find_latest_csv(directory: str) -> str | None:
    """Return the path to the most recently modified CSV in *directory*."""
    pattern = os.path.join(directory, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _clean_value(value):
    """Convert pandas/numpy values to JSON-safe Python primitives."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_border_polygon() -> tuple[dict, list[list[float]]]:
    """Load the geojson and extract the polygon coordinates."""
    with open(_BORDER_GEOJSON, encoding="utf-8") as f:
        geojson = json.load(f)
    polygon = geojson["features"][0]["geometry"]["coordinates"][0]
    return geojson, polygon


def _is_point_in_polygon(lat: float, lon: float, polygon: list[list[float]]) -> bool:
    """Ray-casting algorithm to determine if a point (lat, lon) is inside a polygon."""
    if lat is None or lon is None:
        return False
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]  # [lng, lat]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if lat > min(p1y, p2y):
            if lat <= max(p1y, p2y):
                if lon <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lon <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _build_payload(csv_path: str, polygon: list[list[float]]) -> dict:
    """Read a current-weather CSV, filter by polygon boundary, and return the JS-expected payload dict."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_COLUMN_MAP)

    # Derive 'timestamp' from fetch_time (current data has no separate timestamp)
    if "timestamp" not in df.columns:
        df["timestamp"] = df["fetch_time"]

    fetch_time = (
        df["fetch_time"].dropna().iloc[0]
        if not df["fetch_time"].dropna().empty
        else None
    )

    points = []
    for _, row in df.iterrows():
        lat = _clean_value(row.get("latitude"))
        lon = _clean_value(row.get("longitude"))

        # Limit to points within the Tangerang border polygon
        if not _is_point_in_polygon(lat, lon, polygon):
            continue

        rain_val = row.get("rain_label")
        points.append(
            {
                "latitude": lat,
                "longitude": lon,
                "rain_label": None if pd.isna(rain_val) else int(rain_val),
                "timestamp": _clean_value(row.get("timestamp")),
                "fetch_time": _clean_value(row.get("fetch_time")),
                "temperature": _clean_value(row.get("temperature")),
                "humidity": _clean_value(row.get("humidity")),
                "wind_speed": _clean_value(row.get("wind_speed")),
                "cloud_cover": _clean_value(row.get("cloud_cover")),
                "pressure": _clean_value(row.get("pressure")),
            }
        )

    return {
        "fetch_time": _clean_value(fetch_time),
        "timestamp": _clean_value(fetch_time),
        "points": points,
    }


def _build_historic_payloads(polygon: list[list[float]]) -> dict:
    """Read all CSVs in current directory, sort them, and return grouped payloads."""
    pattern = os.path.join(_CURRENT_DATA_DIR, "*.csv")
    csv_paths = glob.glob(pattern)
    csv_paths.sort()  # Sort chronologically

    # Limit to the last 50 snapshots to keep performance fast and light
    csv_paths = csv_paths[-50:]

    times = []
    data = {}

    for path in csv_paths:
        try:
            payload = _build_payload(path, polygon)
            fetch_time = payload.get("fetch_time")
            if fetch_time:
                times.append(fetch_time)
                data[fetch_time] = payload
        except Exception:
            pass

    return {
        "times": times,
        "data": data,
    }


def _build_forecast_payloads(polygon: list[list[float]]) -> dict:
    """Read the latest future weather CSV and final ONNX forecast predictions, and return payloads."""
    raw_future_csv = _find_latest_csv(_FUTURE_DATA_DIR)
    forecast_csv = _find_latest_csv(_FORECAST_DATA_DIR)

    if not forecast_csv:
        return {"times": [], "data": {}}

    df_forecast = pd.read_csv(forecast_csv)

    # Use Forecast_Target_Time as timestamp
    df_forecast = df_forecast.rename(columns={"Forecast_Target_Time": "timestamp"})

    if raw_future_csv:
        df_raw = pd.read_csv(raw_future_csv)

        # Parse string dates to datetime to bypass formatting differences
        df_raw["dt"] = pd.to_datetime(df_raw["Forecast_Time"])
        df_forecast["dt"] = pd.to_datetime(df_forecast["timestamp"])

        df = pd.merge(
            df_raw, df_forecast, on=["dt", "Latitude", "Longitude"], how="inner"
        )

        # Strip temporary column and map headers
        df = df.drop(columns=["dt"])
        df = df.rename(columns=_COLUMN_MAP)
        df["rain_label"] = df["predicted_rain"]
    else:
        # Fallback if raw future file is not found (still display forecast markers with ML predictions)
        df = df_forecast.rename(columns=_COLUMN_MAP)
        df["rain_label"] = df["predicted_rain"]

        # Populate missing physical weather metrics with None
        for col in ["temperature", "humidity", "wind_speed", "cloud_cover", "pressure"]:
            df[col] = None

    # Sort by timestamp so the times list is chronological
    df = df.sort_values(by="timestamp")

    times = []
    data = {}

    # Extract fetch_time from raw or fallback to timestamp
    fetch_time = None
    if "fetch_time" in df.columns:
        valid_fetch = df["fetch_time"].dropna()
        if not valid_fetch.empty:
            fetch_time = valid_fetch.iloc[0]

    # Group by timestamp (Forecast_Time)
    for timestamp, group in df.groupby("timestamp"):
        points = []
        for _, row in group.iterrows():
            lat = _clean_value(row.get("latitude"))
            lon = _clean_value(row.get("longitude"))

            if not _is_point_in_polygon(lat, lon, polygon):
                continue

            rain_val = row.get("rain_label")
            rain_prob = row.get("rain_probability")
            points.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "rain_label": None if pd.isna(rain_val) else int(rain_val),
                    "rain_probability": None
                    if pd.isna(rain_prob)
                    else float(rain_prob),
                    "timestamp": _clean_value(row.get("timestamp")),
                    "fetch_time": _clean_value(fetch_time),
                    "temperature": _clean_value(row.get("temperature")),
                    "humidity": _clean_value(row.get("humidity")),
                    "wind_speed": _clean_value(row.get("wind_speed")),
                    "cloud_cover": _clean_value(row.get("cloud_cover")),
                    "pressure": _clean_value(row.get("pressure")),
                }
            )

        timestamp_str = str(timestamp)
        times.append(timestamp_str)
        data[timestamp_str] = {
            "fetch_time": _clean_value(fetch_time),
            "timestamp": timestamp_str,
            "points": points,
        }

    return {
        "times": times,
        "data": data,
    }


def _inject_and_render(
    payload: dict, geojson_data: dict, historic_payloads: dict, forecast_payloads: dict
) -> None:
    """Read the live HTML, inject all payloads & border GeoJSON, and render via Streamlit."""
    with open(_LIVE_MAP_HTML, encoding="utf-8") as f:
        html = f.read()

    injection_script = (
        "<script>\n"
        "window.INJECTED_CURRENT_DATA = "
        + json.dumps(payload, ensure_ascii=False)
        + ";\n"
        "window.INJECTED_HISTORIC_DATA = "
        + json.dumps(historic_payloads, ensure_ascii=False)
        + ";\n"
        "window.INJECTED_FORECAST_DATA = "
        + json.dumps(forecast_payloads, ensure_ascii=False)
        + ";\n"
        "window.TANGERANG_BORDER = "
        + json.dumps(geojson_data, ensure_ascii=False)
        + ";\n"
        "</script>\n</head>"
    )
    html = html.replace("</head>", injection_script, 1)

    st.components.v1.html(html, height=700, scrolling=False)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Live Map", page_icon="🗺️", layout="wide")

# Sidebar disclaimer callout
st.sidebar.info(
    "⚠️ **Disclaimer**\n\n"
    "Forecasts are generated using a machine-learning Stacking Ensemble model "
    "(XGBoost, LightGBM, CatBoost) running decoupled ONNX inference on fetched Open-Meteo prediction grids. "
    "These predictions are experimental and should not be used as primary guidance for safety or critical operations."
)

geojson_data, polygon = _load_border_polygon()

csv_path = _find_latest_csv(_CURRENT_DATA_DIR)
historic_payloads = _build_historic_payloads(polygon)
forecast_payloads = _build_forecast_payloads(polygon)

if csv_path is None:
    st.warning(
        "No current weather data found in `data/raw/current/`. "
        "Run the data fetcher first (`python -m src.api_fetcher`)."
    )
    # Still render the map so the basemap tiles are visible
    _inject_and_render(
        {"fetch_time": None, "timestamp": None, "points": []},
        geojson_data,
        historic_payloads,
        forecast_payloads,
    )
else:
    payload = _build_payload(csv_path, polygon)
    _inject_and_render(payload, geojson_data, historic_payloads, forecast_payloads)
