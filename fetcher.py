"""
Fetch Open-Meteo data for a 20x20 Jabodetabek grid.
- current.csv: overwritten with the newest current snapshot
- forecast.csv: overwritten with the newest hourly forecast
- historic.csv: appended with the previous current snapshot before overwrite
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

LAT_MIN, LAT_MAX, LAT_N = -6.36, -6.00, 20
LON_MIN, LON_MAX, LON_N = 106.33, 106.77, 20

LATS = np.linspace(LAT_MIN, LAT_MAX, LAT_N).round(5).tolist()
LONS = np.linspace(LON_MIN, LON_MAX, LON_N).round(5).tolist()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CURRENT_CSV = os.path.join(DATA_DIR, "current.csv")
FORECAST_CSV = os.path.join(DATA_DIR, "forecast.csv")
HISTORIC_CSV = os.path.join(DATA_DIR, "historic.csv")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,surface_pressure,precipitation"
FORECAST_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,surface_pressure,precipitation"

MAX_BATCH_SIZE = 25
MAX_ATTEMPTS = 4
DATA_LAG_DAYS = 1
CURRENT_COLUMN_ORDER = [
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


def build_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "RainMapJabodetabek/1.0",
        "Accept": "application/json",
    })
    return session


def request_open_meteo(session, lats, lons):
    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "hourly": FORECAST_VARS,
        "current": CURRENT_VARS,
        "past_days": DATA_LAG_DAYS,
        "forecast_days": 2,
        "timezone": "Asia/Jakarta",
    }

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(OPEN_METEO_URL, params=params, timeout=45)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            wait_seconds = attempt * 4
            print(f"[fetcher] Request failed, retrying in {wait_seconds}s ({attempt}/{MAX_ATTEMPTS}): {exc}")
            time.sleep(wait_seconds)

    raise last_error


def parse_batch_payload(payload, lats, lons, fetch_time, data_time):
    if isinstance(payload, dict):
        payload = [payload]

    current_rows = []
    forecast_rows = []

    for i, item in enumerate(payload):
        lat = round(lats[i], 5)
        lon = round(lons[i], 5)

        hourly = item.get("hourly", {})
        hourly_times = hourly.get("time", [])
        target_current_time = data_time.strftime("%Y-%m-%dT%H:00")

        if target_current_time not in hourly_times:
            raise RuntimeError(f"Target delayed time {target_current_time} not found in hourly payload")

        current_idx = hourly_times.index(target_current_time)
        current_precip = hourly["precipitation"][current_idx] or 0
        current_rows.append({
            "Fetch_Time": fetch_time,
            "Data_Time": target_current_time.replace("T", " "),
            "Latitude": lat,
            "Longitude": lon,
            "Temperature": hourly["temperature_2m"][current_idx],
            "Humidity": hourly["relative_humidity_2m"][current_idx],
            "Wind_Speed": hourly["wind_speed_10m"][current_idx],
            "Cloud_Cover": hourly["cloud_cover"][current_idx],
            "Pressure": hourly["surface_pressure"][current_idx],
            "Rain": 1 if current_precip > 0.1 else 0,
        })

        forecast_start_idx = current_idx
        forecast_end_idx = min(current_idx + 48, len(hourly_times))
        for j in range(forecast_start_idx, forecast_end_idx):
            target_time = hourly_times[j]
            forecast_precip = hourly["precipitation"][j] or 0
            forecast_rows.append({
                "Fetch_Time": fetch_time,
                "Forecast_Target_Time": target_time,
                "Latitude": lat,
                "Longitude": lon,
                "Temperature": hourly["temperature_2m"][j],
                "Humidity": hourly["relative_humidity_2m"][j],
                "Wind_Speed": hourly["wind_speed_10m"][j],
                "Cloud_Cover": hourly["cloud_cover"][j],
                "Pressure": hourly["surface_pressure"][j],
                "Rain": 1 if forecast_precip > 0.1 else 0,
            })

    return current_rows, forecast_rows


def fetch_resilient(session, lats, lons, fetch_time, data_time):
    try:
        payload = request_open_meteo(session, lats, lons)
        return parse_batch_payload(payload, lats, lons, fetch_time, data_time)
    except requests.RequestException as exc:
        if len(lats) == 1:
            raise RuntimeError(
                f"Failed to fetch coordinate ({lats[0]}, {lons[0]}) after {MAX_ATTEMPTS} attempts"
            ) from exc

        mid = len(lats) // 2
        print(f"[fetcher] Splitting batch of {len(lats)} after repeated failure: {exc}")
        left_current, left_forecast = fetch_resilient(session, lats[:mid], lons[:mid], fetch_time, data_time)
        right_current, right_forecast = fetch_resilient(session, lats[mid:], lons[mid:], fetch_time, data_time)
        return left_current + right_current, left_forecast + right_forecast


def read_existing_current():
    if not os.path.exists(CURRENT_CSV):
        return pd.DataFrame()
    df = pd.read_csv(CURRENT_CSV)
    return ensure_current_schema(df)


def ensure_current_schema(df):
    if df.empty:
        for column in CURRENT_COLUMN_ORDER:
            if column not in df.columns:
                df[column] = pd.Series(dtype="object")
        return df[CURRENT_COLUMN_ORDER]

    if "Data_Time" not in df.columns:
        df["Data_Time"] = df["Fetch_Time"]
    return df[CURRENT_COLUMN_ORDER]


def migrate_historic_schema():
    if not os.path.exists(HISTORIC_CSV):
        return

    df = pd.read_csv(HISTORIC_CSV)
    if "Data_Time" in df.columns:
        return

    ensure_current_schema(df).to_csv(HISTORIC_CSV, index=False)
    print("[fetcher] Migrated historic.csv to include Data_Time")


def append_current_to_historic(df):
    if df.empty:
        return

    migrate_historic_schema()
    if os.path.exists(HISTORIC_CSV):
        ensure_current_schema(df).to_csv(HISTORIC_CSV, mode="a", header=False, index=False)
    else:
        ensure_current_schema(df).to_csv(HISTORIC_CSV, index=False)

    print(f"[fetcher] Appended {len(df)} rows to historic.csv")


def build_coordinate_grid():
    all_lats = [lat for lat in LATS for _ in LONS]
    all_lons = [lon for _ in LATS for lon in LONS]
    return all_lats, all_lons


def run_fetch():
    real_fetch_time = datetime.now()
    delayed_data_time = real_fetch_time - timedelta(days=DATA_LAG_DAYS)
    fetch_time = real_fetch_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[fetcher] Starting fetch at {fetch_time}")
    print(f"[fetcher] Writing delayed data time at {delayed_data_time.strftime('%Y-%m-%d %H:%M:%S')}")

    previous_current = read_existing_current()
    all_lats, all_lons = build_coordinate_grid()
    all_current = []
    all_forecast = []
    session = build_session()

    total_batches = -(-len(all_lats) // MAX_BATCH_SIZE)
    for start in range(0, len(all_lats), MAX_BATCH_SIZE):
        batch_lats = all_lats[start:start + MAX_BATCH_SIZE]
        batch_lons = all_lons[start:start + MAX_BATCH_SIZE]
        print(f"[fetcher] Batch {start // MAX_BATCH_SIZE + 1}/{total_batches} ...")

        current_rows, forecast_rows = fetch_resilient(session, batch_lats, batch_lons, fetch_time, delayed_data_time)
        all_current.extend(current_rows)
        all_forecast.extend(forecast_rows)
        time.sleep(0.75)

    os.makedirs(DATA_DIR, exist_ok=True)
    append_current_to_historic(previous_current)
    pd.DataFrame(all_current)[CURRENT_COLUMN_ORDER].to_csv(CURRENT_CSV, index=False)
    pd.DataFrame(all_forecast).to_csv(FORECAST_CSV, index=False)

    print(f"[fetcher] Done. current={len(all_current)} pts, forecast={len(all_forecast)} rows")


if __name__ == "__main__":
    run_fetch()
