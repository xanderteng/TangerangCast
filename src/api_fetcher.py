import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import os
import glob


class APIFetcher:
    def __init__(self):
        self.lat_min, self.lat_max = -6.36, -6.00
        self.lon_min, self.lon_max = 106.33, 106.77
        self.lats = np.linspace(self.lat_min, self.lat_max, 20)
        self.lons = np.linspace(self.lon_min, self.lon_max, 20)

        # Define explicit GMT+7 timezone
        self.tz_gmt7 = timezone(timedelta(hours=7))

        self.dirs = {
            "historic": "data/raw/historic",
            "current": "data/raw/current",
            "future": "data/raw/future",
            "temp": "data/raw/temp",
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def _fetch_with_retries(self, url, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)

                if response.status_code == 429:
                    raise requests.exceptions.RequestException(
                        "429 Too Many Requests - Rate Limit Hit"
                    )

                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict) and data.get("error"):
                    raise requests.exceptions.RequestException(
                        f"Open-Meteo Internal Error: {data.get('reason', 'Unknown')}"
                    )

                return data
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    raise e

                sleep_time = 3 * (2**attempt)
                print(
                    f"API Interruption ({e}). Polishing connection. "
                    f"Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)

    def fetch_current_grid(self):
        coords = [
            (round(lat, 5), round(lon, 5)) for lat in self.lats for lon in self.lons
        ]
        records = []
        now = datetime.now(self.tz_gmt7)
        fetch_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M")

        print(f"[{fetch_time}] Fetching current grid data with batching...")

        chunk_size = 50
        for i in range(0, len(coords), chunk_size):
            chunk = coords[i : i + chunk_size]
            lat_str = ",".join([str(c[0]) for c in chunk])
            lon_str = ",".join([str(c[1]) for c in chunk])

            # Added &timezone=Asia%2FJakarta to match GMT+7
            url = (
                f"http://api.open-meteo.com/v1/forecast?"
                f"latitude={lat_str}&longitude={lon_str}"
                f"&current=temperature_2m,relative_humidity_2m,cloud_cover,"
                f"surface_pressure,wind_speed_10m,rain"
                f"&timezone=Asia%2FJakarta"
            )

            try:
                response = self._fetch_with_retries(url)

                if isinstance(response, list):
                    for idx, location_data in enumerate(response):
                        current = location_data.get("current", {})
                        if not current:
                            continue

                        records.append(
                            {
                                "Fetch_Time": fetch_time,
                                "Latitude": chunk[idx][0],
                                "Longitude": chunk[idx][1],
                                "Temperature": current.get("temperature_2m"),
                                "Humidity": current.get("relative_humidity_2m"),
                                "Wind_Speed": current.get("wind_speed_10m"),
                                "Cloud_Cover": current.get("cloud_cover"),
                                "Pressure": current.get("surface_pressure"),
                                "Rain": 1 if current.get("rain", 0) > 0 else 0,
                            }
                        )
                time.sleep(3)
            except Exception as e:
                print(
                    f"Skipping current batch {i // chunk_size + 1} due to persistent error: {e}"
                )

        df = pd.DataFrame(records)
        if df.empty:
            print("Warning: No data fetched. Skipping save.")
            return df

        final_file = f"{self.dirs['current']}/current_{file_timestamp}.csv"
        df.to_csv(final_file, index=False)
        print(f"Saved {len(df)} rows to {final_file}")
        self._cleanup_old_files(self.dirs["current"], keep_last=43800)
        return df

    def fetch_future_grid(self):
        coords = [
            (round(lat, 5), round(lon, 5)) for lat in self.lats for lon in self.lons
        ]
        records = []
        now = datetime.now(self.tz_gmt7)
        fetch_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y%m%d_%H%M")

        print(f"\n[{fetch_time}] Fetching future forecast data with batching...")

        chunk_size = 50
        for i in range(0, len(coords), chunk_size):
            chunk = coords[i : i + chunk_size]
            lat_str = ",".join([str(c[0]) for c in chunk])
            lon_str = ",".join([str(c[1]) for c in chunk])

            # Added &timezone=Asia%2FJakarta to match GMT+7
            url = (
                f"http://api.open-meteo.com/v1/forecast?"
                f"latitude={lat_str}&longitude={lon_str}"
                f"&hourly=temperature_2m,relative_humidity_2m,cloud_cover,"
                f"surface_pressure,wind_speed_10m,rain"
                f"&timezone=Asia%2FJakarta"
            )

            try:
                response = self._fetch_with_retries(url)

                if isinstance(response, list):
                    for idx, location_data in enumerate(response):
                        hourly = location_data.get("hourly", {})
                        if not hourly:
                            continue

                        df_temp = pd.DataFrame(hourly)
                        df_temp["Fetch_Time"] = fetch_time
                        df_temp["Latitude"] = chunk[idx][0]
                        df_temp["Longitude"] = chunk[idx][1]
                        df_temp["Rain"] = df_temp["rain"].apply(
                            lambda x: 1 if x > 0 else 0
                        )

                        records.append(df_temp)
                time.sleep(3)
            except Exception as e:
                print(
                    f"Skipping future batch {i // chunk_size + 1} due to persistent error: {e}"
                )

        df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        if df.empty:
            print("Warning: No future data fetched. Skipping save.")
            return df

        df.rename(
            columns={
                "time": "Forecast_Time",
                "temperature_2m": "Temperature",
                "relative_humidity_2m": "Humidity",
                "surface_pressure": "Pressure",
                "cloud_cover": "Cloud_Cover",
                "wind_speed_10m": "Wind_Speed",
            },
            inplace=True,
        )

        cols = [
            "Fetch_Time",
            "Forecast_Time",
            "Latitude",
            "Longitude",
            "Temperature",
            "Humidity",
            "Wind_Speed",
            "Cloud_Cover",
            "Pressure",
            "Rain",
        ]
        df = df[cols]

        temp_file = f"{self.dirs['temp']}/future_{file_timestamp}.csv"
        df.to_csv(temp_file, index=False)

        final_file = f"{self.dirs['future']}/future_{file_timestamp}.csv"
        os.replace(temp_file, final_file)

        print(f"Saved {len(df)} rows to {final_file}")
        self._cleanup_old_files(self.dirs["future"], keep_last=720)

        try:
            from src.preprocessor import preprocess_future_data

            preprocess_future_data(final_file, file_timestamp)
        except Exception as e:
            print(f"Warning: Failed to preprocess future weather forecast data: {e}")

        return df

    def _cleanup_old_files(self, folder_path, keep_last):
        files = glob.glob(f"{folder_path}/*.csv")
        if len(files) > keep_last:
            files.sort(key=os.path.getctime)
            files_to_delete = files[:-keep_last]
            for f in files_to_delete:
                try:
                    os.remove(f)
                    print(f"Cleaned up old file: {f}")
                except Exception:
                    pass


def run_scheduler():
    fetcher = APIFetcher()
    fetcher.fetch_current_grid()
    fetcher.fetch_future_grid()
    current_timer = 0
    print("Auto-fetch scheduler started. Press CTRL+C to stop.")
    try:
        while True:
            # Sleep until the start of the next hour (minute 00:00) with a small buffer
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )
            sleep_seconds = (next_hour - now).total_seconds()
            time.sleep(sleep_seconds + 1.0)

            current_timer += 1
            fetcher.fetch_current_grid()
            if current_timer >= 6:
                fetcher.fetch_future_grid()
                current_timer = 0
    except KeyboardInterrupt:
        print("Scheduler stopped securely.")


if __name__ == "__main__":
    run_scheduler()
