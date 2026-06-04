import os
import glob
import time
from src.api_fetcher import APIFetcher
from src.preprocessor import preprocess_future_data
from src.inference import run_onnx_inference


def get_latest_raw_future_file():
    """Find the latest raw future weather grid CSV file and extract its timestamp."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    future_dir = os.path.join(project_root, "data", "raw", "future")
    files = glob.glob(os.path.join(future_dir, "future_*.csv"))
    if not files:
        return None, None
    latest_file = max(files, key=os.path.getmtime)
    basename = os.path.basename(latest_file)
    # Extracts timestamp YYYYMMDD_HHMM from future_YYYYMMDD_HHMM.csv
    timestamp = basename.replace("future_", "").replace(".csv", "")
    return latest_file, timestamp


def run_pipeline_iteration(fetch_future=True):
    """Execute one full pass of the data pipeline: fetch -> preprocess -> predict."""
    fetcher = APIFetcher()

    print("\n" + "=" * 50)
    print("[PIPELINE] 1. Fetching current grid weather data...")
    try:
        fetcher.fetch_current_grid()
    except Exception as e:
        print(f"[ERROR] Failed to fetch current grid data: {e}")

    if fetch_future:
        print("[PIPELINE] 2. Fetching future grid forecast data...")
        try:
            df_future = fetcher.fetch_future_grid()
        except Exception as e:
            print(f"[ERROR] Failed to fetch future grid data: {e}")
            df_future = None

        if df_future is not None and not df_future.empty:
            raw_file, timestamp = get_latest_raw_future_file()
            if raw_file and timestamp:
                print(f"[PIPELINE] 3. Running Preprocessor on {raw_file}...")
                try:
                    processed_file = preprocess_future_data(raw_file, timestamp)
                except Exception as e:
                    print(f"[ERROR] Preprocessing failed: {e}")
                    processed_file = None

                if processed_file:
                    print(
                        f"[PIPELINE] 4. Running ONNX Stacking Ensemble Inference on {processed_file}..."
                    )
                    try:
                        run_onnx_inference(processed_file, timestamp)
                        print("[PIPELINE] Pipeline iteration complete!")
                    except Exception as e:
                        print(f"[ERROR] Inference failed: {e}")
            else:
                print(
                    "[WARNING] Could not determine the latest raw future file. Skipping preprocessing & inference."
                )
        else:
            print("[WARNING] Future grid fetching skipped or returned no data.")
    else:
        print("[PIPELINE] Skipping future grid fetch and ML inference for this hour.")


def sleep_until_next_hour():
    """Sleep until the start of the next hour (minute 00:00) with a small buffer."""
    from datetime import datetime, timedelta
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sleep_seconds = (next_hour - now).total_seconds()
    # Add a 1-second buffer to ensure we wake up after the hour mark starts
    time.sleep(sleep_seconds + 1.0)


def main():
    print("==================================================")
    print("         TangerangCast Machine Learning Pipeline")
    print("==================================================")

    # Run a full iteration (including future predictions) immediately on startup
    run_pipeline_iteration(fetch_future=True)

    current_timer = 0
    print("\nAuto-pipeline scheduler started. Press CTRL+C to stop.")
    try:
        while True:
            # Sleep until the next hour mark (00:00)
            sleep_until_next_hour()
            current_timer += 1

            # Fetch current grid hourly, and execute future ML forecasts every 6 hours
            if current_timer >= 6:
                run_pipeline_iteration(fetch_future=True)
                current_timer = 0
            else:
                run_pipeline_iteration(fetch_future=False)

    except KeyboardInterrupt:
        print("\nScheduler stopped cleanly.")


if __name__ == "__main__":
    main()
