import os
import glob
import time
import gc
from datetime import datetime, timedelta
from src.api_fetcher import APIFetcher
from src.preprocessor import preprocess_future_data
from src.inference import run_onnx_inference


def get_latest_raw_future_file():
    project_root = os.path.dirname(os.path.abspath(__file__))
    future_dir = os.path.join(project_root, "data", "raw", "future")
    files = glob.glob(os.path.join(future_dir, "future_*.csv"))
    if not files:
        return None, None
    latest_file = max(files, key=os.path.getmtime)
    basename = os.path.basename(latest_file)
    timestamp = basename.replace("future_", "").replace(".csv", "")
    return latest_file, timestamp


def run_pipeline_iteration(fetch_future=True):
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

    # Bersihkan objek besar dari memori lokal fungsi
    if "fetcher" in locals():
        del fetcher
    if "df_future" in locals():
        del df_future

    # Paksa Python mengembalikan RAM ke OS secepatnya
    gc.collect()


def sleep_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sleep_seconds = (next_hour - now).total_seconds()
    time.sleep(sleep_seconds + 1.0)


def main():
    print("==================================================")
    print("         TangerangCast Machine Learning Pipeline")
    print("==================================================")

    # Jalankan iterasi awal saat startup
    run_pipeline_iteration(fetch_future=True)

    # Bersihkan sisa RAM setelah proses startup selesai sebelum user masuk ke web
    gc.collect()

    current_timer = 0
    print("\nAuto-pipeline scheduler started. Press CTRL+C to stop.")
    try:
        while True:
            sleep_until_next_hour()
            current_timer += 1

            if current_timer >= 6:
                run_pipeline_iteration(fetch_future=True)
                current_timer = 0
            else:
                run_pipeline_iteration(fetch_future=False)

            # Selalu bersihkan memori di setiap loop tidur
            gc.collect()

    except KeyboardInterrupt:
        print("\nScheduler stopped cleanly.")


if __name__ == "__main__":
    main()
