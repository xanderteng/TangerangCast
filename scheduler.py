"""
Run the Open-Meteo shift-and-update pipeline every day at 00:00 Asia/Jakarta.

Keep this process running if you want automatic daily updates:
    python scheduler.py

For a one-time immediate fetch:
    python scheduler.py --run-now
"""

import argparse
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fetcher import run_fetch


JAKARTA_TZ = ZoneInfo("Asia/Jakarta")


def next_midnight(now=None):
    now = now or datetime.now(JAKARTA_TZ)
    tomorrow = now.date() + timedelta(days=1)
    return datetime.combine(tomorrow, datetime.min.time(), tzinfo=JAKARTA_TZ)


def run_daily_midnight():
    while True:
        target = next_midnight()
        sleep_seconds = max(1, (target - datetime.now(JAKARTA_TZ)).total_seconds())
        print(f"[scheduler] Next fetch at {target:%Y-%m-%d %H:%M:%S %Z}")
        time.sleep(sleep_seconds)
        try:
            run_fetch()
        except Exception as exc:
            print(f"[scheduler] Fetch failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now", action="store_true", help="Run one fetch immediately and exit.")
    args = parser.parse_args()

    if args.run_now:
        run_fetch()
    else:
        run_daily_midnight()
