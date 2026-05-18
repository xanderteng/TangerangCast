# RainMap Jabodetabek

Interactive weather map built with Flask, Leaflet, and Open-Meteo.

## Run order

Install dependencies first:

```bash
pip install -r requirements.txt
```

If you want fresh data before opening the app, run one fetch:

```bash
py scheduler.py --run-now
```

That pipeline will:

1. Read the existing `data/current.csv`
2. Append the previous current snapshot into `data/historic.csv`
3. Fetch fresh Open-Meteo data
4. Overwrite `data/current.csv` and `data/forecast.csv`

Then start the web app:

```bash
py app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Scheduler

For automatic daily fetching at `00:00` Asia/Jakarta:

```bash
py scheduler.py
```

Keep that process running. If one fetch fails, the scheduler now stays alive and will try again on the next midnight.

## D-1 behavior

The dataset is intentionally stored as `D-1`:

- `Fetch_Time`: the real moment the API call ran
- `Data_Time`: the delayed data timestamp used by `Current` and `Historic`
- `Forecast_Target_Time`: delayed forecast timeline shown in `Forecast`

## Map modes

- `Current`: delayed current snapshot
- `Forecast`: delayed forecast timeline with arrows and direct time selector
- `Historic`: archived previous current snapshots with time selector

## Map styles

- `Dark`
- `Light`
- `Colorful`

## API Endpoints

| Endpoint                   | Description                              |
| -------------------------- | ---------------------------------------- |
| `GET /api/current`         | Latest current dataset                   |
| `GET /api/forecast/times`  | Available forecast timestamps            |
| `GET /api/forecast/<time>` | Forecast points for a selected timestamp |
| `GET /api/historic/times`  | Available historic timestamps            |
| `GET /api/historic/<time>` | Historic points for a selected timestamp |

## Main files

- [app.py](<C:\Users\User\Documents\College Files\Semester 4\MachineLearning\Experimen\app.py>): Flask app and API
- [fetcher.py](<C:\Users\User\Documents\College Files\Semester 4\MachineLearning\Experimen\fetcher.py>): fetch pipeline and CSV writing
- [scheduler.py](<C:\Users\User\Documents\College Files\Semester 4\MachineLearning\Experimen\scheduler.py>): midnight scheduler
