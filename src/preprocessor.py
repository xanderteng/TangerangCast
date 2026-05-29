import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, PowerTransformer

_SCALER = None
_PT = None

HISTORIC_LOCATIONS = [
    (-6.2226, 106.6533, 0),  # Alam_Sutera
    (-6.3006, 106.6538, 1),  # BSD_City
    (-6.1557, 106.6579, 2),  # Batuceper
    (-6.2738, 106.7136, 3),  # Bintaro
    (-6.2427, 106.5173, 4),  # Cikupa
    (-6.2416, 106.6285, 5),  # Gading_Serpong
    (-6.2269, 106.6074, 6),  # Karawaci
    (-6.3407, 106.7371, 7),  # Pamulang
    (-6.1956, 106.6322, 8),  # Tangerang_Kota
]


def _get_fitted_models():
    """Load historical weather data to fit StandardScaler and PowerTransformer."""
    global _SCALER, _PT
    if _SCALER is None or _PT is None:
        # Find historical data path relative to this file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hist_path = os.path.join(project_root, "data", "raw", "historic", "historic.csv")

        if not os.path.exists(hist_path):
            raise FileNotFoundError(
                f"Historical dataset not found at {hist_path}. Cannot fit baseline scaling models."
            )

        df_hist = pd.read_csv(hist_path)
        numerical_features = [
            "Temperature",
            "Humidity",
            "Cloud_Cover",
            "Pressure",
            "Wind_Speed",
        ]

        # Fit StandardScaler
        _SCALER = StandardScaler()
        df_scaled = _SCALER.fit_transform(df_hist[numerical_features])

        # Fit PowerTransformer (Yeo-Johnson)
        _PT = PowerTransformer(method="yeo-johnson")
        _PT.fit(df_scaled)

    return _SCALER, _PT


def _assign_nearest_location_encoded(df: pd.DataFrame) -> None:
    """Vectorized mapping of future coordinates to nearest historical location code."""
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    best_dists = np.full(len(df), float("inf"))
    best_encodeds = np.zeros(len(df), dtype=int)

    for h_lat, h_lon, encoded in HISTORIC_LOCATIONS:
        dists = (lats - h_lat) ** 2 + (lons - h_lon) ** 2
        closer = dists < best_dists
        best_dists[closer] = dists[closer]
        best_encodeds[closer] = encoded

    df["Location_Encoded"] = best_encodeds


def preprocess_future_data(file_path: str, file_timestamp: str) -> str:
    """Preprocess raw future weather forecasts using temporal cyclical encoding and baseline scaling.

    Saves the output to data/processed/future/future_<timestamp>.csv.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw future data file not found: {file_path}")

    # Read future raw dataset
    df = pd.read_csv(file_path)

    # Standardize timestamp column name
    if "Forecast_Time" in df.columns:
        df = df.rename(columns={"Forecast_Time": "Timestamp"})
    elif "time" in df.columns:
        df = df.rename(columns={"time": "Timestamp"})

    # Extract temporal components
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month

    # Assign nearest location encoded based on coordinates
    _assign_nearest_location_encoded(df)

    # Cyclical sin/cos encodings for Hour and Month
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["Month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["Month"] - 1) / 12)

    # Retrieve scaling models
    scaler, pt = _get_fitted_models()

    # Scale and power transform numerical columns
    numerical_features = [
        "Temperature",
        "Humidity",
        "Cloud_Cover",
        "Pressure",
        "Wind_Speed",
    ]
    df_scaled = scaler.transform(df[numerical_features])
    df_transformed = pt.transform(df_scaled)

    # Overwrite numerical features with standard and power scaled variants
    df[numerical_features] = df_transformed

    # Standardize column structure matching ProcessedHistoric
    columns_structure = [
        "Timestamp",
        "Temperature",
        "Humidity",
        "Cloud_Cover",
        "Pressure",
        "Wind_Speed",
        "Latitude",
        "Longitude",
        "Rain",
        "Hour",
        "Month",
        "Location_Encoded",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    # Filter to existing columns in standard structure
    valid_cols = [col for col in columns_structure if col in df.columns]
    df_final = df[valid_cols]

    # Target directory path: data/processed/future
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "data", "processed", "future")
    os.makedirs(processed_dir, exist_ok=True)

    # Save output dataset
    processed_file = os.path.join(processed_dir, f"future_{file_timestamp}.csv")
    df_final.to_csv(processed_file, index=False)

    print(f"Successfully preprocessed and saved future data to {processed_file}")

    # Run cleanup of older preprocessed snapshots
    _cleanup_old_processed_files(processed_dir, keep_last=720)

    return processed_file


def _cleanup_old_processed_files(folder_path: str, keep_last: int = 720) -> None:
    """Scan and delete oldest processed future weather CSV snapshots."""
    pattern = os.path.join(folder_path, "future_*.csv")
    files = glob.glob(pattern)

    if len(files) > keep_last:
        files.sort(key=os.path.getctime)
        files_to_delete = files[:-keep_last]
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Cleaned up old processed file: {f}")
            except Exception as e:
                print(f"Failed to delete old preprocessed file {f}: {e}")
