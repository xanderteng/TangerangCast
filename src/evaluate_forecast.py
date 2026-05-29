import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def get_latest_csv(directory: str, pattern: str = "*.csv") -> str | None:
    """Find the latest CSV file by modification time in the specified directory."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def evaluate_latest_forecast():
    """Evaluate final ONNX Stacking predictions against Open-Meteo future forecast baseline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    future_dir = os.path.join(project_root, "data", "raw", "future")
    forecast_dir = os.path.join(project_root, "data", "processed", "forecast")

    raw_future_csv = get_latest_csv(future_dir)
    forecast_csv = get_latest_csv(forecast_dir)

    print("==================================================")
    print("      Forecast Machine Learning Performance")
    print("==================================================")

    if not raw_future_csv:
        print("[ERROR] No raw future weather files found under 'data/raw/future/'.")
        return
    if not forecast_csv:
        print(
            "[ERROR] No forecast prediction files found under 'data/processed/forecast/'."
        )
        return

    print(f"Baseline raw future file : {os.path.basename(raw_future_csv)}")
    print(f"ML Stacking forecast file: {os.path.basename(forecast_csv)}")

    # Load datasets
    df_raw = pd.read_csv(raw_future_csv)
    df_forecast = pd.read_csv(forecast_csv)

    # Standardize and parse datetimes
    df_raw["dt"] = pd.to_datetime(df_raw["Forecast_Time"])
    df_forecast["dt"] = pd.to_datetime(df_forecast["Forecast_Target_Time"])

    # Inner merge on datetimes and coordinates
    df = pd.merge(
        df_raw, df_forecast, on=["dt", "Latitude", "Longitude"], how="inner"
    )

    if df.empty:
        print(
            "[ERROR] Merged dataset is empty. Check that coordinates and datetimes match."
        )
        return

    print(f"Total matching points evaluated: {df.shape[0]}")

    # Ground truth (Rain status from Open-Meteo grid) vs. predictions (Tuned ONNX Stacking)
    y_true = df["Rain"].values
    y_pred = df["predicted_rain"].values
    y_prob = df["rain_probability"].values

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )  # Safe default if no positive predictions
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        roc_auc_str = f"{roc_auc:.4f}"
    except ValueError:
        # Fails if there is only one class in y_true
        roc_auc_str = "N/A (Only one class present in true values)"

    # Print Report
    print("-"*50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc_str}")
    print("-"*50)

    # Confusion matrix elements for more detail
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    print("Confusion Matrix Details:")
    print(f"  True Positives  (Rain correctly predicted)   : {tp}")
    print(f"  False Positives (Rain predicted but clear)   : {fp}")
    print(f"  False Negatives (Rain missed by model)       : {fn}")
    print(f"  True Negatives  (Clear correctly predicted)  : {tn}")
    print("==================================================")


if __name__ == "__main__":
    evaluate_latest_forecast()
