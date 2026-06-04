import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
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
    df = pd.merge(df_raw, df_forecast, on=["dt", "Latitude", "Longitude"], how="inner")

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

    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            roc_auc_str = f"{roc_auc:.4f}"
        except ValueError:
            roc_auc_str = "N/A (Only one class present in true values)"
    else:
        roc_auc_str = "N/A (Only one class present in true values)"

    # Print Report
    print("-" * 50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc_str}")
    print("-" * 50)

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


def evaluate_forecast_vs_current():
    """Evaluate final ONNX Stacking predictions against actual hourly weather observations."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_dir = os.path.join(project_root, "data", "raw", "current")
    forecast_dir = os.path.join(project_root, "data", "processed", "forecast")

    forecast_files = sorted(glob.glob(os.path.join(forecast_dir, "forecast_*.csv")))
    current_files = sorted(glob.glob(os.path.join(current_dir, "current_*.csv")))

    print("==================================================")
    print("      Latest Forecast vs Current 24h Performance")
    print("==================================================")
    print(f"Total forecast files found: {len(forecast_files)}")
    print(f"Total current files found: {len(current_files)}")

    if len(forecast_files) < 5 or len(current_files) < 25:
        print("[ERROR] Not enough data files to perform the standard 24-hour evaluation cycle.")
        print(f"Required: at least 5 forecast files (found {len(forecast_files)}) and 25 current files (found {len(current_files)}).")
        return

    # Select the latest 5 forecasts and latest 25 current files
    selected_forecasts = forecast_files[-5:]
    selected_currents = current_files[-25:]

    print("\nSelected Forecast files for evaluation:")
    for idx, f in enumerate(selected_forecasts):
        mark = " (Excluded/Latest)" if idx == 4 else f" (F_{idx+1})"
        print(f"  Forecast {idx+1}: {os.path.basename(f)}{mark}")

    print("\nSelected Current snapshots slice details:")
    print(f"  First current (Excluded): {os.path.basename(selected_currents[0])}")
    print(f"  Evaluating Q1 current: {os.path.basename(selected_currents[1])} to {os.path.basename(selected_currents[6])}")
    print(f"  Evaluating Q2 current: {os.path.basename(selected_currents[7])} to {os.path.basename(selected_currents[12])}")
    print(f"  Evaluating Q3 current: {os.path.basename(selected_currents[13])} to {os.path.basename(selected_currents[18])}")
    print(f"  Evaluating Q4 current: {os.path.basename(selected_currents[19])} to {os.path.basename(selected_currents[24])}")

    # Helper function to load and process a list of current files
    def load_quarter_current(files_list):
        dfs = []
        for filepath in files_list:
            df = pd.read_csv(filepath)
            if df.empty or "Fetch_Time" not in df.columns:
                continue
            # Get Fetch_Time from the first row and round down to 00 minutes
            fetch_time = df["Fetch_Time"].iloc[0]
            dt_obj = pd.to_datetime(fetch_time).replace(minute=0, second=0, microsecond=0)
            df["dt"] = dt_obj
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Define the quarters
    quarters = [
        ("Quarter 1", selected_forecasts[0], selected_currents[1:7]),
        ("Quarter 2", selected_forecasts[1], selected_currents[7:13]),
        ("Quarter 3", selected_forecasts[2], selected_currents[13:19]),
        ("Quarter 4", selected_forecasts[3], selected_currents[19:25]),
    ]

    matched_dfs = []

    for name, forecast_path, current_paths in quarters:
        print(f"\nProcessing {name}...")
        df_forecast = pd.read_csv(forecast_path)
        df_forecast["dt"] = pd.to_datetime(df_forecast["Forecast_Target_Time"])

        df_current_q = load_quarter_current(current_paths)
        if df_current_q.empty:
            print(f"  [WARNING] No current data loaded for {name}.")
            continue

        df_merged_q = pd.merge(
            df_current_q,
            df_forecast,
            on=["dt", "Latitude", "Longitude"],
            how="inner"
        )
        print(f"  Matched {df_merged_q.shape[0]} grid points for {name}.")
        matched_dfs.append(df_merged_q)

    if not matched_dfs:
        print("\n[ERROR] No matched data across any of the quarters.")
        return

    df_all_matched = pd.concat(matched_dfs, ignore_index=True)

    # Compute overall metrics on the combined dataset
    y_true = df_all_matched["Rain"].values
    y_pred = df_all_matched["predicted_rain"].values
    y_prob = df_all_matched["rain_probability"].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            roc_auc_str = f"{roc_auc:.4f}"
        except ValueError:
            roc_auc_str = "N/A (Only one class present in true values)"
    else:
        roc_auc_str = "N/A (Only one class present in true values)"

    # Generate Scikit-Learn Classification Report
    class_report_str = classification_report(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # Print Report to Console
    print("\n" + "=" * 50)
    print("      24-HOUR EVALUATION REPORT (4 QUARTERS)")
    print("==================================================")
    print(f"Total evaluated matched points: {df_all_matched.shape[0]}")
    print("-" * 50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc_str}")
    print("-" * 50)
    print("Classification Report:")
    print(class_report_str)
    print("-" * 50)
    print("Confusion Matrix Details:")
    print(f"  True Positives  (Rain correctly predicted)   : {tp}")
    print(f"  False Positives (Rain predicted but clear)   : {fp}")
    print(f"  False Negatives (Rain missed by model)       : {fn}")
    print(f"  True Negatives  (Clear correctly predicted)  : {tn}")
    print("==================================================")

    # Save to file
    report_path = os.path.join(forecast_dir, "classification_report_24h.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write("==================================================\n")
            rf.write("      TangerangCast 24-Hour Stacking Performance Report\n")
            rf.write("==================================================\n")
            rf.write(f"Generated at             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            rf.write(f"Total matched grid points: {df_all_matched.shape[0]}\n")
            rf.write(f"Forecasts evaluated      : 4 cycles\n")
            rf.write(f"Currents evaluated       : 24 snapshots\n")
            rf.write("-" * 50 + "\n")
            rf.write(f"Accuracy  : {accuracy:.4f}\n")
            rf.write(f"Precision : {precision:.4f}\n")
            rf.write(f"Recall    : {recall:.4f}\n")
            rf.write(f"F1-Score  : {f1:.4f}\n")
            rf.write(f"ROC-AUC   : {roc_auc_str}\n")
            rf.write("-" * 50 + "\n")
            rf.write("Classification Report:\n")
            rf.write(class_report_str)
            rf.write("-" * 50 + "\n")
            rf.write("Confusion Matrix Details:\n")
            rf.write(f"  True Positives  (Rain correctly predicted)   : {tp}\n")
            rf.write(f"  False Positives (Rain predicted but clear)   : {fp}\n")
            rf.write(f"  False Negatives (Rain missed by model)       : {fn}\n")
            rf.write(f"  True Negatives  (Clear correctly predicted)  : {tn}\n")
            rf.write("==================================================\n")
        print(f"\n[SUCCESS] 24-hour classification report saved at: {report_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save classification report: {e}")


if __name__ == "__main__":
    evaluate_latest_forecast()
    evaluate_forecast_vs_current()
