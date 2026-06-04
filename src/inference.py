import os
import glob
import pandas as pd
import numpy as np
import onnxruntime as rt


def run_onnx_inference(processed_file_path: str, file_timestamp: str) -> str:
    """Run decoupled ONNX models (XGB, LGBM, CatBoost) and Logistic Stacking Meta-model

    on preprocessed future grid forecasts, saving to data/processed/forecast/forecast_<timestamp>.csv.
    """
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Preprocessed future file not found: {processed_file_path}"
        )

    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

    xgb_path = os.path.join(models_dir, "xgboost_model.onnx")
    lgb_path = os.path.join(models_dir, "lightgbm_model.onnx")
    cat_path = os.path.join(models_dir, "catboost_model.onnx")
    meta_path = os.path.join(models_dir, "meta_model.onnx")

    # Load preprocessed future data
    df = pd.read_csv(processed_file_path)

    # 12 features in the exact training structure order
    features = [
        "Temperature",
        "Humidity",
        "Cloud_Cover",
        "Pressure",
        "Wind_Speed",
        "Hour",
        "Month",
        "Location_Encoded",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    # Convert features matrix to NumPy float32 (fixes input names bug)
    X = df[features].values.astype(np.float32)

    # Initialize ONNX inference sessions with error handling
    try:
        sess_xgb = rt.InferenceSession(xgb_path)
        sess_lgb = rt.InferenceSession(lgb_path)
        sess_cat = rt.InferenceSession(cat_path)
        sess_meta = rt.InferenceSession(meta_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ONNX models. Ensure all 4 models exist in {models_dir}. Error: {e}"
        )

    def get_class_1_prob(onnx_output):
        probs = onnx_output[1]
        if isinstance(probs, list) and isinstance(probs[0], dict):
            return np.array([p[1] for p in probs])
        return probs[:, 1]

    # Run base classifiers
    prob_xgb = get_class_1_prob(sess_xgb.run(None, {sess_xgb.get_inputs()[0].name: X}))
    prob_lgb = get_class_1_prob(sess_lgb.run(None, {sess_lgb.get_inputs()[0].name: X}))
    prob_cat = get_class_1_prob(sess_cat.run(None, {sess_cat.get_inputs()[0].name: X}))

    # Stack prediction probabilities
    stacked_features = np.column_stack((prob_xgb, prob_lgb, prob_cat)).astype(
        np.float32
    )

    # Execute meta-stacking classifier
    final_prob = get_class_1_prob(
        sess_meta.run(None, {sess_meta.get_inputs()[0].name: stacked_features})
    )

    # Best stacking threshold optimized in Modelling.ipynb is 0.400
    threshold = 0.400
    final_pred = (final_prob >= threshold).astype(int)

    # Build final forecast results
    final_output = pd.DataFrame()
    final_output["Forecast_Target_Time"] = df["Timestamp"]
    final_output["Latitude"] = df["Latitude"]
    final_output["Longitude"] = df["Longitude"]
    final_output["rain_probability"] = final_prob.round(4)
    final_output["predicted_rain"] = final_pred

    # Target directory path: data/processed/forecast
    forecast_dir = os.path.join(project_root, "data", "processed", "forecast")
    os.makedirs(forecast_dir, exist_ok=True)

    # Save final forecast dataset
    forecast_file = os.path.join(forecast_dir, f"forecast_{file_timestamp}.csv")
    final_output.to_csv(forecast_file, index=False)

    print(
        f"Successfully ran ONNX Stacking Stacker and saved predictions to {forecast_file}"
    )

    # Sweep older forecast snapshots
    _cleanup_old_forecast_files(forecast_dir, keep_last=720)

    return forecast_file


def _cleanup_old_forecast_files(folder_path: str, keep_last: int = 720) -> None:
    """Scan and delete oldest forecast prediction CSV snapshots."""
    pattern = os.path.join(folder_path, "forecast_*.csv")
    files = glob.glob(pattern)

    if len(files) > keep_last:
        files.sort(key=os.path.getctime)
        files_to_delete = files[:-keep_last]
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Cleaned up old forecast file: {f}")
            except Exception as e:
                print(f"Failed to delete old forecast file {f}: {e}")
