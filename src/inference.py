import os
import glob
import pandas as pd
import numpy as np
import onnxruntime as rt
import gc


def run_onnx_inference(processed_file_path: str, file_timestamp: str) -> str:
    """Run decoupled ONNX models (XGB, LGBM, CatBoost) and Logistic Stacking Meta-model
    sequentially to save memory on 1GB RAM VPS.
    """
    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Preprocessed future file not found: {processed_file_path}"
        )

    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")

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

    # Low-memory session options configuration
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False

    def get_class_1_prob(onnx_output):
        probs = onnx_output[1]
        if isinstance(probs, list) and isinstance(probs[0], dict):
            return np.array([p[1] for p in probs])
        return probs[:, 1]

    try:
        # 1. Run XGBoost Session & Clear Memory Immediately
        xgb_path = os.path.join(models_dir, "xgboost_model.onnx")
        sess = rt.InferenceSession(xgb_path, opts)
        prob_xgb = get_class_1_prob(sess.run(None, {sess.get_inputs()[0].name: X}))
        del sess
        gc.collect()

        # 2. Run LightGBM Session & Clear Memory Immediately
        lgb_path = os.path.join(models_dir, "lightgbm_model.onnx")
        sess = rt.InferenceSession(lgb_path, opts)
        prob_lgb = get_class_1_prob(sess.run(None, {sess.get_inputs()[0].name: X}))
        del sess
        gc.collect()

        # 3. Run CatBoost Session & Clear Memory Immediately
        cat_path = os.path.join(models_dir, "catboost_model.onnx")
        sess = rt.InferenceSession(cat_path, opts)
        prob_cat = get_class_1_prob(sess.run(None, {sess.get_inputs()[0].name: X}))
        del sess
        gc.collect()

        # Stack prediction probabilities
        stacked_features = np.column_stack((prob_xgb, prob_lgb, prob_cat)).astype(
            np.float32
        )

        # 4. Run Meta Classifier Session & Clear Memory Immediately
        meta_path = os.path.join(models_dir, "meta_model.onnx")
        sess = rt.InferenceSession(meta_path, opts)
        final_prob = get_class_1_prob(
            sess.run(None, {sess.get_inputs()[0].name: stacked_features})
        )
        del sess
        gc.collect()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or execute ONNX models sequentially. Ensure all 4 models exist in {models_dir}. Error: {e}"
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
