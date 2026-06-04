from __future__ import annotations

import argparse
import glob
import os
import warnings

import mlflow
import numpy as np
import onnxmltools
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from onnxmltools.convert import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklFloat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROCESSED_HIST_PATH = os.path.join(
    _PROJECT_ROOT, "data", "processed", "ProcessedHistoric.csv"
)
_CURRENT_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "current")
_HISTORIC_RAW_PATH = os.path.join(
    _PROJECT_ROOT, "data", "raw", "historic", "historic.csv"
)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

TRAINING_FEATURES = [
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

_NUMERIC_FEATURES = [
    "Temperature",
    "Humidity",
    "Cloud_Cover",
    "Pressure",
    "Wind_Speed",
]

HISTORIC_LOCATIONS = [
    ("Alam_Sutera", -6.2226, 106.6533, 0),
    ("BSD_City", -6.3006, 106.6538, 1),
    ("Batuceper", -6.1557, 106.6579, 2),
    ("Bintaro", -6.2738, 106.7136, 3),
    ("Cikupa", -6.2427, 106.5173, 4),
    ("Gading_Serpong", -6.2416, 106.6285, 5),
    ("Karawaci", -6.2269, 106.6074, 6),
    ("Pamulang", -6.3407, 106.7371, 7),
    ("Tangerang_Kota", -6.1956, 106.6322, 8),
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _get_frozen_scalers() -> tuple[StandardScaler, PowerTransformer]:
    """Fit StandardScaler + PowerTransformer on the raw historic baseline.

    These are *frozen* — they are always fitted from historic.csv so that
    the feature distribution seen by the model stays consistent across
    retrain cycles.
    """
    df_hist = pd.read_csv(_HISTORIC_RAW_PATH)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_hist[_NUMERIC_FEATURES])
    pt = PowerTransformer(method="yeo-johnson")
    pt.fit(scaled)
    return scaler, pt


def _assign_location(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the nearest historic location name and encoded ID."""
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    best_dists = np.full(len(df), float("inf"))
    best_names = np.empty(len(df), dtype=object)
    best_codes = np.zeros(len(df), dtype=int)

    for name, h_lat, h_lon, code in HISTORIC_LOCATIONS:
        dists = (lats - h_lat) ** 2 + (lons - h_lon) ** 2
        closer = dists < best_dists
        best_dists[closer] = dists[closer]
        best_names[closer] = name
        best_codes[closer] = code

    df["Location"] = best_names
    df["Location_Encoded"] = best_codes
    return df


def _preprocess_current_csvs(
    days: int = 7,
    scaler: StandardScaler | None = None,
    pt: PowerTransformer | None = None,
) -> pd.DataFrame:
    """Load current CSVs from the last *days* days and transform them to
    match the ProcessedHistoric schema (17 columns).
    """
    if scaler is None or pt is None:
        scaler, pt = _get_frozen_scalers()

    paths = sorted(glob.glob(os.path.join(_CURRENT_DIR, "current_*.csv")))
    if not paths:
        print("[WARN] No current CSV files found.")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Rename & parse timestamp
    df = df.rename(columns={"Fetch_Time": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Filter to last N days
    cutoff = df["Timestamp"].max() - pd.Timedelta(days=days)
    df = df[df["Timestamp"] >= cutoff].copy()

    if df.empty:
        return df

    # Temporal features
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month

    # Location assignment
    df = _assign_location(df)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["Month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["Month"] - 1) / 12)

    # Scale numerical features (frozen scalers)
    df[_NUMERIC_FEATURES] = pt.transform(scaler.transform(df[_NUMERIC_FEATURES]))

    # Format timestamp back to string to match ProcessedHistoric
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Select and order columns to match ProcessedHistoric schema
    output_cols = [
        "Timestamp",
        "Temperature",
        "Humidity",
        "Cloud_Cover",
        "Pressure",
        "Wind_Speed",
        "Location",
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
    return df[output_cols]


def _apply_sliding_window(
    df_historic: pd.DataFrame,
    df_new: pd.DataFrame,
) -> pd.DataFrame:
    """Drop the oldest 7 days from *df_historic*, append *df_new*, dedupe,
    and overwrite ProcessedHistoric.csv.
    """
    df_historic["_ts"] = pd.to_datetime(df_historic["Timestamp"])

    oldest = df_historic["_ts"].min()
    cutoff = oldest + pd.Timedelta(days=7)
    df_trimmed = df_historic[df_historic["_ts"] >= cutoff].copy()
    df_trimmed = df_trimmed.drop(columns=["_ts"])

    print(
        f"  Sliding window: dropped {len(df_historic) - len(df_trimmed)} rows "
        f"older than {cutoff}"
    )

    if not df_new.empty:
        df_merged = pd.concat([df_trimmed, df_new], ignore_index=True)
    else:
        df_merged = df_trimmed

    # Deduplicate
    before = len(df_merged)
    df_merged = df_merged.drop_duplicates(
        subset=["Timestamp", "Latitude", "Longitude"],
        keep="last",
    )
    dupes = before - len(df_merged)
    if dupes:
        print(f"  Removed {dupes} duplicate rows")

    df_merged.to_csv(_PROCESSED_HIST_PATH, index=False)
    print(f"  Saved updated ProcessedHistoric.csv: {len(df_merged)} rows")
    return df_merged


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    weight = n_neg / max(n_pos, 1)
    print(
        f"  Class balance — no-rain: {n_neg}, rain: {n_pos}, scale_pos_weight: {weight:.3f}"
    )
    return weight


def _tune_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_iter: int = 30,
    cv: int = 3,
) -> tuple:
    """Tune XGBoost, LightGBM, and CatBoost with RandomizedSearchCV."""

    print("\n=== Tuning XGBoost ===")
    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        ),
        param_distributions={
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  Best XGB params: {xgb_search.best_params_}")

    print("\n=== Tuning LightGBM ===")
    lgb_search = RandomizedSearchCV(
        LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            is_unbalance=True,
        ),
        param_distributions={
            "n_estimators": [200, 300, 500],
            "num_leaves": [31, 50, 70, 100],
            "max_depth": [-1, 5, 10, 15],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    lgb_search.fit(X_train, y_train)
    best_lgb = lgb_search.best_estimator_
    print(f"  Best LGB params: {lgb_search.best_params_}")

    print("\n=== Tuning CatBoost ===")
    cat_search = RandomizedSearchCV(
        CatBoostClassifier(
            random_state=42,
            verbose=0,
            thread_count=-1,
            scale_pos_weight=scale_pos_weight,
        ),
        param_distributions={
            "iterations": [200, 300, 500],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 7],
        },
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    cat_search.fit(X_train, y_train)
    best_cat = cat_search.best_estimator_
    print(f"  Best CatBoost params: {cat_search.best_params_}")

    return best_xgb, best_lgb, best_cat


def _train_stacking(
    best_xgb,
    best_lgb,
    best_cat,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv: int = 3,
) -> tuple:
    """Train StackingClassifier and optimise threshold for F1."""
    from sklearn.ensemble import StackingClassifier

    print("\n=== Training Stacking Ensemble ===")
    estimators = [("xgb", best_xgb), ("lgb", best_lgb), ("cat", best_cat)]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight="balanced"),
        cv=cv,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)

    # Threshold optimisation
    print("\n=== Optimising Threshold ===")
    probs = stacking.predict_proba(X_val)[:, 1]
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in np.linspace(0.3, 0.7, 41):
        preds = (probs >= threshold).astype(int)
        current_f1 = f1_score(y_val, preds)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    final_preds = (probs >= best_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, final_preds),
        "precision": precision_score(y_val, final_preds, zero_division=0),
        "recall": recall_score(y_val, final_preds, zero_division=0),
        "f1": f1_score(y_val, final_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probs),
        "best_threshold": best_threshold,
    }

    print(f"  Best threshold: {best_threshold:.3f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    return stacking, metrics


def _export_onnx(
    best_xgb,
    best_lgb,
    best_cat,
    meta_model,
    n_features: int,
) -> list[str]:
    """Export the 4 decoupled ONNX models to models/."""
    print("\n=== Exporting ONNX Models ===")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    paths: list[str] = []

    # XGBoost
    onnx_input = [("input", OnnxFloat([None, n_features]))]
    xgb_onnx = convert_xgboost(best_xgb, initial_types=onnx_input)
    xgb_path = os.path.join(_MODELS_DIR, "xgboost_model.onnx")
    onnxmltools.utils.save_model(xgb_onnx, xgb_path)
    paths.append(xgb_path)
    print(f"  Saved {xgb_path}")

    # LightGBM
    lgb_onnx = convert_lightgbm(best_lgb, initial_types=onnx_input)
    lgb_path = os.path.join(_MODELS_DIR, "lightgbm_model.onnx")
    onnxmltools.utils.save_model(lgb_onnx, lgb_path)
    paths.append(lgb_path)
    print(f"  Saved {lgb_path}")

    # CatBoost (native ONNX export)
    cat_path = os.path.join(_MODELS_DIR, "catboost_model.onnx")
    best_cat.save_model(cat_path, format="onnx")
    paths.append(cat_path)
    print(f"  Saved {cat_path}")

    # Meta-model (LogisticRegression)
    meta_input = [("meta_input", SklFloat([None, 3]))]
    meta_onnx = convert_sklearn(meta_model, initial_types=meta_input)
    meta_path = os.path.join(_MODELS_DIR, "meta_model.onnx")
    with open(meta_path, "wb") as f:
        f.write(meta_onnx.SerializeToString())
    paths.append(meta_path)
    print(f"  Saved {meta_path}")

    return paths


def _log_to_mlflow(
    metrics: dict,
    best_xgb,
    best_lgb,
    best_cat,
    model_paths: list[str],
    train_rows: int,
) -> None:
    """Log training run to MLflow."""
    print("\n=== Logging to MLflow ===")
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlruns.db')}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("TangerangCast-Retrain")

    with mlflow.start_run():
        # Log metrics
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        # Log training metadata
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("n_features", len(TRAINING_FEATURES))

        # Log best hyperparams
        for param, val in best_xgb.get_params().items():
            if param in (
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "scale_pos_weight",
            ):
                mlflow.log_param(f"xgb_{param}", val)

        for param, val in best_lgb.get_params().items():
            if param in (
                "n_estimators",
                "num_leaves",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
            ):
                mlflow.log_param(f"lgb_{param}", val)

        for param, val in best_cat.get_all_params().items():
            if param in ("iterations", "depth", "learning_rate", "l2_leaf_reg"):
                mlflow.log_param(f"cat_{param}", val)

        # Log model artifacts
        for path in model_paths:
            mlflow.log_artifact(path)

    print(f"  MLflow tracking URI: {tracking_uri}")
    print("  Run logged successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dry_run: bool = False) -> None:
    """Orchestrate the full weekly retrain pipeline."""
    print("=" * 60)
    print("  TangerangCast Weekly Model Retrain")
    print("=" * 60)

    if dry_run:
        print("\n*** DRY RUN MODE — subsampled data, minimal search ***\n")

    # 1. Load ProcessedHistoric
    print("\n[1/7] Loading ProcessedHistoric.csv...")
    if not os.path.exists(_PROCESSED_HIST_PATH):
        print(f"  ERROR: {_PROCESSED_HIST_PATH} not found.")
        return
    df_historic = pd.read_csv(_PROCESSED_HIST_PATH)
    print(f"  Loaded {len(df_historic)} rows")

    # 2. Preprocess current CSVs
    print("\n[2/7] Preprocessing current weather data (last 7 days)...")
    scaler, pt = _get_frozen_scalers()
    df_current = _preprocess_current_csvs(days=7, scaler=scaler, pt=pt)
    print(f"  Preprocessed {len(df_current)} current rows")

    # 3. Apply sliding window
    print("\n[3/7] Applying 7-day sliding window...")
    df_train_full = _apply_sliding_window(df_historic, df_current)

    # 4. Prepare features
    print("\n[4/7] Preparing feature matrix...")
    columns_to_drop = ["Rain", "Timestamp", "Location", "Latitude", "Longitude"]
    X = df_train_full.drop(columns=columns_to_drop, errors="ignore").values
    y = df_train_full["Rain"].values.astype(int)

    if dry_run:
        # Subsample for quick verification
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=min(1000, len(X)), replace=False)
        X, y = X[idx], y[idx]
        print(f"  Dry-run subsample: {len(X)} rows")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} rows, Val: {len(X_val)} rows")

    # 5. Train
    print("\n[5/7] Training models...")
    spw = _compute_scale_pos_weight(y_train)

    n_iter = 2 if dry_run else 30
    cv = 2 if dry_run else 3

    best_xgb, best_lgb, best_cat = _tune_base_models(
        X_train, y_train, spw, n_iter=n_iter, cv=cv
    )

    stacking, metrics = _train_stacking(
        best_xgb, best_lgb, best_cat, X_train, y_train, X_val, y_val, cv=cv
    )

    # 6. Export ONNX
    print("\n[6/7] Exporting ONNX models...")
    model_paths = _export_onnx(
        best_xgb,
        best_lgb,
        best_cat,
        stacking.final_estimator_,
        n_features=X_train.shape[1],
    )

    # 7. Log to MLflow
    print("\n[7/7] Logging to MLflow...")
    _log_to_mlflow(metrics, best_xgb, best_lgb, best_cat, model_paths, len(X_train))

    # Summary
    print("\n" + "=" * 60)
    print("  RETRAIN COMPLETE")
    print("=" * 60)
    print(f"  Training rows  : {len(X_train)}")
    print(f"  Validation rows: {len(X_val)}")
    print(f"  Best threshold : {metrics['best_threshold']:.3f}")
    print(f"  F1-Score       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Models saved to: {_MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TangerangCast weekly model retrain")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick verification with subsampled data and minimal search",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
