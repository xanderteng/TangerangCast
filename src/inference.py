import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

FUTURE_PATH = "data/raw/future.csv"
OUTPUT_PATH = "data/Final_Forecast_Results.csv"

MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_model.joblib")
LGB_PATH = os.path.join(MODEL_DIR, "lightgbm_model.joblib")
CAT_PATH = os.path.join(MODEL_DIR, "catboost_model.joblib")

WEIGHTS = {'xgb': 0.50, 'lgb': 0.15, 'cat': 0.35}

# Only used if models are not found 
def train_and_save_models():
    HISTORIC_PATH = "data/raw/historic.csv"

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(HISTORIC_PATH):
        raise FileNotFoundError(f"Training data not found: {HISTORIC_PATH}")
        
    df_hist = pd.read_csv(HISTORIC_PATH)
    
    TRAIN_COLUMNS_TO_DROP = ['Rain', 'Timestamp', 'Location', 'Latitude', 'Longitude']
    X = df_hist.drop(columns=TRAIN_COLUMNS_TO_DROP, errors='ignore')
    y = df_hist['Rain']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    lgb = LGBMClassifier(random_state=42, verbose=-1)
    cat = CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
    
    xgb.fit(X_train, y_train)
    lgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
        
    joblib.dump(xgb, XGB_PATH)
    joblib.dump(lgb, LGB_PATH)
    joblib.dump(cat, CAT_PATH)
    print("models have been saved to 'models/'\n")

def run_inference():    
    if not os.path.exists(FUTURE_PATH):
        print(f"Prediction data not found: {FUTURE_PATH}")
        return
        
    df_future = pd.read_csv(FUTURE_PATH)
    
    PREDICT_COLUMNS_TO_DROP = ['Rain', 'Fetch_Time', 'Forecast_Target_Time', 'Latitude', 'Longitude']
    X_future = df_future.drop(columns=PREDICT_COLUMNS_TO_DROP, errors='ignore')
    
    loaded_xgb = joblib.load(XGB_PATH)
    loaded_lgb = joblib.load(LGB_PATH)
    loaded_cat = joblib.load(CAT_PATH)
    
    X_future = X_future[loaded_xgb.feature_names_in_]

    f_prob_xgb = loaded_xgb.predict_proba(X_future)[:, 1]
    f_prob_lgb = loaded_lgb.predict_proba(X_future)[:, 1]
    f_prob_cat = loaded_cat.predict_proba(X_future)[:, 1]
    
    forecast_probs = (WEIGHTS['xgb'] * f_prob_xgb) + (WEIGHTS['lgb'] * f_prob_lgb) + (WEIGHTS['cat'] * f_prob_cat)
    forecast_preds = (forecast_probs >= 0.5).astype(int)
    
    final_output = pd.DataFrame()
    COLS_TO_KEEP = ['Forecast_Target_Time', 'Latitude', 'Longitude'] 

    for col in COLS_TO_KEEP:
        if col in df_future.columns:
            final_output[col] = df_future[col]
        
    final_output['rain_probability'] = forecast_probs.round(4)
    final_output['predicted_rain'] = forecast_preds
    
    final_output.to_csv(OUTPUT_PATH, index=False)
    print(f">>> Finished forecasting, file saved at: '{OUTPUT_PATH}'")

def main():
    # check if all models exist
    models_exist = os.path.exists(XGB_PATH) and os.path.exists(LGB_PATH) and os.path.exists(CAT_PATH)
    
    if not models_exist:
        print("[STATUS] Pre-trained models not found. Starting training phase...")
        train_and_save_models()
    else:
        print("[STATUS] Found pre-trained models. Skipping training phase")
        
    # Predict outcome
    run_inference()

if __name__ == "__main__":
    main()
