import os
import glob
import pandas as pd
import pytest
from src.inference import run_onnx_inference, _cleanup_old_forecast_files


@pytest.fixture
def dummy_processed_csv(tmp_path):
    """Create a temporary preprocessed future CSV file containing all 12 model features."""
    data = {
        "Timestamp": [
            "2026-05-28 00:00:00",
            "2026-05-28 01:00:00",
            "2026-05-28 02:00:00",
        ],
        "Temperature": [-1.0, 0.0, 1.2],
        "Humidity": [0.5, -0.2, 0.9],
        "Cloud_Cover": [-0.8, 0.4, 0.7],
        "Pressure": [0.1, -0.3, 0.8],
        "Wind_Speed": [-0.5, 0.2, 1.1],
        "Hour": [0, 1, 2],
        "Month": [5, 5, 5],
        "Location_Encoded": [4, 7, 0],
        "hour_sin": [0.0, 0.2588, 0.5],
        "hour_cos": [1.0, 0.9659, 0.8660],
        "month_sin": [0.8660, 0.8660, 0.8660],
        "month_cos": [-0.5, -0.5, -0.5],
        "Latitude": [-6.2427, -6.3407, -6.2226],
        "Longitude": [106.5173, 106.7371, 106.6533],
        "Rain": [0, 0, 1],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "future_processed.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_onnx_inference_execution(dummy_processed_csv):
    """Test that run_onnx_inference loads all 4 decoupled ONNX models and runs the stacking predictions correctly."""
    timestamp = "99999999_9999"
    forecast_file = run_onnx_inference(str(dummy_processed_csv), timestamp)

    try:
        # Verify the file was created in data/processed/forecast/
        assert os.path.exists(forecast_file)
        assert "forecast_" in os.path.basename(forecast_file)

        # Read forecast output and verify column structure and predictions
        df_forecast = pd.read_csv(forecast_file)
        expected_cols = [
            "Forecast_Target_Time",
            "Latitude",
            "Longitude",
            "rain_probability",
            "predicted_rain",
        ]

        for col in expected_cols:
            assert col in df_forecast.columns

        # Verify shapes
        assert df_forecast.shape[0] == 3

        # Verify predictions constraints
        assert df_forecast["predicted_rain"].isin([0, 1]).all()
        assert (df_forecast["rain_probability"] >= 0.0).all()
        assert (df_forecast["rain_probability"] <= 1.0).all()

    finally:
        # Cleanup test output file
        if os.path.exists(forecast_file):
            os.remove(forecast_file)


def test_forecast_cleanup(tmp_path):
    """Test that _cleanup_old_forecast_files removes oldest forecast files keeping only the requested limit."""
    # Create 5 dummy forecast files in tmp_path
    for i in range(5):
        file_path = tmp_path / f"forecast_20260528_100{i}.csv"
        # Write dummy csv
        pd.DataFrame({"x": [1]}).to_csv(file_path, index=False)
        # Sleep briefly to ensure unique creation times
        import time

        time.sleep(0.01)

    # Check 5 files exist
    files = glob.glob(str(tmp_path / "forecast_*.csv"))
    assert len(files) == 5

    # Run cleanup, keeping only the last 3
    _cleanup_old_forecast_files(str(tmp_path), keep_last=3)

    # Check 3 files remain
    remaining_files = glob.glob(str(tmp_path / "forecast_*.csv"))
    assert len(remaining_files) == 3

    # Check the oldest ones (1000 and 1001) were deleted
    remaining_basenames = [os.path.basename(f) for f in remaining_files]
    assert "forecast_20260528_1000.csv" not in remaining_basenames
    assert "forecast_20260528_1001.csv" not in remaining_basenames
    assert "forecast_20260528_1002.csv" in remaining_basenames
    assert "forecast_20260528_1003.csv" in remaining_basenames
    assert "forecast_20260528_1004.csv" in remaining_basenames
