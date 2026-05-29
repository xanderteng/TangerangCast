import os
import glob
import pandas as pd
import numpy as np
import pytest
import src.preprocessor
from src.preprocessor import preprocess_future_data, _cleanup_old_processed_files
from sklearn.preprocessing import StandardScaler, PowerTransformer


@pytest.fixture
def dummy_future_csv(tmp_path):
    """Create a temporary raw future grid CSV file for testing."""
    data = {
        "Forecast_Time": ["2026-05-28T00:00", "2026-05-28T01:00", "2026-05-28T02:00"],
        "Latitude": [-6.2427, -6.3407, -6.2226],
        "Longitude": [106.5173, 106.7371, 106.6533],
        "Temperature": [25.5, 26.0, 24.8],
        "Humidity": [80.0, 78.0, 85.0],
        "Wind_Speed": [5.0, 6.2, 4.8],
        "Cloud_Cover": [10, 20, 95],
        "Pressure": [1008.2, 1007.9, 1008.5],
        "Rain": [0, 0, 1],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "future_dummy.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_preprocessor_execution(dummy_future_csv, monkeypatch, tmp_path):
    """Test that preprocess_future_data extracts features, executes nearest location mapping, standardizes shapes, and saves the output."""
    # Pre-fit dummy StandardScaler and PowerTransformer to avoid reading historic.csv
    dummy_scaler = StandardScaler()
    dummy_pt = PowerTransformer(method="yeo-johnson")
    dummy_df = pd.DataFrame(
        np.random.normal(size=(10, 5)),
        columns=["Temperature", "Humidity", "Cloud_Cover", "Pressure", "Wind_Speed"],
    )
    dummy_scaler.fit(dummy_df)
    dummy_pt.fit(dummy_scaler.transform(dummy_df))

    # Monkeypatch preprocessor variables directly to bypass file-based _get_fitted_models
    monkeypatch.setattr(src.preprocessor, "_SCALER", dummy_scaler)
    monkeypatch.setattr(src.preprocessor, "_PT", dummy_pt)

    # We redirect project_root to tmp_path for saving output
    # to avoid writing to standard data/processed/ during unit tests
    monkeypatch.setattr(
        os.path,
        "abspath",
        lambda path: str(tmp_path) if "src" in path else os.path.realpath(path),
    )

    timestamp = "99999999_9999"
    processed_file = preprocess_future_data(str(dummy_future_csv), timestamp)

    try:
        # Check that the file was created
        assert os.path.exists(processed_file)

        # Read file and verify structure and content
        df_processed = pd.read_csv(processed_file)

        expected_cols = [
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

        for col in expected_cols:
            assert col in df_processed.columns

        # Verify columns counts
        assert df_processed.shape[0] == 3

        # Verify cyclical features
        assert df_processed["Hour"].tolist() == [0, 1, 2]
        assert df_processed["Month"].tolist() == [5, 5, 5]

        # Verify location mappings (Cikupa -> 4, Pamulang -> 7, Alam_Sutera -> 0)
        assert df_processed["Location_Encoded"].tolist() == [4, 7, 0]

        # Verify sin/cos calculations
        np.testing.assert_allclose(df_processed["hour_sin"].iloc[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(df_processed["hour_cos"].iloc[0], 1.0, atol=1e-7)

    finally:
        # Cleanup test output file
        if os.path.exists(processed_file):
            os.remove(processed_file)


def test_preprocessor_cleanup(tmp_path):
    """Test that _cleanup_old_processed_files removes oldest files keeping only the requested limit."""
    # Create 5 dummy future preprocessed files in tmp_path
    for i in range(5):
        file_path = tmp_path / f"future_20260528_100{i}.csv"
        # Write dummy csv
        pd.DataFrame({"x": [1]}).to_csv(file_path, index=False)
        # Sleep briefly to ensure unique creation times
        import time

        time.sleep(0.01)

    # Check 5 files exist
    files = glob.glob(str(tmp_path / "future_*.csv"))
    assert len(files) == 5

    # Run cleanup, keeping only the last 3
    _cleanup_old_processed_files(str(tmp_path), keep_last=3)

    # Check 3 files remain
    remaining_files = glob.glob(str(tmp_path / "future_*.csv"))
    assert len(remaining_files) == 3

    # Check the oldest ones (1000 and 1001) were deleted
    remaining_basenames = [os.path.basename(f) for f in remaining_files]
    assert "future_20260528_1000.csv" not in remaining_basenames
    assert "future_20260528_1001.csv" not in remaining_basenames
    assert "future_20260528_1002.csv" in remaining_basenames
    assert "future_20260528_1003.csv" in remaining_basenames
    assert "future_20260528_1004.csv" in remaining_basenames
