import os
from src.api_fetcher import APIFetcher


def test_api_fetcher_initialization(tmp_path, monkeypatch):
    """Test that APIFetcher initializes correctly and creates directories inside a sandbox."""
    monkeypatch.chdir(tmp_path)

    fetcher = APIFetcher()

    assert isinstance(fetcher.dirs, dict)
    expected_keys = {"historic", "current", "future", "temp"}
    assert expected_keys.issubset(fetcher.dirs.keys())

    for path in fetcher.dirs.values():
        assert os.path.exists(path)
        assert os.path.isdir(path)
