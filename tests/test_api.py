import os
import shutil
from src.api_fetcher import APIFetcher


def test_api_fetcher_initialization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Copy GeoJSON asset into the sandbox so _load_polygon can find it
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_geojson = os.path.join(project_root, "assets", "tangerang_border.geojson")
    dst_assets = os.path.join(tmp_path, "assets")
    os.makedirs(dst_assets, exist_ok=True)
    shutil.copy2(src_geojson, dst_assets)

    # Temporarily override the class-level path to point at the sandbox
    monkeypatch.setattr(APIFetcher, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        APIFetcher,
        "_BORDER_GEOJSON",
        os.path.join(str(tmp_path), "assets", "tangerang_border.geojson"),
    )

    fetcher = APIFetcher()

    assert isinstance(fetcher.dirs, dict)
    expected_keys = {"historic", "current", "future", "temp"}
    assert expected_keys.issubset(fetcher.dirs.keys())

    for path in fetcher.dirs.values():
        assert os.path.exists(path)
        assert os.path.isdir(path)

    # Verify polygon filtering: coords should be fewer than the full 20x20 = 400 grid
    assert len(fetcher.coords) > 0, "No coordinates survived polygon filtering"
    assert len(fetcher.coords) < 400, (
        f"Expected filtered coords < 400, got {len(fetcher.coords)}"
    )
