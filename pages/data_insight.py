from __future__ import annotations

import glob
import io
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

_DIRS = {
    "historic": os.path.join(_ROOT, "data", "raw", "historic"),
    "current": os.path.join(_ROOT, "data", "raw", "current"),
    "future": os.path.join(_ROOT, "data", "raw", "future"),
    "processed_historic": os.path.join(_ROOT, "data", "processed"),
    "processed_forecast": os.path.join(_ROOT, "data", "processed", "forecast"),
}

_CLR = {
    "primary": "#4A90D9",
    "rain": "#2196F3",
    "no_rain": "#FF9800",
    "temp": "#E53935",
    "humidity": "#43A047",
    "wind": "#8E24AA",
    "pressure": "#00838F",
    "cloud": "#78909C",
}

_COLS = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure", "Rain"]
_COL_COLORS = {
    "Temperature": _CLR["temp"],
    "Humidity": _CLR["humidity"],
    "Wind_Speed": _CLR["wind"],
    "Cloud_Cover": _CLR["cloud"],
    "Pressure": _CLR["pressure"],
    "Rain": _CLR["rain"],
}
_COL_UNITS = {
    "Temperature": "°C",
    "Humidity": "%",
    "Wind_Speed": "km/h",
    "Cloud_Cover": "%",
    "Pressure": "hPa",
    "Rain": "(0/1)",
}


def _list_csvs(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "*.csv")))


def _load_concat(paths: list[str]) -> pd.DataFrame | None:
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else None


def _parse_time_col(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Fetch_Time", "fetch_time", "Forecast_Time", "timestamp"]:
        if col in df.columns:
            try:
                df["_time"] = pd.to_datetime(df[col])
                return df
            except Exception:
                pass
    return df


@st.cache_data(ttl=300, show_spinner=False)
def _load_tier(tier: str) -> pd.DataFrame | None:
    directory = _DIRS.get(tier, "")
    paths = _list_csvs(directory)

    # Forecast tiers: use only the latest file to keep memory low
    if tier in ("future", "processed_forecast"):
        if not paths:
            return None
        latest = max(paths, key=os.path.getmtime)
        try:
            df = pd.read_csv(latest)
        except Exception:
            return None
        df = _parse_time_col(df)
        return df

    df = _load_concat(paths)
    if df is None:
        return df

    df = _parse_time_col(df)

    # Historic / current tiers: apply 48-hour rolling window
    if tier in ("historic", "current") and "_time" in df.columns:
        max_ts = df["_time"].max()
        cutoff = max_ts - pd.Timedelta(hours=48)
        df = df[df["_time"] >= cutoff].reset_index(drop=True)

    return df


def _make_dummy_current(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    lats = np.linspace(-6.36, -6.00, 20)
    lons = np.linspace(106.33, 106.77, 20)
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    now = datetime.now()
    df = pd.DataFrame(
        {
            "Fetch_Time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude": lat_grid.ravel(),
            "Longitude": lon_grid.ravel(),
            "Temperature": rng.uniform(24, 34, n),
            "Humidity": rng.uniform(55, 95, n),
            "Wind_Speed": rng.uniform(0, 25, n),
            "Cloud_Cover": rng.uniform(0, 100, n),
            "Pressure": rng.uniform(1005, 1015, n),
            "Rain": rng.integers(0, 2, n),
        }
    )
    df["_time"] = pd.to_datetime(df["Fetch_Time"])
    return df


def _make_dummy_historic(n_snapshots: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    base = pd.Timestamp("2025-01-01 06:00")
    for i in range(n_snapshots):
        ts = base + pd.Timedelta(hours=i * 6)
        rows.append(
            {
                "Fetch_Time": ts,
                "Temperature": 27 + 4 * np.sin(i * 0.4) + rng.normal(0, 0.8),
                "Humidity": 75 + 12 * np.cos(i * 0.3) + rng.normal(0, 2),
                "Wind_Speed": abs(8 + 5 * np.sin(i * 0.2) + rng.normal(0, 1)),
                "Cloud_Cover": np.clip(
                    50 + 30 * np.sin(i * 0.5) + rng.normal(0, 5), 0, 100
                ),
                "Pressure": 1010 + 2 * np.sin(i * 0.15) + rng.normal(0, 0.3),
                "Rain": int(rng.random() < (0.4 + 0.3 * np.sin(i * 0.5))),
            }
        )
    df = pd.DataFrame(rows)
    df["_time"] = pd.to_datetime(df["Fetch_Time"])
    return df


def _make_dummy_future(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    base = pd.Timestamp.now().floor("h")
    rows = []
    for i in range(n):
        ts = base + pd.Timedelta(hours=i)
        rows.append(
            {
                "Forecast_Time": ts,
                "Temperature": 28 + 3 * np.sin(i * 0.4) + rng.normal(0, 0.5),
                "Humidity": 72 + 10 * np.cos(i * 0.35) + rng.normal(0, 1.5),
                "Wind_Speed": abs(7 + 4 * np.sin(i * 0.25) + rng.normal(0, 0.8)),
                "Cloud_Cover": np.clip(
                    45 + 25 * np.sin(i * 0.5) + rng.normal(0, 4), 0, 100
                ),
                "Pressure": 1009 + 1.5 * np.sin(i * 0.2) + rng.normal(0, 0.2),
                "Rain": int(rng.random() < 0.35),
            }
        )
    df = pd.DataFrame(rows)
    df["_time"] = pd.to_datetime(df["Forecast_Time"])
    return df


def _summary_metrics(df: pd.DataFrame) -> None:
    items = [
        ("Avg Temp", "Temperature", "°C", False),
        ("Avg Humidity", "Humidity", "%", False),
        ("Avg Wind", "Wind_Speed", "km/h", False),
        ("Avg Cloud", "Cloud_Cover", "%", False),
        ("Rain Rate", "Rain", "", True),
    ]
    mcols = st.columns(len(items))
    for mc, (label, col, unit, pct) in zip(mcols, items):
        if col in df.columns:
            val = df[col].mean()
            mc.metric(label, f"{val * 100:.1f}%" if pct else f"{val:.1f} {unit}")
        else:
            mc.metric(label, "N/A")


def _time_series(df: pd.DataFrame, col: str, container) -> None:
    if col not in df.columns or "_time" not in df.columns:
        return
    plot_df = df.groupby("_time")[col].mean().reset_index()
    container.line_chart(plot_df.set_index("_time"), color=_COL_COLORS.get(col))


def _rain_bar(df: pd.DataFrame, container) -> None:
    """Simple blue bar chart for rain distribution."""
    if "Rain" not in df.columns:
        return
    counts = df["Rain"].value_counts().sort_index()
    bar_df = pd.DataFrame(
        {
            "Condition": [("No Rain" if k == 0 else "Rain") for k in counts.index],
            "Count": counts.values,
        }
    )
    container.bar_chart(bar_df.set_index("Condition"), color=_CLR["rain"])


def _rain_by_hour_line(df: pd.DataFrame, container) -> None:
    if "Rain" not in df.columns or "_time" not in df.columns:
        return
    tmp = df.copy()
    tmp["Hour"] = tmp["_time"].dt.hour
    grouped = (
        tmp.groupby("Hour")["Rain"]
        .mean()
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    grouped.columns = ["Hour", "Rain Probability (%)"]
    grouped["Rain Probability (%)"] = (grouped["Rain Probability (%)"] * 100).round(1)
    container.line_chart(grouped.set_index("Hour"), color=_CLR["rain"])


def _feature_stats_table(df: pd.DataFrame, container) -> None:
    num_cols = [c for c in _COLS if c in df.columns and c != "Rain"]
    if not num_cols:
        return
    stats = df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
    stats.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
    stats.index.name = "Feature"
    # add unit column
    stats.insert(0, "Unit", [_COL_UNITS.get(c, "") for c in stats.index])
    container.dataframe(stats, use_container_width=True)


def _correlation_table(df: pd.DataFrame, container) -> None:
    num_cols = [c for c in _COLS if c in df.columns]
    if len(num_cols) < 2:
        return
    corr = df[num_cols].corr().round(2)
    container.dataframe(
        corr.style.background_gradient(cmap="Blues", axis=None),
        use_container_width=True,
    )


def _forecast_rain_timeseries(df: pd.DataFrame, container) -> None:
    if "Rain" not in df.columns or "_time" not in df.columns:
        return
    tmp = df.copy()
    tmp["Hour"] = tmp["_time"].dt.floor("h")
    grouped = (
        tmp.groupby("Hour")
        .agg(
            Rain_Rate=("Rain", "mean"),
            Rainy_Points=("Rain", "sum"),
            Total_Points=("Rain", "count"),
        )
        .reset_index()
    )
    grouped["Rain Rate (%)"] = (grouped["Rain_Rate"] * 100).round(1)

    container.markdown("**Rain Rate Over Forecast Window (% of grid points)**")
    container.line_chart(grouped.set_index("Hour")["Rain Rate (%)"], color=_CLR["rain"])

    container.markdown("**Rainy Grid Points per Forecast Hour**")
    container.bar_chart(grouped.set_index("Hour")["Rainy_Points"], color=_CLR["rain"])


def _to_csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _export_expander(df: pd.DataFrame | None, label: str, filename: str) -> None:
    with st.expander(f"Export {label} Data as CSV", expanded=False):
        if df is None or df.empty:
            st.info("No data available.")
            return
        export_df = df.drop(columns=["_time"], errors="ignore")
        st.write(f"**{len(export_df):,} rows · {len(export_df.columns)} columns**")
        st.dataframe(export_df.head(10), use_container_width=True)
        st.download_button(
            label=f"Download {label} CSV",
            data=_to_csv(export_df),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )


def _section_current() -> None:
    df = _load_tier("current")
    dummy = df is None or df.empty
    if dummy:
        df = _make_dummy_current()

    badge = "  *(dummy — run fetcher first)*" if dummy else ""
    st.markdown(f"### Current Weather Snapshot{badge}")
    st.caption("Latest fetched grid snapshot across the 20×20 Tangerang grid.")

    _summary_metrics(df)

    st.markdown("**Rain Distribution**")
    _rain_bar(df, st)

    st.markdown("**Feature Summary Statistics**")
    _feature_stats_table(df, st)

    _export_expander(
        df,
        "Current",
        f"tangerangcast_current_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    )
    st.divider()


def _section_historic() -> None:
    df = _load_tier("historic")
    dummy = df is None or df.empty
    if dummy:
        df = _make_dummy_historic()

    badge = "  *(dummy — add CSVs to data/raw/historic/)*" if dummy else ""
    st.markdown(f"### Historic Trends{badge}")
    st.caption("Time-series trends from past weather snapshots.")

    _summary_metrics(df)

    st.markdown("**Variable Trends Over Time**")
    t1, t2 = st.columns(2)
    for feat, container in [
        ("Temperature", t1),
        ("Humidity", t2),
        ("Wind_Speed", t1),
        ("Cloud_Cover", t2),
    ]:
        container.markdown(f"*{feat} ({_COL_UNITS[feat]})*")
        _time_series(df, feat, container)

    st.markdown("**Rain Probability by Hour of Day (0–23)**")
    _rain_by_hour_line(df, st)

    st.markdown("**Feature Correlation Matrix**")
    _correlation_table(df, st)

    _export_expander(
        df,
        "Historic",
        f"tangerangcast_historic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    )
    st.divider()


def _section_future() -> None:
    df = _load_tier("future")
    dummy = df is None or df.empty
    if dummy:
        df = _make_dummy_future()

    badge = "  *(dummy — run fetcher to populate)*" if dummy else ""
    st.markdown(f"### Forecast Trends{badge}")
    st.caption("Next-24h Open-Meteo forecast variables.")

    _summary_metrics(df)

    st.markdown("**Forecast Variable Trends**")
    fc1, fc2 = st.columns(2)
    for feat, container in [
        ("Temperature", fc1),
        ("Humidity", fc2),
        ("Cloud_Cover", fc1),
        ("Wind_Speed", fc2),
    ]:
        container.markdown(f"*{feat} ({_COL_UNITS[feat]})*")
        _time_series(df, feat, container)

    _forecast_rain_timeseries(df, st)

    _export_expander(
        df,
        "Forecast",
        f"tangerangcast_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    )
    st.divider()


def _section_bulk_export() -> None:
    st.markdown("### Bulk Export")
    st.caption("Select individual files to include, then download as a ZIP.")

    tier_map = {
        "current": "Current",
        "historic": "Historic",
        "future": "Forecast",
    }

    all_files: list[dict] = []
    for tier, label in tier_map.items():
        paths = _list_csvs(_DIRS.get(tier, ""))
        for p in paths:
            all_files.append({"tier": label, "path": p, "name": os.path.basename(p)})

    if not all_files:
        st.info("No CSV files found yet. Run the data fetcher first.")
        return

    st.markdown(f"**{len(all_files)} file(s) available across all tiers**")

    selected_paths: list[str] = []
    for tier_label in ["Current", "Historic", "Forecast"]:
        tier_files = [f for f in all_files if f["tier"] == tier_label]
        if not tier_files:
            continue
        with st.expander(f"📂 {tier_label} ({len(tier_files)} file(s))", expanded=True):
            for f in tier_files:
                checked = st.checkbox(f["name"], value=True, key=f"export_{f['path']}")
                if checked:
                    selected_paths.append(f["path"])

    st.markdown(f"**{len(selected_paths)} file(s) selected**")

    if selected_paths:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in selected_paths:
                try:
                    zf.write(p, arcname=os.path.basename(p))
                except Exception:
                    pass
        zip_buf.seek(0)

        st.download_button(
            label=f"⬇️ Download {len(selected_paths)} file(s) as ZIP",
            data=zip_buf.getvalue(),
            file_name=f"tangerangcast_export_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.warning("Select at least one file to enable download.")


st.set_page_config(page_title="Data Insight", page_icon="📊", layout="wide")

with st.sidebar:
    st.markdown("## Data Insight")
    st.markdown("Explore weather trends and export raw data from each pipeline tier.")
    st.divider()
    st.markdown("**Show Sections**")
    show_current = st.checkbox("Current Snapshot", value=True)
    show_historic = st.checkbox("Historic Trends", value=True)
    show_future = st.checkbox("Forecast Trends", value=True)
    st.divider()
    st.info(
        "Note: Sections without real data display dummy visualisations "
        "until the fetcher has been run at least once."
    )

st.markdown("# Data Insight Dashboard")
st.markdown(
    "Visualisation and export hub for TangerangCast data tiers: "
    "**current**, **historic**, and **forecast**."
)
st.divider()

with st.expander("Data Availability", expanded=True):
    bcols = st.columns(3)
    for col, (tier, label) in zip(
        bcols,
        [("current", "Current"), ("historic", "Historic"), ("future", "Forecast")],
    ):
        paths = _list_csvs(_DIRS.get(tier, ""))
        if paths:
            col.success(f"**{label}** — {len(paths)} file(s)")
        else:
            col.warning(f"**{label}** — no data (dummy mode)")

st.divider()

if show_current:
    _section_current()

if show_historic:
    _section_historic()

if show_future:
    _section_future()

_section_bulk_export()
