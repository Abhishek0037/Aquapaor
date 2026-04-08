"""
Streamlit dashboard: Aquapor.
"""

import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from model import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    REGION_COLUMN,
    filter_by_region,
    load_data,
    available_regions,
    fit_and_evaluate,
    make_predictions,
    model_debug_info,
    predict_next,
    to_download_csv,
)

DATA_PATH = Path(__file__).resolve().parent / "data.csv"

LOW_WATER_THRESHOLD = 75.0
HIGH_WATER_THRESHOLD = 85.0


def compute_trend(values: pd.Series) -> tuple[str, str, float]:
    """
    Compute trend label from a time-ordered series.

    Returns: (label, icon, slope)
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 3:
        return "Stable", "📍", 0.0

    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0])
    spread = float(np.nanstd(arr))
    eps = max(0.05, 0.02 * spread)

    if abs(slope) <= eps:
        return "Stable", "📍", slope
    if slope > 0:
        return "Increasing", "📈", slope
    return "Decreasing", "📉", slope


def water_recommendations(future_preds: np.ndarray) -> list[tuple[str, str, str]]:
    """
    Returns list of (title, detail, color) callouts based on next-7-days predictions.
    """
    preds = np.asarray(future_preds, dtype=float)
    preds = preds[np.isfinite(preds)]
    if preds.size == 0:
        return [("No prediction available", "Not enough data to generate future predictions.", "#6b7280")]

    future_min = float(np.min(preds))
    future_max = float(np.max(preds))

    drought = future_min < LOW_WATER_THRESHOLD
    flood = future_max > HIGH_WATER_THRESHOLD

    recs: list[tuple[str, str, str]] = []
    if drought:
        recs.append(
            (
                "Drought Risk",
                f"Projected low in next 7 days is {future_min:.2f} (< {LOW_WATER_THRESHOLD:.0f}). Reduce irrigation and conserve water.",
                "#ef4444",
            )
        )
    if flood:
        recs.append(
            (
                "Flood Risk",
                f"Projected high in next 7 days is {future_max:.2f} (> {HIGH_WATER_THRESHOLD:.0f}). Prepare storage and flood control measures.",
                "#f59e0b",
            )
        )
    if not recs:
        recs.append(
            (
                "Water Levels Stable",
                f"Projected range stays within {LOW_WATER_THRESHOLD:.0f}–{HIGH_WATER_THRESHOLD:.0f}. Maintain current usage.",
                "#22c55e",
            )
        )
    return recs


def callout(title: str, detail: str, color: str) -> None:
    """Render a prominent color-coded callout."""
    st.markdown(
        f"""
        <div style="background:{color}; padding:14px 16px; border-radius:12px; color:white; margin-top:8px; margin-bottom:8px;">
          <div style="font-size:18px; font-weight:800; line-height:1.2;">{title}</div>
          <div style="font-size:14px; opacity:0.95; margin-top:6px; white-space:pre-wrap;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert_message(level: float) -> tuple[str, str]:
    """Return (label, short explanation) for display."""
    if level < LOW_WATER_THRESHOLD:
        return "Drought Risk", "Predicted water level is below the safe lower threshold (75)."
    if level > HIGH_WATER_THRESHOLD:
        return "Flood Risk", "Predicted water level is above the safe upper threshold (85)."
    return "Water Levels Stable", "Predicted water level is within the normal range (75–85)."


@st.cache_data(show_spinner=False)
def load_from_upload(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return load_data(str(DATA_PATH))


@st.cache_resource(show_spinner=False)
def train_cached(df: pd.DataFrame):
    # cache key is based on df content hash computed by streamlit
    return fit_and_evaluate(df)


def main() -> None:
    st.set_page_config(page_title="Aquapor", layout="wide")
    st.title("Aquapor")
    st.caption("Forecast water levels from rainfall and temperature.")

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return

    st.sidebar.header("Controls")
    data_mode = st.sidebar.radio("Data source", ["Use sample data", "Upload CSV"], index=0)
    uploaded = None
    if data_mode == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    try:
        if uploaded is not None:
            raw_df = load_from_upload(uploaded.getvalue())
        else:
            raw_df = load_sample()
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    regions = available_regions(raw_df)
    region = st.sidebar.selectbox("Region", regions, index=0)
    df_region = filter_by_region(raw_df, region)

    if df_region.empty:
        st.warning("No rows after filtering. Try a different region or CSV.")
        return

    # Date range filter
    try:
        df_region[DATE_COLUMN] = pd.to_datetime(df_region[DATE_COLUMN], errors="coerce")
        df_region = df_region.dropna(subset=[DATE_COLUMN]).copy()
    except Exception:
        pass

    min_date = df_region[DATE_COLUMN].min()
    max_date = df_region[DATE_COLUMN].max()
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("Date column could not be parsed. Ensure the CSV has a valid Date column.")
        return

    dates = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if len(dates) != 2:
        st.warning("Please select an end date.")
        return
    start_date, end_date = dates
    mask = (df_region[DATE_COLUMN].dt.date >= start_date) & (df_region[DATE_COLUMN].dt.date <= end_date)
    df_view = df_region.loc[mask].copy()
    df_view = df_view.sort_values(DATE_COLUMN).reset_index(drop=True)

    if df_view.empty:
        st.warning("No rows in the selected date range.")
        return

    try:
        result = train_cached(df_view)
    except Exception as e:
        st.error(str(e))
        st.info("Expected CSV columns: Date, Rainfall, Temperature, Water_Level (Region optional).")
        return

    pred_df = make_predictions(df_view, result.model)
    latest_pred = float(pred_df["Predicted_Water_Level"].iloc[-1])

    # Future 7-day forecast computed once, reused across Alerts/Forecast/Insights.
    last_date = pd.to_datetime(pred_df[DATE_COLUMN].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq="D")
    rain_series = pd.to_numeric(df_view["Rainfall"], errors="coerce")
    temp_series = pd.to_numeric(df_view["Temperature"], errors="coerce")

    base_rain = float(rain_series.median()) if rain_series.notna().any() else 0.0
    base_temp = float(temp_series.median()) if temp_series.notna().any() else 0.0

    # Extend a simple recent trend for features (last up to 3 points).
    def recent_slope(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce").tail(3).dropna().values
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals), dtype=float)
        return float(np.polyfit(x, vals, 1)[0])

    slope_rain = recent_slope(rain_series)
    slope_temp = recent_slope(temp_series)

    last_rain = float(rain_series.dropna().iloc[-1]) if rain_series.dropna().shape[0] else base_rain
    last_temp = float(temp_series.dropna().iloc[-1]) if temp_series.dropna().shape[0] else base_temp

    day_idx = np.arange(1, 8, dtype=float)
    rain_future = last_rain + slope_rain * day_idx
    temp_future = last_temp + slope_temp * day_idx

    # Clamp to realistic ranges for the UI sliders (keeps plots/predictions stable).
    rain_future = np.clip(rain_future, 0.0, 300.0)
    temp_future = np.clip(temp_future, 0.0, 45.0)

    future_features = pd.DataFrame({"Rainfall": rain_future, "Temperature": temp_future})
    future_pred = result.model.predict(future_features)
    future_df = pd.DataFrame({DATE_COLUMN: future_dates, "Predicted_Water_Level": future_pred})
    future_trend_label, future_trend_icon, _ = compute_trend(pd.Series(future_pred))
    future_recs = water_recommendations(future_pred)
    future_min = float(np.min(future_pred))
    future_max = float(np.max(future_pred))

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest predicted", f"{latest_pred:.2f}")
    actual_latest = float(pred_df["Actual_Water_Level"].iloc[-1]) if not pd.isna(pred_df["Actual_Water_Level"].iloc[-1]) else None
    k2.metric("Latest actual", f"{actual_latest:.2f}" if actual_latest is not None else "—")
    k3.metric("MAE (hold-out)", f"{result.metrics['mae']:.2f}" if result.metrics["mae"] is not None else "—")
    k4.metric("R² (hold-out)", f"{result.metrics['r2']:.3f}" if result.metrics["r2"] is not None else "—")

    st.divider()

    st.subheader("🚨 Alerts")
    st.caption("Risk assessment based on the next 7 days predicted range.")

    risk_items: list[tuple[str, str, str]] = []
    if future_min < LOW_WATER_THRESHOLD:
        risk_items.append(
            ("Drought Risk", f"Next 7 days lowest predicted level: {future_min:.2f} (< {LOW_WATER_THRESHOLD:.0f}).", "#ef4444")
        )
    if future_max > HIGH_WATER_THRESHOLD:
        risk_items.append(
            ("Flood Risk", f"Next 7 days highest predicted level: {future_max:.2f} (> {HIGH_WATER_THRESHOLD:.0f}).", "#f59e0b")
        )
    if not risk_items:
        risk_items.append(
            (
                "Water Levels Stable",
                f"Next 7 days predicted range stays within {LOW_WATER_THRESHOLD:.0f}–{HIGH_WATER_THRESHOLD:.0f}.",
                "#22c55e",
            )
        )

    for title, detail, color in risk_items:
        callout(title, detail, color)

    st.caption(
        f"Latest predicted (history): {latest_pred:.2f}. Next 7 days predicted range: {future_min:.2f}–{future_max:.2f}."
    )

    tab_forecast, tab_analysis, tab_insights, tab_data, tab_whatif = st.tabs(
        ["🔮 Forecast", "📊 Data Analysis", "Insights", "Explore data", "What-if"]
    )

    with tab_forecast:
        st.subheader("🔮 Forecast")

        st.caption("History + next-7-days future forecast. Future rainfall/temperature are assumed as medians of your selected range.")

        st.divider()
        st.subheader("History (Actual vs Predicted)")
        history_df = pred_df.copy()
        history_chart = (
            alt.Chart(history_df)
            .mark_line(color="blue")
            .encode(
                x=alt.X(f"{DATE_COLUMN}:T", title="Date"),
                y=alt.Y("Predicted_Water_Level:Q", title="Water level"),
                tooltip=[DATE_COLUMN, "Predicted_Water_Level:Q"],
            )
        )
        if "Actual_Water_Level" in history_df.columns and not history_df["Actual_Water_Level"].isna().all():
            actual_chart = (
                alt.Chart(history_df.dropna(subset=["Actual_Water_Level"]))
                .mark_line(color="gray", opacity=0.6)
                .encode(
                    x=alt.X(f"{DATE_COLUMN}:T"),
                    y=alt.Y("Actual_Water_Level:Q", title="Water level"),
                    tooltip=[DATE_COLUMN, "Actual_Water_Level:Q"],
                )
            )
            history_chart = history_chart + actual_chart
        st.altair_chart(history_chart.properties(height=260), width="stretch")

        st.divider()
        st.subheader("Next 7 days (Future) — trend highlighted")
        future_color = "green" if future_trend_label == "Increasing" else ("red" if future_trend_label == "Decreasing" else "gray")
        future_chart = alt.Chart(future_df).mark_line(strokeWidth=4, color=future_color).encode(
            x=alt.X(f"{DATE_COLUMN}:T", title="Date"),
            y=alt.Y("Predicted_Water_Level:Q", title="Predicted water level"),
            tooltip=[DATE_COLUMN, "Predicted_Water_Level:Q"],
        )
        future_points = alt.Chart(future_df.tail(2)).mark_circle(size=100, color=future_color).encode(
            x=alt.X(f"{DATE_COLUMN}:T"),
            y=alt.Y("Predicted_Water_Level:Q"),
        )
        st.altair_chart((future_chart + future_points).properties(height=240), width="stretch")
        st.caption(f"Future trend: {future_trend_label} {future_trend_icon}")

        st.divider()
        st.subheader("Decision Recommendations")
        for title, detail, color in future_recs:
            st.markdown(f"### {title}")
            st.write(detail)

        st.download_button(
            "Download history + future predictions (CSV)",
            data=to_download_csv(
                pd.concat([pred_df, future_df.assign(Actual_Water_Level=np.nan)], ignore_index=True)[
                    [DATE_COLUMN, "Actual_Water_Level", "Predicted_Water_Level"]
                ]
            ),
            file_name="water_level_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

        with st.expander("Model details"):
            info = model_debug_info(result.model)
            st.write(f"Features: {FEATURE_COLUMNS}")
            st.write(f"Region mode: {'enabled' if REGION_COLUMN in raw_df.columns else 'not present in CSV'}")
            st.write(f"Rows used: {len(df_view)}")
            if info:
                st.write("Coefficients:", info["coefficients"])
                st.write(f"Intercept: {info['intercept']:.4f}")

    with tab_analysis:
        st.subheader("📊 Data Analysis")

        # Water Stress Index
        wsi = pd.to_numeric(df_view["Rainfall"], errors="coerce") - (pd.to_numeric(df_view["Temperature"], errors="coerce") * 0.3)
        df_wsi = df_view[[DATE_COLUMN, "Rainfall", "Temperature", "Water_Level"]].copy()
        df_wsi["Water_Stress_Index"] = wsi
        latest_wsi = float(df_wsi["Water_Stress_Index"].iloc[-1])
        st.metric("Water Stress Index (latest)", f"{latest_wsi:.2f}")
        st.line_chart(df_wsi.set_index(DATE_COLUMN)["Water_Stress_Index"])

        st.divider()

        # Scatter plots with trend line
        scatter_base = alt.Chart(df_wsi.dropna()).properties(height=260)
        rain_scatter = scatter_base.mark_circle(size=80, opacity=0.7).encode(
            x=alt.X("Rainfall:Q"),
            y=alt.Y("Water_Level:Q", title="Water Level"),
            tooltip=[DATE_COLUMN, "Rainfall", "Temperature", "Water_Level"],
        )
        rain_trend = rain_scatter.transform_regression("Rainfall", "Water_Level").mark_line(color="white")
        st.subheader("Rainfall vs Water Level")
        st.altair_chart((rain_scatter + rain_trend), width="stretch")

        temp_scatter = scatter_base.mark_circle(size=80, opacity=0.7).encode(
            x=alt.X("Temperature:Q"),
            y=alt.Y("Water_Level:Q", title="Water Level"),
            tooltip=[DATE_COLUMN, "Rainfall", "Temperature", "Water_Level"],
        )
        temp_trend = temp_scatter.transform_regression("Temperature", "Water_Level").mark_line(color="white")
        st.subheader("Temperature vs Water Level")
        st.altair_chart((temp_scatter + temp_trend), width="stretch")

        st.subheader("Correlation heatmap")
        corr_cols = ["Rainfall", "Temperature", "Water_Level", "Water_Stress_Index"]
        corr = df_wsi[corr_cols].corr(numeric_only=True).round(2)
        corr_long = corr.reset_index().melt(id_vars="index", var_name="Variable", value_name="Correlation").rename(
            columns={"index": "Index"}
        )
        heat = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x=alt.X("Variable:N", title=""),
                y=alt.Y("Index:N", title=""),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                tooltip=["Index", "Variable", "Correlation"],
            )
            .properties(height=260)
        )
        text = alt.Chart(corr_long).mark_text(baseline="middle").encode(
            x="Variable:N", y="Index:N", text="Correlation:Q"
        )
        st.altair_chart((heat + text), width="stretch")

    with tab_insights:
        st.subheader("Insights")
        df_i = df_view.copy()
        for c in ["Rainfall", "Temperature", "Water_Level"]:
            df_i[c] = pd.to_numeric(df_i[c], errors="coerce")
        df_i["Water_Stress_Index"] = df_i["Rainfall"] - (df_i["Temperature"] * 0.3)
        df_i = df_i.sort_values(DATE_COLUMN)

        rain_trend_label, rain_trend_icon, _ = compute_trend(df_i["Rainfall"])
        temp_trend_label, temp_trend_icon, _ = compute_trend(df_i["Temperature"])
        wl_trend_label, wl_trend_icon, _ = compute_trend(df_i["Water_Level"])

        st.info(f"Trend: {wl_trend_label} {wl_trend_icon}")

        bullets: list[str] = []
        if rain_trend_label == "Increasing":
            bullets.append("Rainfall is increasing, so water levels are likely to increase too.")
        elif rain_trend_label == "Decreasing":
            bullets.append("Rainfall is decreasing, so water levels are likely to decrease.")
        else:
            bullets.append("Rainfall is stable, so water levels are likely to stay steadier.")

        if temp_trend_label == "Increasing":
            bullets.append("Temperature is increasing, which tends to reduce water levels.")
        elif temp_trend_label == "Decreasing":
            bullets.append("Temperature is decreasing, which can support higher water levels.")
        else:
            bullets.append("Temperature is stable, so temperature-driven changes are limited.")

        bullets.append(f"Water level trend (historical): {wl_trend_label} {wl_trend_icon}")

        latest_wsi = float(df_i["Water_Stress_Index"].iloc[-1]) if df_i["Water_Stress_Index"].notna().any() else np.nan
        if np.isfinite(latest_wsi):
            bullets.append(f"Latest Water Stress Index: {latest_wsi:.2f} (Rainfall - 0.3*Temperature).")

        st.markdown("\n".join([f"- {b}" for b in bullets[:4]]))
        st.caption("Insights are computed from your selected region + date range.")

        st.divider()
        st.subheader("Decision Recommendations")
        for title, detail, color in future_recs:
            callout(title, detail, color)

    with tab_data:
        st.subheader("Filtered data preview")
        cols_to_show = [c for c in [DATE_COLUMN, *FEATURE_COLUMNS, "Water_Level", REGION_COLUMN] if c in df_view.columns]
        st.dataframe(df_view[cols_to_show].sort_values(DATE_COLUMN), width="stretch")
        st.subheader("Summary")
        st.write(df_view[FEATURE_COLUMNS + ["Water_Level"]].describe())

    with tab_whatif:
        st.subheader("What-if: predict from custom inputs")
        st.caption("Adjust rainfall and temperature to see the predicted water level using the trained model.")
        default_rain = float(pd.to_numeric(df_view["Rainfall"], errors="coerce").median())
        default_temp = float(pd.to_numeric(df_view["Temperature"], errors="coerce").median())

        if np.isnan(default_rain):
            default_rain = 0.0
        if np.isnan(default_temp):
            default_temp = 0.0

        c1, c2 = st.columns(2)
        rainfall = c1.slider("Rainfall", min_value=0.0, max_value=300.0, value=float(default_rain), step=1.0)
        temperature = c2.slider("Temperature", min_value=0.0, max_value=45.0, value=float(default_temp), step=0.5)

        predicted = predict_next(result.model, rainfall=rainfall, temperature=temperature)
        st.metric("What-if predicted water level", f"{predicted:.2f}")
        s2, d2 = alert_message(predicted)
        st.info(f"**{s2}** — {d2}")


if __name__ == "__main__":
    main()
