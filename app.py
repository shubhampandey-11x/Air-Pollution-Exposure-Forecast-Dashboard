import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="🌫️ AI-Powered AQI Forecast Dashboard",
    page_icon="😷",
    layout="wide",
)

# -------------------------------------------------
# DARK THEME BASE STYLING
# -------------------------------------------------
DARK_CSS = """
<style>
body, .stApp {
    background-color: #020617;
    color: #e5e7eb;
}

header, .css-18ni7ap, .css-1avcm0n {
    background-color: #020617 !important;
}

.block-container {
    padding-top: 1.5rem;
}

h1, h2, h3, h4, h5 {
    color: #e5e7eb !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

plot_template = "plotly_dark"

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
FEATURE_COLS_DEFAULT = [
    "latitude",
    "longitude",
    "pollutant_min",
    "pollutant_max",
    "hour",
    "day",
    "month",
    "weekday",
    "is_weekend",
    "dist_from_center",
    "city",
    "state",
    "station",
]

CATEGORICAL_COLS = ["city", "state", "station"]


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    # Load model from main or models/ folder
    try:
        saved = joblib.load("aq_model.pkl")
    except:
        saved = joblib.load("models/aq_model.pkl")

    # If saved as (model, feature_cols)
    if isinstance(saved, tuple) and len(saved) == 2:
        model, feat_cols = saved
        if isinstance(feat_cols, (list, tuple)) and len(feat_cols) > 0:
            feature_cols = list(feat_cols)
        else:
            # Try reading from feature_columns.txt
            try:
                with open("models/feature_columns.txt", "r") as f:
                    feature_cols = f.read().strip().split(",")
            except:
                feature_cols = FEATURE_COLS_DEFAULT
    else:
        model = saved
        # Prefer feature_columns.txt if present
        try:
            with open("models/feature_columns.txt", "r") as f:
                feature_cols = f.read().strip().split(",")
        except:
            feature_cols = FEATURE_COLS_DEFAULT

    return model, feature_cols


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = None
    for fname in [
        "air_quality_expanded_synthetic.csv",
        "air_quality_expanded.csv",
        "air_quality.csv",
    ]:
        try:
            df = pd.read_csv(fname)
            break
        except Exception:
            df = None

    if df is None:
        st.error("❌ Could not find air quality CSV file.")
        st.stop()

    df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
    df = df.dropna(subset=["last_update"])

    # Time features
    df["hour"] = df["last_update"].dt.hour
    df["day"] = df["last_update"].dt.day
    df["month"] = df["last_update"].dt.month
    df["weekday"] = df["last_update"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Spatial feature
    df["dist_from_center"] = np.sqrt(
        (df["latitude"] - 22.0) ** 2 + (df["longitude"] - 79.0) ** 2
    )

    # Fill target-related NaNs
    df[["pollutant_min", "pollutant_max", "pollutant_avg"]] = (
        df[["pollutant_min", "pollutant_max", "pollutant_avg"]]
        .fillna(df[["pollutant_min", "pollutant_max", "pollutant_avg"]].median())
    )

    # Categorical cleanup
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].astype("category")

    return df


# -------------------------------------------------
# BUILD FUTURE DF (24 HOURS)
# -------------------------------------------------
def build_future_df(latest_row, hours=24):
    rows = []
    base_time = latest_row["last_update"]

    for h in range(1, hours + 1):
        t = base_time + pd.Timedelta(hours=h)
        row = {
            "latitude": latest_row["latitude"],
            "longitude": latest_row["longitude"],
            "pollutant_min": latest_row["pollutant_min"],
            "pollutant_max": latest_row["pollutant_max"],
            "hour": t.hour,
            "day": t.day,
            "month": t.month,
            "weekday": t.weekday(),
            "is_weekend": 1 if t.weekday() in [5, 6] else 0,
            "dist_from_center": np.sqrt(
                (latest_row["latitude"] - 22.0) ** 2
                + (latest_row["longitude"] - 79.0) ** 2
            ),
            "city": latest_row["city"],
            "state": latest_row["state"],
            "station": latest_row["station"],
            "future_time": t,
        }
        rows.append(row)

    future_df = pd.DataFrame(rows)

    for col in CATEGORICAL_COLS:
        future_df[col] = future_df[col].astype(str).str.strip()
        future_df[col] = future_df[col].astype("category")

    return future_df


# -------------------------------------------------
# AQI RATING + HEALTH TIPS
# -------------------------------------------------
def get_aqi_rating_and_suggestions(value):
    if value <= 50:
        rating = "Excellent 🟢"
        tips = [
            "Air quality is clean and fresh.",
            "Perfect time for outdoor exercise.",
            "Keep windows open for natural ventilation.",
        ]
    elif value <= 100:
        rating = "Good 🟡"
        tips = [
            "Air quality is acceptable.",
            "Slight caution for extremely sensitive individuals.",
        ]
    elif value <= 200:
        rating = "Moderate 🟠"
        tips = [
            "Avoid very intense outdoor workouts.",
            "Children & asthma patients should reduce exposure.",
        ]
    elif value <= 300:
        rating = "Poor 🔴"
        tips = [
            "Wear an N95 mask when going outdoors.",
            "Avoid running or heavy physical activity outside.",
            "Keep windows closed during peak traffic hours.",
        ]
    elif value <= 400:
        rating = "Very Poor 🟣"
        tips = [
            "Stay indoors as much as possible.",
            "Use an air purifier if available.",
            "Children, elderly & pregnant women must avoid exposure.",
        ]
    else:
        rating = "Hazardous ⚫"
        tips = [
            "Serious health risk. Avoid going outdoors.",
            "Follow government/health advisories.",
            "Use air purifier and tightly close windows.",
        ]
    return rating, tips


# -------------------------------------------------
# POLLUTANT SOURCE INTELLIGENCE
# -------------------------------------------------
def get_pollutant_source_info(pollutant_id):
    p = pollutant_id.upper()
    if p == "PM2.5":
        return "Fine particulate matter (PM2.5)", [
            "Likely sources: vehicle exhaust, biomass/garbage burning, diesel generators.",
            "These fine particles can deeply penetrate the lungs and bloodstream.",
        ]
    elif p == "PM10":
        return "Coarse particulate matter (PM10)", [
            "Likely sources: road dust, construction activity, unpaved areas.",
            "High PM10 suggests dusty environments or nearby civil work.",
        ]
    elif p == "NO2":
        return "Nitrogen Dioxide (NO₂)", [
            "Likely sources: vehicle emissions, traffic congestion.",
            "Strongly associated with dense traffic zones and fuel combustion.",
        ]
    elif p == "SO2":
        return "Sulfur Dioxide (SO₂)", [
            "Likely sources: coal-based power plants, industrial combustion.",
            "Can lead to respiratory irritation and contributes to acid rain.",
        ]
    elif p == "CO":
        return "Carbon Monoxide (CO)", [
            "Likely sources: incomplete combustion, vehicles, generators.",
            "Dangerous in closed spaces and high-traffic tunnels/parking areas.",
        ]
    elif p == "OZONE":
        return "Ground-level Ozone (O₃)", [
            "Formed when NOx and VOCs react in strong sunlight.",
            "Typically higher on hot, sunny, low-wind days.",
        ]
    elif p == "NH3":
        return "Ammonia (NH₃)", [
            "Likely sources: agriculture, fertilizers, waste and sewage.",
            "Can irritate eyes, nose, throat and affect those with lung issues.",
        ]
    else:
        return pollutant_id, [
            "Specific source mapping not configured.",
            "Check local emission patterns and industrial activities.",
        ]


# -------------------------------------------------
# PERSONAL EXPOSURE INDEX (PEI)
# -------------------------------------------------
def compute_pei(predicted_aqi, exposure_level, sensitivity_level):
    exposure_factor = {
        "Low (≤ 2 hours outside)": 0.6,
        "Medium (2–6 hours outside)": 1.0,
        "High (≥ 6 hours outside)": 1.4,
    }[exposure_level]

    sensitivity_factor = {
        "Normal / Healthy": 1.0,
        "Asthma / Allergies": 1.3,
        "Elderly / Child / Pregnant": 1.5,
    }[sensitivity_level]

    pei = predicted_aqi * exposure_factor * sensitivity_factor

    if pei <= 80:
        label = "Low Personal Risk 🟢"
        msg = "Your overall exposure is low. Normal outdoor routine is okay."
    elif pei <= 160:
        label = "Moderate Personal Risk 🟠"
        msg = "Be a bit cautious. Limit outdoor time if you feel discomfort."
    elif pei <= 260:
        label = "High Personal Risk 🔴"
        msg = "Try to reduce outdoor time. Prefer indoor activities and wear a mask outside."
    else:
        label = "Very High Personal Risk ⚫"
        msg = "Avoid going out unless necessary. Take strong protection measures."

    return pei, label, msg


# -------------------------------------------------
# BEST & WORST TIME TO GO OUT
# -------------------------------------------------
def get_best_and_worst_time(future_df):
    idx_min = future_df["predicted_pollutant"].idxmin()
    idx_max = future_df["predicted_pollutant"].idxmax()

    best_time = future_df.loc[idx_min, "future_time"]
    best_value = future_df.loc[idx_min, "predicted_pollutant"]

    worst_time = future_df.loc[idx_max, "future_time"]
    worst_value = future_df.loc[idx_max, "predicted_pollutant"]

    return best_time, best_value, worst_time, worst_value


# -------------------------------------------------
# MODEL METRICS (for "Model Accuracy" Page)
# -------------------------------------------------
@st.cache_data
def compute_model_metrics(df, _model, feature_cols):
    target = "pollutant_avg"
    df_clean = df.dropna(subset=feature_cols + [target])

    n = len(df_clean)
    if n < 100:
        return None

    train_size = int(n * 0.8)
    test_df = df_clean.iloc[train_size:]

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target].copy()

    for col in CATEGORICAL_COLS:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # Predict with LightGBM/sklearn model
    y_pred = _model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Sample table
    results = pd.DataFrame(
        {"Actual": y_test.values[:100], "Predicted": y_pred[:100]}
    )

    feature_importance = None
    if hasattr(_model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": _model.feature_importances_}
        ).sort_values("importance", ascending=False)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "results_sample": results,
        "feature_importance": feature_importance,
    }


# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
def main():
    st.title("🌫️ AI-Powered AQI Forecast Dashboard")

    df = load_data()
    model, feature_cols = load_model()

    # ------- SIDEBAR NAVIGATION -------
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🌍 Air Quality Overview",
            "📈 Forecast",
            "🩺 Health Insights",
            "🌍 Pollution Sources",
            "📊 Model Accuracy",
            "⚙️ Settings",
        ],
    )

    # ------- SIDEBAR CONTROLS (shared) -------
    st.sidebar.markdown("### 🌍 Location & Pollutant")
    pollutant = st.sidebar.selectbox(
        "Pollutant", sorted(df["pollutant_id"].unique())
    )
    stations = df[df["pollutant_id"] == pollutant]["station"].unique()
    station = st.sidebar.selectbox("Station", sorted(stations))

    st.sidebar.markdown("### 👤 Personal Profile")
    exposure_level = st.sidebar.selectbox(
        "Daily outdoor exposure",
        [
            "Low (≤ 2 hours outside)",
            "Medium (2–6 hours outside)",
            "High (≥ 6 hours outside)",
        ],
        index=1,
    )
    sensitivity_level = st.sidebar.selectbox(
        "Health condition",
        [
            "Normal / Healthy",
            "Asthma / Allergies",
            "Elderly / Child / Pregnant",
        ],
        index=0,
    )

    # ------- FILTER DATA FOR SELECTED STATION -------
    station_df = df[
        (df["pollutant_id"] == pollutant) & (df["station"] == station)
    ].sort_values("last_update")

    if station_df.empty:
        st.warning("No data for this pollutant & station.")
        return

    latest = station_df.iloc[-1]

    # ------- BUILD FUTURE + PREDICT ONCE -------
    future_df = build_future_df(latest, hours=24)
    use_cols = [c for c in feature_cols if c in future_df.columns]
    X_future = future_df[use_cols].copy()
    for col in CATEGORICAL_COLS:
        if col in X_future.columns:
            X_future[col] = X_future[col].astype("category")

    preds = model.predict(X_future)
    future_df["predicted_pollutant"] = preds

    next_hour_val = future_df.iloc[0]["predicted_pollutant"]
    rating, health_tips = get_aqi_rating_and_suggestions(next_hour_val)
    pei, pei_label, pei_msg = compute_pei(
        next_hour_val, exposure_level, sensitivity_level
    )
    best_time, best_val, worst_time, worst_val = get_best_and_worst_time(future_df)
    pollutant_title, source_points = get_pollutant_source_info(pollutant)

    # =========================================================
    # PAGE 1: AIR QUALITY OVERVIEW
    # =========================================================
    if page == "🌍 Air Quality Overview":
        st.subheader(f"📍 Station: {station}")
        st.write(f"**Pollutant:** {pollutant}")
        st.write(f"**Last Update:** {latest['last_update']}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current Level (µg/m³)", f"{latest['pollutant_avg']:.2f}")
        with c2:
            st.metric(
                "Today's Min (µg/m³)",
                f"{station_df['pollutant_avg'].min():.2f}",
            )
        with c3:
            st.metric(
                "Today's Max (µg/m³)",
                f"{station_df['pollutant_avg'].max():.2f}",
            )

        st.markdown("---")
        st.subheader("📈 Latest 24-Hour Forecast Snapshot")
        fig = px.line(
            future_df,
            x="future_time",
            y="predicted_pollutant",
            markers=True,
            template=plot_template,
            title=f"{pollutant} forecast (next 24 hours) - {station}",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Recent Historical Trend")
        hist_df = station_df.tail(200)
        fig_hist = px.line(
            hist_df,
            x="last_update",
            y="pollutant_avg",
            template=plot_template,
            title=f"Recent {pollutant} levels at {station}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # =========================================================
    # PAGE 2: FORECAST
    # =========================================================
    elif page == "📈 Forecast":
        st.subheader(f"📈 Forecast – {pollutant} at {station}")
        st.write(f"**From:** {future_df['future_time'].min()}")
        st.write(f"**To:**   {future_df['future_time'].max()}")

        fig_forecast = px.line(
            future_df,
            x="future_time",
            y="predicted_pollutant",
            markers=True,
            template=plot_template,
            title="24-hour predicted concentration",
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.markdown("### 🔍 Forecast Table (Top 10 rows)")
        st.dataframe(
            future_df[["future_time", "predicted_pollutant"]].head(10),
            use_container_width=True,
        )

    # =========================================================
    # PAGE 3: HEALTH INSIGHTS
    # =========================================================
    elif page == "🩺 Health Insights":
        st.subheader("🏅 Air Quality Rating (Next Hour)")
        st.markdown(f"### {rating}")
        st.write(f"**Predicted Next Hour Level:** {next_hour_val:.2f} µg/m³")

        st.markdown("**Health & Lifestyle Suggestions:**")
        for tip in health_tips:
            st.markdown(f"- {tip}")

        st.markdown("---")
        st.subheader("👤 Personal Exposure Index (PEI)")
        st.markdown(f"**{pei_label}** — PEI: `{pei:.1f}`")
        st.markdown(f"*{pei_msg}*")

        st.markdown("---")
        st.subheader("🕒 Best Time to Go Outside (Next 24 Hours)")
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            st.markdown(
                f"""
                <div style="padding:12px;border-radius:10px;background:#14532d;color:white;">
                <b>Best Time</b><br>
                {best_time}<br>
                Predicted Level: {best_val:.2f} µg/m³
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_bt2:
            st.markdown(
                f"""
                <div style="padding:12px;border-radius:10px;background:#7f1d1d;color:white;">
                <b>Worst Time</b><br>
                {worst_time}<br>
                Predicted Level: {worst_val:.2f} µg/m³
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================
    # PAGE 4: POLLUTION SOURCES
    # =========================================================
    elif page == "🌍 Pollution Sources":
        st.subheader("🧠 Pollutant Source Intelligence")
        st.markdown(f"**Pollutant Profile:** {pollutant_title}")
        for s in source_points:
            st.markdown(f"- {s}")

        st.markdown("---")
        st.subheader("📌 Station Context")
        st.write(f"**Station:** {station}")
        st.write(f"**City:** {latest['city']}")
        st.write(f"**State:** {latest['state']}")
        st.write(f"**Coordinates:** ({latest['latitude']:.4f}, {latest['longitude']:.4f})")

        st.markdown("You can use this information to hypothesize local emission sources like:")
        st.markdown("- Nearby highways or traffic corridors")
        st.markdown("- Industrial zones or power plants")
        st.markdown("- Construction-heavy areas")
        st.markdown("- Agriculture or waste treatment facilities")

    # =========================================================
    # PAGE 5: MODEL ACCURACY
    # =========================================================
    elif page == "📊 Model Accuracy":
        st.subheader("📊 Model Performance Report")

        use_cols_metrics = [c for c in feature_cols if c in df.columns]
        metrics = compute_model_metrics(df, _model=model, feature_cols=use_cols_metrics)

        if metrics is None:
            st.warning("Not enough clean data to compute metrics.")
        else:
            mae = metrics["mae"]
            rmse = metrics["rmse"]
            r2 = metrics["r2"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE (µg/m³)", f"{mae:.3f}")
            with c2:
                st.metric("RMSE (µg/m³)", f"{rmse:.3f}")
            with c3:
                st.metric("R² Score", f"{r2:.3f}")

            st.markdown("These metrics are calculated on the latest 20% of the dataset (time-based split).")

            st.markdown("---")
            st.subheader("📋 Sample: Actual vs Predicted (first 100 points)")
            st.dataframe(metrics["results_sample"], use_container_width=True)

            if metrics["feature_importance"] is not None:
                st.markdown("---")
                st.subheader("🌿 Feature Importance (Model’s view of what matters)")
                fig_imp = px.bar(
                    metrics["feature_importance"].head(15),
                    x="importance",
                    y="feature",
                    orientation="h",
                    template=plot_template,
                    title="Top 15 Important Features",
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    # =========================================================
    # PAGE 6: SETTINGS
    # =========================================================
    elif page == "⚙️ Settings":
        st.subheader("⚙️ Settings & Info")
        st.markdown(
            """
            - **App Name:** AI-Powered AQI Forecast Dashboard  
            - **Model:** Trained regression model (e.g., LightGBM / RandomForest)  
            - **Task:** Next 24-hour pollutant concentration forecasting  
            - **Data:** Historical air quality readings with time & location features  
            
            
            """
        )


if __name__ == "__main__":
    main()
