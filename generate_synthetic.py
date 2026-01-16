import pandas as pd
import numpy as np

# ------------- CONFIG -------------
DAYS = 30          # 30 days
FREQ = "H"         # hourly
MIN_VALUE = 5.0    # minimum pollution value
# ----------------------------------


print("📂 Loading original dataset: air_quality.csv ...")
orig = pd.read_csv("air_quality.csv")

required_cols = [
    "country", "state", "city", "station",
    "last_update", "latitude", "longitude",
    "pollutant_id", "pollutant_min",
    "pollutant_max", "pollutant_avg"
]
missing = [c for c in required_cols if c not in orig.columns]
if missing:
    raise ValueError(f"Missing columns in air_quality.csv: {missing}")

# Clean duplicates
orig = orig.dropna(subset=["station", "pollutant_id"]).copy()

# Unique station–pollutant pairs that actually exist in your data
groups = orig.groupby(["station", "pollutant_id"])

print(f"✅ Found {len(groups)} station–pollutant combinations.")

# Create time index: last 30 days, hourly
end_time = pd.Timestamp.now().floor("H")
hours = pd.date_range(end=end_time, periods=DAYS * 24, freq=FREQ)

rows = []

for (station, pollutant), grp in groups:
    base = grp.iloc[0]
    country = base["country"]
    state = base["state"]
    city = base["city"]
    lat = base["latitude"]
    lon = base["longitude"]

    base_val = grp["pollutant_avg"].mean()
    if np.isnan(base_val):
        base_val = 50.0

    print(f"⏳ Generating for station='{station}', pollutant='{pollutant}', base={base_val:.2f}")

    for t in hours:
        hour = t.hour
        weekday = t.weekday()

        # Daily baseline (smooth sinusoidal day cycle)
        baseline_variation = 8 * np.sin((hour / 24) * 2 * np.pi)

        # Traffic peaks: morning & evening
        morning_peak = 15 * np.exp(-((hour - 8) ** 2) / (2 * 2.5**2))
        evening_peak = 12 * np.exp(-((hour - 19) ** 2) / (2 * 3.0**2))
        traffic_peak = morning_peak + evening_peak

        # Weekend effect (slightly higher or lower pollution)
        weekend_effect = 6 if weekday >= 5 else 0

        # Random spikes (industrial / firecrackers / events)
        random_spike = np.random.normal(0, 6)

        # Final synthetic value
        val = base_val + baseline_variation + traffic_peak + weekend_effect + random_spike
        val = max(val, MIN_VALUE)

        # min & max around avg
        min_val = max(val - np.random.uniform(0, 5), 1)
        max_val = val + np.random.uniform(0, 5)

        rows.append({
            "country": country,
            "state": state,
            "city": city,
            "station": station,
            "last_update": t,
            "latitude": lat,
            "longitude": lon,
            "pollutant_id": pollutant,
            "pollutant_min": min_val,
            "pollutant_max": max_val,
            "pollutant_avg": val
        })

print("🧮 Converting to DataFrame ...")
synthetic = pd.DataFrame(rows)

out_path = "air_quality_expanded_synthetic.csv"
print(f"💾 Saving synthetic dataset to {out_path} ...")
synthetic.to_csv(out_path, index=False)

print("🎉 Done! Synthetic dataset created successfully.")
print(f"Rows generated: {len(synthetic)}")
