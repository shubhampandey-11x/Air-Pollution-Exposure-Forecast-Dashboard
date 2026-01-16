import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CATEGORICAL_COLS = ["city", "state", "station"]

FEATURE_COLS = [
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

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
print("📂 Loading model...")
try:
    saved = joblib.load("aq_model.pkl")
except:
    saved = joblib.load("models/aq_model.pkl")

if isinstance(saved, tuple):
    model, feature_cols = saved
else:
    model = saved
    feature_cols = FEATURE_COLS

print("🔍 Model Loaded Successfully!")


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
print("📂 Loading dataset...")

for fname in ["air_quality_expanded_synthetic.csv", "air_quality_expanded.csv", "air_quality.csv"]:
    try:
        df = pd.read_csv(fname)
        print("Using file:", fname)
        break
    except:
        df = None

if df is None:
    print("❌ ERROR: No dataset found.")
    exit()

df["last_update"] = pd.to_datetime(df["last_update"])
df = df.sort_values("last_update")

df["hour"] = df["last_update"].dt.hour
df["day"] = df["last_update"].dt.day
df["month"] = df["last_update"].dt.month
df["weekday"] = df["last_update"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

df["dist_from_center"] = np.sqrt(
    (df["latitude"] - 22.0) ** 2 + (df["longitude"] - 79.0) ** 2
)

for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].astype("category")

# ---------------------------------------------------
# Prepare Training/Test Dataset
# ---------------------------------------------------
target = "pollutant_avg"

df_clean = df.dropna(subset=FEATURE_COLS + [target])

n = len(df_clean)
train_size = int(n * 0.8)

train_df = df_clean.iloc[:train_size]
test_df = df_clean.iloc[train_size:]

X_test = test_df[feature_cols]
y_test = test_df[target]

print(f"📊 Test Data Size: {len(test_df)} rows")

# ---------------------------------------------------
# Ensure categorical dtypes
# ---------------------------------------------------
for col in CATEGORICAL_COLS:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype("category")

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
print("🤖 Predicting on test set...")
y_pred = model.predict(X_test)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===============================")
print("📈 MODEL PERFORMANCE")
print("===============================")
print(f"MAE  (Mean Absolute Error): {mae:.3f}")
print(f"RMSE (Root Mean Sq Error):  {rmse:.3f}")
print(f"R² Score:                   {r2:.3f}")
print("===============================")

# ---------------------------------------------------
# Show Actual vs Predicted
# ---------------------------------------------------
results = pd.DataFrame({
    "Actual": y_test.values[:50],
    "Predicted": y_pred[:50]
})

print("\n📋 Sample Actual vs Predicted Values (first 50 rows):")
print(results)

# ---------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------
results.to_csv("evaluation_results.csv", index=False)
print("\n📁 Saved detailed results to evaluation_results.csv")
