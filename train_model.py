import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

df = pd.read_csv("air_quality_expanded_synthetic.csv")
df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce")
df = df.dropna(subset=["last_update"])

df["hour"] = df["last_update"].dt.hour
df["day"] = df["last_update"].dt.day
df["month"] = df["last_update"].dt.month
df["weekday"] = df["last_update"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
df["dist_from_center"] = np.sqrt((df["latitude"] - 22.0)**2 + (df["longitude"] - 79.0)**2)

df[["pollutant_min","pollutant_max","pollutant_avg"]] = df[
    ["pollutant_min","pollutant_max","pollutant_avg"]
].fillna(df[["pollutant_min","pollutant_max","pollutant_avg"]].median())

target = "pollutant_avg"

feature_cols = [
    "latitude", "longitude", 
    "pollutant_min", "pollutant_max",
    "hour","day","month","weekday","is_weekend",
    "dist_from_center",
    "city","state","station"
]

df = df.dropna(subset=feature_cols+[target])
df = df.sort_values("last_update")

train_size = int(len(df)*0.8)
train_df = df.iloc[:train_size]

X_train = train_df[feature_cols]
y_train = train_df[target]

for col in ["city", "state", "station"]:
    X_train[col] = X_train[col].astype("category")

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31
)

model.fit(X_train, y_train)

joblib.dump((model, feature_cols), "aq_model.pkl")

print("Model saved as aq_model.pkl")
