import requests
import pandas as pd
import time

df_orig = pd.read_csv("air_quality.csv")

cities = df_orig["city"].dropna().unique().tolist()
pollutants_map = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "OZONE": "o3",
    "NH3": "nh3"
}

available_pollutants = df_orig["pollutant_id"].dropna().unique().tolist()
pollutants_to_fetch = [pollutants_map[p] for p in available_pollutants if p in pollutants_map]

BASE_URL = "https://api.openaq.org/v2/measurements"

def fetch_page(city, pollutant, page, limit=100):
    try:
        params = {
            "city": city,
            "parameter": pollutant,
            "limit": limit,
            "page": page,
            "date_from": "2025-07-01T00:00:00+05:30",
            "date_to": "2025-08-15T23:59:59+05:30"
        }
        r = requests.get(BASE_URL, params=params, timeout=10)
        if r.status_code != 200:
            return None

        js = r.json()
        return js.get("results", [])
    except:
        return None


df_all = []

for city in cities:
    for pollutant in pollutants_to_fetch:
        print(f"\n📌 Fetching data for {city} - {pollutant}…")

        all_records = []
        page = 1

        while True:
            print(f"   → Fetching page {page}…")

            data = fetch_page(city, pollutant, page)

            if not data:
                print("   → No more data or timeout.")
                break

            all_records.extend(data)

            # If fewer results than limit, no more pages
            if len(data) < 100:
                break

            page += 1
            time.sleep(1)  # Prevent rate limit

        if len(all_records) == 0:
            print("   ⚠️ No data found.")
            continue

        df = pd.DataFrame(all_records)

        df2 = pd.DataFrame({
            "city": df["city"],
            "state": None,
            "station": df["location"],
            "latitude": df["coordinates"].apply(lambda x: x["latitude"]),
            "longitude": df["coordinates"].apply(lambda x: x["longitude"]),
            "pollutant_id": pollutant.upper(),
            "pollutant_avg": df["value"],
            "pollutant_min": df["value"],
            "pollutant_max": df["value"],
            "last_update": pd.to_datetime(df["date"].apply(lambda x: x["local"]))
        })

        print(f"   ✅ Fetched {len(df2)} rows.")
        df_all.append(df2)


if df_all:
    df_expanded = pd.concat([df_orig] + df_all, ignore_index=True)
else:
    df_expanded = df_orig.copy()

df_expanded.to_csv("air_quality_expanded.csv", index=False)
print("\n🎉 Saved expanded dataset as air_quality_expanded.csv")
