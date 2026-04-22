import pandas as pd
import numpy as np

# lat/lon coordinates for each city - needed to calculate distances between logins
COORDS = {
    "London": (51.5074, -0.1278),
    "Manchester": (53.4808, -2.2426),
    "Paris": (48.8566, 2.3522),
    "Lyon": (45.7640, 4.8357),
    "Berlin": (52.5200, 13.4050),
    "Munich": (48.1351, 11.5820),
    "New York": (40.7128, -74.0060),
    "Chicago": (41.8781, -87.6298),
    "Tokyo": (35.6895, 139.6917),
    "Osaka": (34.6937, 135.5023),
}


def calculate_distance(lat1, lon1, lat2, lon2):
    # haversine formula - accounts for the curvature of the earth
    # works on entire numpy arrays at once rather than row by row
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


df = pd.read_csv("data/logins.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# just in case a city appears that we haven't added coordinates for yet
unknown_cities = set(df["city"]) - set(COORDS)
if unknown_cities:
    print(f"WARNING: unknown cities will get distance=0: {unknown_cities}")

df["_lat"] = df["city"].map(lambda c: COORDS[c][0] if c in COORDS else np.nan)
df["_lon"] = df["city"].map(lambda c: COORDS[c][1] if c in COORDS else np.nan)

# what hour and day the login happened - useful for spotting night-time logins
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.weekday

# how many seconds have passed since this user's previous login
df["time_since_last_login"] = (
    df.groupby("user_id")["timestamp"]
      .diff()
      .dt.total_seconds()
      .fillna(0)
)

# marks 1 if this is the first time this device has been seen for this user
def first_occurrence_flag(series):
    seen = set()
    result = []
    for val in series:
        result.append(0 if val in seen else 1)
        seen.add(val)
    return pd.Series(result, index=series.index)

df["new_device_flag"] = (
    df.groupby("user_id")["device_id"]
      .transform(first_occurrence_flag)
      .astype(int)
)

# counts how many times this user has visited this city before the current login
# only looks backwards - no future data leaks into this number
def prior_location_count(series):
    counts = {}
    result = []
    for val in series:
        result.append(counts.get(val, 0))
        counts[val] = counts.get(val, 0) + 1
    return pd.Series(result, index=series.index)

df["location_frequency"] = (
    df.groupby("user_id")["city"]
      .transform(prior_location_count)
)

# calculate how far they travelled from their last login and at what implied speed
# this is the main feature for catching impossible travel
prev_lat = df.groupby("user_id")["_lat"].shift(1)
prev_lon = df.groupby("user_id")["_lon"].shift(1)

dist_km = calculate_distance(
    prev_lat.values, prev_lon.values,
    df["_lat"].values, df["_lon"].values,
)
dist_km = np.nan_to_num(dist_km, nan=0.0)

hours = df["time_since_last_login"] / 3600.0
travel_spd = np.where(hours > 0, dist_km / hours, 0.0)

df["distance_from_last_login"] = dist_km
df["travel_speed"] = travel_spd

# rolling count of how many times the IP address changed in the last 5 logins
df["_ip_changed"] = (
    df.groupby("user_id")["ip_address"]
      .transform(lambda s: (s != s.shift(1)).fillna(False))
      .astype(int)
)

df["ip_change_count"] = (
    df.groupby("user_id")["_ip_changed"]
      .transform(lambda s: s.rolling(5, min_periods=1).sum())
      .fillna(0)
)

df.drop(columns=["_lat", "_lon", "_ip_changed"], inplace=True)

feature_cols = [
    "hour_of_day",
    "day_of_week",
    "time_since_last_login",
    "distance_from_last_login",
    "travel_speed",
    "new_device_flag",
    "location_frequency",
    "ip_change_count",
]

df.to_csv("data/logins_with_features.csv", index=False)

print("Saved: data/logins_with_features.csv")
print(f"Rows: {len(df)}")
print(f"Features computed: {feature_cols}")
