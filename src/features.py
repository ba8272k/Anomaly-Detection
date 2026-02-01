import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# City coordinates (approx) for distance + travel speed simulation
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

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Load dataset
df = pd.read_csv("data/logins.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# -----------------------------
# Engineered Features (matches report)
# -----------------------------

# Time features
df["hour_of_day"] = df["timestamp"].dt.hour            # 0-23
df["day_of_week"] = df["timestamp"].dt.weekday         # 0-6 (Mon=0)

# Time since last login (seconds)
df["time_since_last_login"] = (
    df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0)
)

# New device flag (0/1)
df["new_device_flag"] = (
    df.groupby("user_id")["device_id"]
      .transform(lambda s: (s != s.shift(1)).fillna(False))
      .astype(int)
)

# Location frequency per user (how often the user used that city)
df["location_frequency"] = df.groupby(["user_id", "city"])["city"].transform("count")

# Distance + travel speed since last login
distance = []
speed = []

for i in range(len(df)):
    if i == 0 or df.loc[i, "user_id"] != df.loc[i - 1, "user_id"]:
        distance.append(0.0)
        speed.append(0.0)
        continue

    city_prev = df.loc[i - 1, "city"]
    city_now = df.loc[i, "city"]

    lat1, lon1 = COORDS[city_prev]
    lat2, lon2 = COORDS[city_now]

    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    hours = df.loc[i, "time_since_last_login"] / 3600.0
    travel_spd = dist_km / hours if hours > 0 else 0.0

    distance.append(dist_km)
    speed.append(travel_spd)

df["distance_from_last_login"] = distance
df["travel_speed"] = speed

# IP change count in last 5 login events per user
df["ip_changed"] = (
    df.groupby("user_id")["ip_address"]
      .transform(lambda s: (s != s.shift(1)).fillna(False))
)

df["ip_change_count"] = (
    df.groupby("user_id")["ip_changed"]
      .transform(lambda s: s.rolling(5, min_periods=1).sum())
      .fillna(0)
)

# Save ML features only
features = df[[
    "hour_of_day",
    "day_of_week",
    "time_since_last_login",
    "distance_from_last_login",
    "travel_speed",
    "new_device_flag",
    "location_frequency",
    "ip_change_count"
]]

features.to_csv("data/features.csv", index=False)

# Save full data + features (useful later for dashboard)
df.to_csv("data/logins_with_features.csv", index=False)

print("Saved: data/features.csv")
print("Saved: data/logins_with_features.csv")
