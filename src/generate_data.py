import os
import random
import pandas as pd
from datetime import datetime, timedelta

# seed makes sure we get the same dataset every time we run this
random.seed(42)

NUM_USERS = 50
LOGINS_PER_USER = 40
ANOMALY_RATE = 0.05  # about 5% of all logins will be attacks

# all users start from Jan 2024, keeps the data consistent across runs
BASE_DATE = datetime(2024, 1, 1)

countries = ["UK", "France", "Germany", "USA", "Japan"]
cities = {
    "UK": ["London", "Manchester"],
    "France": ["Paris", "Lyon"],
    "Germany": ["Berlin", "Munich"],
    "USA": ["New York", "Chicago"],
    "Japan": ["Tokyo", "Osaka"],
}
devices = ["Chrome", "Firefox", "Safari", "Edge"]

# pairs of countries that are far apart geographically
# used to simulate the impossible travel attack scenario
FAR_PAIRS = {
    "UK": ["Japan", "USA"],
    "France": ["Japan", "USA"],
    "Germany": ["Japan", "USA"],
    "USA": ["Japan", "UK"],
    "Japan": ["UK", "USA"],
}

os.makedirs("data", exist_ok=True)


def random_ip():
    return (
        f"{random.randint(1, 255)}."
        f"{random.randint(0, 255)}."
        f"{random.randint(0, 255)}."
        f"{random.randint(0, 255)}"
    )


# attack 1: user somehow logs in from a distant country just 15-45 minutes later
def inject_impossible_travel(base_country, last_time):
    far_country = random.choice(FAR_PAIRS[base_country])
    far_city = random.choice(cities[far_country])
    new_time = last_time + timedelta(minutes=random.randint(15, 45))
    return far_country, far_city, new_time


# attack 2: a brand new device/browser shows up in the middle of the night
def inject_new_device_night(base_device):
    new_device = random.choice([d for d in devices if d != base_device])
    return new_device, random.randint(1, 4)


# attack 4: login comes from a country this user has never used before
def inject_rare_location(base_country):
    rare_country = random.choice([c for c in countries if c != base_country])
    rare_city = random.choice(cities[rare_country])
    return rare_country, rare_city


data = []
event_id = 1

for user_id in range(1, NUM_USERS + 1):
    # stagger start times so not all users begin on the same day
    last_time = BASE_DATE + timedelta(days=random.randint(0, 30))

    base_country = random.choice(countries)
    base_city = random.choice(cities[base_country])
    base_device = random.choice(devices)
    base_ip = random_ip()

    for _ in range(LOGINS_PER_USER):
        last_time += timedelta(hours=random.randint(1, 12))

        country = base_country
        city = base_city
        device_id = base_device
        ip_address = base_ip
        login_result = "success"
        is_anomaly = 0
        anomaly_type = "normal"

        if random.random() < ANOMALY_RATE:
            attack = random.choice([
                "impossible_travel",
                "new_device_night",
                "rapid_ip_switching",
                "rare_location",
            ])

            if attack == "impossible_travel":
                country, city, last_time = inject_impossible_travel(
                    base_country, last_time
                )
                login_result = random.choice(["success", "failure"])
                is_anomaly = 1
                anomaly_type = "impossible_travel"

            elif attack == "new_device_night":
                device_id, hour_override = inject_new_device_night(base_device)
                last_time = last_time.replace(hour=hour_override)
                login_result = "success"
                is_anomaly = 1
                anomaly_type = "new_device_night"

            elif attack == "rapid_ip_switching":
                # add 3 consecutive logins, each from a different IP
                for _ in range(3):
                    rapid_time = last_time + timedelta(minutes=random.randint(1, 5))
                    data.append([
                        event_id, user_id, rapid_time,
                        random_ip(),
                        country, city, device_id,
                        "success", 1, "rapid_ip_switching",
                    ])
                    event_id += 1
                    last_time = rapid_time
                is_anomaly = 1
                anomaly_type = "rapid_ip_switching"
                ip_address = random_ip()

            elif attack == "rare_location":
                country, city = inject_rare_location(base_country)
                is_anomaly = 1
                anomaly_type = "rare_location"

        else:
            # sometimes the user travels to another city in their home country - normal
            if random.random() < 0.1:
                city = random.choice(cities[base_country])

            # very occasionally logs in from abroad, but not flagged as an attack
            if random.random() < 0.05:
                country = random.choice(countries)
                city = random.choice(cities[country])

            # users do switch browsers from time to time
            if random.random() < 0.1:
                device_id = random.choice(devices)

            # IP stays the same most of the time, changes occasionally (e.g. different network)
            ip_address = base_ip if random.random() < 0.8 else random_ip()

        data.append([
            event_id, user_id, last_time,
            ip_address, country, city, device_id,
            login_result, is_anomaly, anomaly_type,
        ])
        event_id += 1


df = pd.DataFrame(data, columns=[
    "event_id", "user_id", "timestamp",
    "ip_address", "country", "city", "device_id",
    "login_result", "is_anomaly", "anomaly_type",
])

df.to_csv("data/logins.csv", index=False)

total = len(df)
anomalies = int(df["is_anomaly"].sum())
print(f"Dataset generated: data/logins.csv")
print(f"Total records: {total}")
print(f"Injected anomalies: {anomalies} ({anomalies / total * 100:.1f}%)")
print(f"Anomaly breakdown:")
print(df[df["is_anomaly"] == 1]["anomaly_type"].value_counts().to_string())
