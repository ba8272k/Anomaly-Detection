import pandas as pd
import random
from datetime import datetime, timedelta

# -----------------------------
# Student-friendly configuration
# -----------------------------
NUM_USERS = 50
LOGINS_PER_USER = 40

countries = ["UK", "France", "Germany", "USA", "Japan"]
cities = {
    "UK": ["London", "Manchester"],
    "France": ["Paris", "Lyon"],
    "Germany": ["Berlin", "Munich"],
    "USA": ["New York", "Chicago"],
    "Japan": ["Tokyo", "Osaka"]
}
devices = ["Chrome", "Firefox", "Safari", "Edge"]

def random_ip():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"

data = []
event_id = 1

# Generate "normal" behaviour for each user
for user_id in range(1, NUM_USERS + 1):
    last_time = datetime.now() - timedelta(days=10)

    base_country = random.choice(countries)
    base_city = random.choice(cities[base_country])
    base_device = random.choice(devices)

    # Make IP mostly stable (so IP changes mean something)
    base_ip = random_ip()

    for _ in range(LOGINS_PER_USER):
        # Time moves forward
        last_time = last_time + timedelta(hours=random.randint(1, 12))

        # Most logins are normal
        country = base_country
        city = base_city
        device_id = base_device

        # 80% keep same IP, 20% change (realistic-ish)
        if random.random() < 0.8:
            ip_address = base_ip
        else:
            ip_address = random_ip()

        login_result = "success"

        data.append([
            event_id,
            user_id,
            last_time,
            ip_address,
            country,
            city,
            device_id,
            login_result
        ])
        event_id += 1

df = pd.DataFrame(data, columns=[
    "event_id",
    "user_id",
    "timestamp",
    "ip_address",
    "country",
    "city",
    "device_id",
    "login_result"
])

df.to_csv("data/logins.csv", index=False)
print("Dataset generated: data/logins.csv")
