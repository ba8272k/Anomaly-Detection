import pandas as pd
import random
from datetime import datetime, timedelta

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

for user_id in range(1, NUM_USERS + 1):
    last_time = datetime.now() - timedelta(days=10)
    base_country = random.choice(countries)
    base_city = random.choice(cities[base_country])
    base_device = random.choice(devices)

    for _ in range(LOGINS_PER_USER):
        last_time = last_time + timedelta(hours=random.randint(1, 12))

        country = base_country
        city = base_city
        device_id = base_device
        ip_address = random_ip()

        data.append([
            event_id,
            user_id,
            last_time,
            ip_address,
            country,
            city,
            device_id,
            "success"
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
