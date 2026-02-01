import pandas as pd
import random
from datetime import datetime, timedelta

# Number of users and logins
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

data = []
event_id = 1

for user_id in range(1, NUM_USERS + 1):
    last_time = datetime.now() - timedelta(days=10)
    last_country = random.choice(countries)
    last_city = random.choice(cities[last_country])
    last_device = random.choice(devices)

    for _ in range(LOGINS_PER_USER):
        time = last_time + timedelta(hours=random.randint(1, 12))
        country = last_country
        city = last_city
        device = last_device

        data.append([
            event_id,
            user_id,
            time,
            country,
            city,
            device,
            "success"
        ])

        last_time = time
        event_id += 1

df = pd.DataFrame(data, columns=[
    "event_id",
    "user_id",
    "timestamp",
    "country",
    "city",
    "device_id",
    "login_result"
])

df.to_csv("data/logins.csv", index=False)
print("Dataset generated: data/logins.csv")
