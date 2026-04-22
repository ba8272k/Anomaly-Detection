import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# these values were chosen to match the ~5% anomaly rate in the dataset
CONTAMINATION = 0.05   # tells Isolation Forest to expect ~5% anomalies
RANDOM_STATE = 42
N_ESTIMATORS = 100

KMEANS_K = 5           # picked after looking at the elbow plot
KMEANS_ANOMALY_PCT = 5  # top 5% of points farthest from their cluster centre

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# features.py needs to have run first before this
df = pd.read_csv("data/logins_with_features.csv")

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

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X = df[feature_cols].values

# K-Means is distance-based so all features need to be on the same scale
# Isolation Forest doesn't need this
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# try K from 2 to 10 and plot inertia - we pick K where the curve bends
K_range = range(2, 11)
inertias = []
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km_tmp.fit(X_scaled)
    inertias.append(km_tmp.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("K-Means Elbow Plot")
plt.tight_layout()
plt.savefig("plots/kmeans_elbow.png", dpi=150)
plt.close()
print("Elbow plot saved: plots/kmeans_elbow.png")

# Isolation Forest - builds random trees and anomalies get isolated faster
iso = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
)
iso.fit(X)

iso_pred_raw = iso.predict(X)           # 1 = normal, -1 = anomaly
iso_score = -iso.decision_function(X)   # flip sign so higher score = more suspicious

df["if_anomaly_flag"] = pd.Series(iso_pred_raw).map({1: 0, -1: 1}).values
df["if_anomaly_score"] = iso_score

# K-Means - points far from their cluster centre are likely anomalies
kmeans = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

centers = kmeans.cluster_centers_
distances = np.linalg.norm(X_scaled - centers[clusters], axis=1)

threshold = np.percentile(distances, 100 - KMEANS_ANOMALY_PCT)

df["km_distance"] = distances
df["km_anomaly_flag"] = (distances >= threshold).astype(int)

# if either model flags a login as anomalous, we count it
df["combined_anomaly_flag"] = (
    (df["if_anomaly_flag"] == 1) | (df["km_anomaly_flag"] == 1)
).astype(int)

# write everything to CSV so the dashboard can read it
output_path = "data/anomaly_scores.csv"
df.to_csv(output_path, index=False)
print(f"Results saved: {output_path}")

# save the trained models so we don't have to retrain each time the dashboard loads
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(iso, "models/isolation_forest.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
print("Models saved: models/scaler.pkl, models/isolation_forest.pkl, models/kmeans.pkl")

# check predictions against the labels that were injected during data generation
if "is_anomaly" not in df.columns:
    print("\nWARNING: 'is_anomaly' column not found - skipping evaluation.")
    print("Run generate_data.py first to produce ground-truth labels.")
else:
    y_true = df["is_anomaly"].values

    print("\nEVALUATION AGAINST GROUND-TRUTH LABELS")
    print("-" * 50)

    for name, y_pred, y_score in [
        ("Isolation Forest", df["if_anomaly_flag"].values, df["if_anomaly_score"].values),
        ("K-Means", df["km_anomaly_flag"].values, df["km_distance"].values),
        ("Combined", df["combined_anomaly_flag"].values, None),
    ]:
        print(f"\n-- {name} --")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion matrix:\n{cm}")

        if y_score is not None:
            try:
                auc = roc_auc_score(y_true, y_score)
                print(f"ROC-AUC: {auc:.4f}")
            except ValueError:
                print("ROC-AUC: could not compute")

    # plot both models together so we can compare them visually
    plt.figure(figsize=(7, 5))
    for name, y_score in [
        ("Isolation Forest", df["if_anomaly_score"].values),
        ("K-Means distance", df["km_distance"].values),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Anomaly Detection Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/roc_curves.png", dpi=150)
    plt.close()
    print("\nROC curve saved: plots/roc_curves.png")

# quick summary of how many anomalies each method found
print("\nDETECTION SUMMARY")
print("-" * 50)
print(f"Total records: {len(df)}")
print(f"Isolation Forest anomalies: {int(df['if_anomaly_flag'].sum())}")
print(f"K-Means anomalies: {int(df['km_anomaly_flag'].sum())}")
print(f"Combined anomalies: {int(df['combined_anomaly_flag'].sum())}")
if "is_anomaly" in df.columns:
    print(f"Ground-truth anomalies: {int(df['is_anomaly'].sum())}")
