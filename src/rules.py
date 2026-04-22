import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# thresholds chosen based on what would be physically or behaviourally impossible
IMPOSSIBLE_SPEED = 900   # km/h - faster than a commercial aircraft
IP_CHANGE_LIMIT = 3      # more than 3 IP changes in a 5-login window looks suspicious
NIGHT_HOURS = {0, 1, 2, 3, 4, 5, 23}  # late night and early morning hours

df = pd.read_csv("data/anomaly_scores.csv")

# rule 1: travel speed is too high to be physically possible
df["rule_impossible_travel"] = (df["travel_speed"] > IMPOSSIBLE_SPEED).astype(int)

# rule 2: a device that has never been seen before appears during night hours
df["rule_new_device_night"] = (
    (df["new_device_flag"] == 1) &
    (df["hour_of_day"].isin(NIGHT_HOURS))
).astype(int)

# rule 3: IP address keeps changing across recent logins
df["rule_ip_change"] = (df["ip_change_count"] > IP_CHANGE_LIMIT).astype(int)

# rule 4: user is logging in from a city they have never visited before
df["rule_rare_location"] = (df["location_frequency"] == 0).astype(int)

# raise an alert if any of the four rules fired
df["rule_alert"] = (
    (df["rule_impossible_travel"] == 1) |
    (df["rule_new_device_night"] == 1) |
    (df["rule_ip_change"] == 1) |
    (df["rule_rare_location"] == 1)
).astype(int)

output_file = "data/rule_alerts.csv"
df.to_csv(output_file, index=False)

print("RULE-BASED DETECTION SUMMARY")
print("-" * 50)
print(f"Total records         : {len(df)}")
print(f"Rule alerts (total)   : {int(df['rule_alert'].sum())}")
print(f"  Impossible travel   : {int(df['rule_impossible_travel'].sum())}")
print(f"  New device at night : {int(df['rule_new_device_night'].sum())}")
print(f"  Rapid IP switching  : {int(df['rule_ip_change'].sum())}")
print(f"  Rare location       : {int(df['rule_rare_location'].sum())}")
print(f"Saved: {output_file}")

# compare rule-based vs ML models using the ground-truth labels from data generation
if "is_anomaly" not in df.columns:
    print("\nWARNING: 'is_anomaly' column not found - skipping comparison.")
    print("Regenerate data with generate_data.py to get ground-truth labels.")
else:
    y_true = df["is_anomaly"].values

    print("\nSIDE-BY-SIDE COMPARISON: RULES vs ML MODELS")
    print("-" * 50)

    summary_rows = []

    def evaluate(label, flag_col, score_col=None):
        y_pred = df[flag_col].values
        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Anomaly"],
            output_dict=True,
        )
        prec = report["Anomaly"]["precision"]
        rec = report["Anomaly"]["recall"]
        f1 = report["Anomaly"]["f1-score"]
        alerts = int(y_pred.sum())
        auc = None
        if score_col:
            try:
                auc = roc_auc_score(y_true, df[score_col].values)
            except ValueError:
                pass
        summary_rows.append({
            "Detector": label,
            "Alerts": alerts,
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "ROC-AUC": round(auc, 3) if auc else "N/A",
        })
        print(f"\n-- {label} --")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
        print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
        if auc:
            print(f"ROC-AUC: {auc:.4f}")

    evaluate("Rule-Based (baseline)", "rule_alert")
    evaluate("Isolation Forest", "if_anomaly_flag", "if_anomaly_score")
    evaluate("K-Means", "km_anomaly_flag", "km_distance")
    evaluate("Combined ML", "combined_anomaly_flag")

    summary_df = pd.DataFrame(summary_rows)
    print("\nSUMMARY TABLE")
    print("-" * 50)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("data/comparison_summary.csv", index=False)
    print("\nComparison saved: data/comparison_summary.csv")
