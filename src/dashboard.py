import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import roc_curve, roc_auc_score
import os

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide",
)

# cache the CSV files so the page doesn't reload them on every interaction
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    scores = pd.read_csv(os.path.join(base, "data", "anomaly_scores.csv"))
    rules = pd.read_csv(os.path.join(base, "data", "rule_alerts.csv"))
    comp = pd.read_csv(os.path.join(base, "data", "comparison_summary.csv"))
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    return scores, rules, comp

try:
    df, rules_df, comp_df = load_data()
except FileNotFoundError:
    st.error("Data files not found. Run: generate_data.py, features.py, models.py, rules.py")
    st.stop()

has_gt = "is_anomaly" in df.columns

st.title("Cloud-Based Authentication Anomaly Detection System")
st.divider()

# top-level numbers shown as metric cards at the top of the page
total = len(df)
if_alerts = int(df["if_anomaly_flag"].sum())
km_alerts = int(df["km_anomaly_flag"].sum())
combined = int(df["combined_anomaly_flag"].sum())
gt_total = int(df["is_anomaly"].sum()) if has_gt else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Login Events", f"{total:,}")
c2.metric("Injected Anomalies", gt_total)
c3.metric("Isolation Forest Alerts", if_alerts)
c4.metric("K-Means Alerts", km_alerts)
c5.metric("Combined Alerts", combined)

st.divider()

# left: breakdown of which attack types were injected
# right: how many times each rule fired
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Anomaly Type Breakdown")
    if has_gt:
        label_map = {
            "rapid_ip_switching": "Rapid IP Switching",
            "impossible_travel": "Impossible Travel",
            "new_device_night": "New Device at Night",
            "rare_location": "Rare Location",
        }
        breakdown = df[df["is_anomaly"] == 1]["anomaly_type"].value_counts().reset_index()
        breakdown.columns = ["Attack Type", "Count"]
        breakdown["Attack Type"] = breakdown["Attack Type"].map(label_map).fillna(breakdown["Attack Type"])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(breakdown["Attack Type"], breakdown["Count"],
                color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"][:len(breakdown)])
        ax.set_xlabel("Count")
        for i, val in enumerate(breakdown["Count"]):
            ax.text(val + 0.3, i, str(val), va="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No ground truth labels available.")

with col_right:
    st.subheader("Rule-Based Alert Breakdown")
    rule_data = {
        "Rule": ["Impossible Travel", "New Device at Night", "Rapid IP Switching", "Rare Location"],
        "Alerts": [
            int(rules_df["rule_impossible_travel"].sum()),
            int(rules_df["rule_new_device_night"].sum()),
            int(rules_df["rule_ip_change"].sum()),
            int(rules_df["rule_rare_location"].sum()),
        ]
    }
    rule_breakdown = pd.DataFrame(rule_data)
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.barh(rule_breakdown["Rule"], rule_breakdown["Alerts"],
             color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"])
    ax2.set_xlabel("Alerts")
    for i, val in enumerate(rule_breakdown["Alerts"]):
        ax2.text(val + 1, i, str(val), va="center", fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.divider()

# model performance table and ROC curves placed side by side
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Model Performance Comparison")
    styled = comp_df.copy()
    styled.columns = [c.title() for c in styled.columns]

    def colour_f1(val):
        try:
            v = float(val)
            if v >= 0.4: return "color: green"
            elif v >= 0.2: return "color: orange"
            else: return "color: red"
        except: return ""

    def colour_auc(val):
        try:
            v = float(val)
            if v >= 0.7: return "color: green"
            elif v >= 0.6: return "color: orange"
            else: return "color: red"
        except: return ""

    st.dataframe(
        styled.style
            .map(colour_f1, subset=["F1-Score"])
            .map(colour_auc, subset=["Roc-Auc"])
            .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Precision = how many alerts were real | Recall = how many attacks were caught | ROC-AUC: 0.5 = random, 1.0 = perfect")

with col_b:
    st.subheader("ROC Curves")
    if has_gt:
        y_true = df["is_anomaly"].values
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        for name, score_col, colour in [
            ("Isolation Forest", "if_anomaly_score", "blue"),
            ("K-Means", "km_distance", "orange"),
        ]:
            fpr, tpr, _ = roc_curve(y_true, df[score_col].values)
            auc_val = roc_auc_score(y_true, df[score_col].values)
            ax3.plot(fpr, tpr, color=colour, lw=2, label=f"{name} (AUC={auc_val:.3f})")
        ax3.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curves")
        ax3.legend()
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

st.divider()

# score distribution shows how well the model separates normal vs anomalous logins
# timeline shows when anomalies occurred relative to total login volume
col_x, col_y = st.columns(2)

with col_x:
    st.subheader("Isolation Forest — Score Distribution")
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    normal = df[df["if_anomaly_flag"] == 0]["if_anomaly_score"]
    anomaly = df[df["if_anomaly_flag"] == 1]["if_anomaly_score"]
    ax4.hist(normal, bins=40, color="green", alpha=0.6, label="Normal")
    ax4.hist(anomaly, bins=40, color="red", alpha=0.8, label="Anomaly")
    ax4.set_xlabel("Anomaly Score (higher = more suspicious)")
    ax4.set_ylabel("Count")
    ax4.legend()
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close()

with col_y:
    st.subheader("Login Events Over Time")
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date").agg(
        total=("event_id", "count"),
        anomalies=("combined_anomaly_flag", "sum")
    ).reset_index()
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    ax5.fill_between(range(len(daily)), daily["total"], color="steelblue", alpha=0.4, label="Total logins")
    ax5.fill_between(range(len(daily)), daily["anomalies"], color="red", alpha=0.7, label="Anomalies")
    ax5.set_xlabel("Days")
    ax5.set_ylabel("Count")
    ax5.legend()
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close()

st.divider()

# bottom table: the actual flagged logins sorted by how suspicious they are
st.subheader("Recent Flagged Login Events")

flagged = df[df["combined_anomaly_flag"] == 1][[
    "timestamp", "user_id", "country", "city", "device_id",
    "travel_speed", "ip_change_count", "if_anomaly_score",
    "if_anomaly_flag", "km_anomaly_flag",
]].copy()

if has_gt:
    flagged["ground_truth"] = df.loc[flagged.index, "anomaly_type"]

flagged = flagged.sort_values("if_anomaly_score", ascending=False).head(50)
flagged.columns = [c.replace("_", " ").title() for c in flagged.columns]
flagged["Travel Speed"] = flagged["Travel Speed"].round(1).astype(str) + " km/h"
flagged["If Anomaly Score"] = flagged["If Anomaly Score"].round(4)

st.dataframe(flagged, use_container_width=True, hide_index=True)
