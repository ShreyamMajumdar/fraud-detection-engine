import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import IsolationForest
import os

df = pd.read_csv('data/transactions.csv')
os.makedirs('outputs', exist_ok=True)

with open('data/fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

features = ['amount', 'hour', 'is_foreign', 'is_new_device', 'transactions_today', 'account_age_days']

print("Calculating risk scores for all transactions...")

proba = model.predict_proba(df[features])[:, 1]
df['risk_score'] = (proba * 100).round(1)

def get_risk_label(score):
    if score >= 70: return 'High Risk'
    elif score >= 40: return 'Medium Risk'
    else: return 'Low Risk'

df['risk_level'] = df['risk_score'].apply(get_risk_label)

print("\nRisk Level Breakdown:")
print(df['risk_level'].value_counts().to_string())

print("\nTop 10 Highest Risk Transactions:")
top10 = df.nlargest(10, 'risk_score')[
    ['transaction_id','amount','hour','risk_score','risk_level','is_fraud']
]
print(top10.to_string(index=False))

print("\nRunning anomaly detection...")

iso = IsolationForest(contamination=0.15, random_state=42)
df['anomaly'] = iso.fit_predict(df[features])

df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

anomalies = df[df['anomaly'] == -1]
print("Anomalies detected:", len(anomalies))
print("These are unusual transactions worth investigating!")

df.to_csv('data/transactions_scored.csv', index=False)
print("\nScored data saved to: data/transactions_scored.csv")

plt.figure(figsize=(10, 5))
plt.hist(df[df['is_fraud']==0]['risk_score'], bins=30, color='steelblue', alpha=0.7, label='Normal')
plt.hist(df[df['is_fraud']==1]['risk_score'], bins=30, color='red', alpha=0.7, label='Fraud')
plt.title('Risk Score Distribution -- Normal vs Fraud', fontsize=13)
plt.xlabel('Risk Score (0 = safe, 100 = high risk)')
plt.ylabel('Number of Transactions')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/chart3_risk_scores.png')
plt.show()
print("Chart 1 saved.")

plt.figure(figsize=(9, 5))
normal_tx   = df[df['anomaly'] ==  1]
anomaly_tx  = df[df['anomaly'] == -1]

plt.scatter(normal_tx['hour'],  normal_tx['amount'], color='steelblue', alpha=0.4, s=15, label='Normal')
plt.scatter(anomaly_tx['hour'], anomaly_tx['amount'], color='red', alpha=0.7, s=25, label='Anomaly')
plt.title('Anomaly Detection -- Amount vs Hour', fontsize=13)
plt.xlabel('Hour of Day')
plt.ylabel('Transaction Amount')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/chart4_anomalies.png')
plt.show()
print("Chart 2 saved.")