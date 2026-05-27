import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('data/transactions_scored.csv')

with open('data/fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

features = ['amount', 'hour', 'is_foreign', 'is_new_device', 'transactions_today', 'account_age_days']

X = df[features]
y = df['is_fraud']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
acc = accuracy_score(y_test, model.predict(X_test))

print("=" * 50)
print("  FINAL REPORT -- Financial Fraud Intelligence")
print("=" * 50)

total = len(df)
fraud = df['is_fraud'].sum()
normal = total - fraud
high_risk = len(df[df['risk_level'] == 'High Risk'])
anomalies = len(df[df['anomaly_label'] == 'Anomaly'])
fraud_pct = round(fraud / total * 100, 1)

print("\nTransaction Summary:")
print(" Total transactions :", total)
print(" Normal :", normal)
print(" Fraudulent :", fraud,f" ({fraud_pct}%)")
print(" High risk flagged :", high_risk)
print(" Anomalies detected :", anomalies)
print("\nModel Accuracy :", round(acc * 100, 1), "%")

print("\nAverage Risk Score by Fraud Status:")
print(df.groupby('is_fraud')['risk_score'].mean().round(1).to_string())

print("\nRisk Level Breakdown:")
print(df['risk_level'].value_counts().to_string())

print("\n" + "=" * 50)
print("REAL TIME TRANSACTION SCREENING:")
print("=" * 50)

new_transactions = [
    {'amount': 50, 'hour': 14, 'is_foreign': 0,
     'is_new_device': 0, 'transactions_today': 2, 'account_age_days': 500},
    {'amount': 3500, 'hour': 3, 'is_foreign': 1,
     'is_new_device': 1, 'transactions_today': 15, 'account_age_days': 5},
    {'amount': 200, 'hour': 11, 'is_foreign': 0,
     'is_new_device': 0, 'transactions_today': 1, 'account_age_days': 900},
    {'amount': 4900, 'hour': 2, 'is_foreign': 1,
     'is_new_device': 1, 'transactions_today': 12, 'account_age_days': 3},
]

for i, tx in enumerate(new_transactions, 1):
    row = pd.DataFrame([tx])
    risk_score = model.predict_proba(row)[0][1] * 100
    decision = "BLOCK -- likely fraud!" if risk_score >= 70 else \
                "REVIEW -- medium risk" if risk_score >= 40 else \
                 "ALLOW -- looks normal"

    print("\nTransaction " + str(i) + ":")
    print(" Amount: " + str(tx['amount']) + "  |  Hour: " + str(tx['hour']) + "  |  Foreign: " + str(tx['is_foreign']))
    print(" Risk Score : " + str(round(risk_score, 1)))
    print(" Decision : " + decision)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Financial Fraud Intelligence Engine -- Dashboard', fontsize=15, fontweight='bold')

axes[0,0].bar(['Normal', 'Fraud'], [normal, fraud], color=['#2ecc71','#e74c3c'], edgecolor='black', width=0.4)
axes[0,0].set_title(f'Fraud Rate: {fraud_pct}%')
axes[0,0].set_ylabel('Transactions')
for i, val in enumerate([normal, fraud]):
    axes[0,0].text(i, val + 5, str(val), ha='center', fontsize=12)

risk_counts = df['risk_level'].value_counts()
axes[0,1].pie(risk_counts.values,
              labels=risk_counts.index,
              colors=['#e74c3c','#f39c12','#2ecc71'],
              autopct='%1.1f%%', startangle=140)
axes[0,1].set_title('Risk Level Distribution')

axes[1,0].hist(df[df['is_fraud']==0]['risk_score'], bins=25, color='steelblue', alpha=0.7, label='Normal')
axes[1,0].hist(df[df['is_fraud']==1]['risk_score'], bins=25, color='red', alpha=0.7, label='Fraud')
axes[1,0].set_title('Risk Score -- Normal vs Fraud')
axes[1,0].set_xlabel('Risk Score')
axes[1,0].set_ylabel('Count')
axes[1,0].legend()

importances = pd.Series(
    model.feature_importances_, index=features
).sort_values(ascending=True)
axes[1,1].barh(importances.index, importances.values, color='steelblue', edgecolor='black')
axes[1,1].set_title('What triggers fraud most?')
axes[1,1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('outputs/chart5_final_dashboard.png', dpi=150)
plt.show()
print("\nDashboard saved: outputs/chart5_final_dashboard.png")


print("\n" + "=" * 50)
print("RECOMMENDATIONS:")
print("=" * 50)
print("""
1. BLOCK HIGH RISK TRANSACTIONS AUTOMATICALLY
   Any transaction with risk score above 70
   should be blocked and flagged for review.

2. MONITOR LATE NIGHT TRANSACTIONS
   Most fraud happens between 12am and 6am.
   Apply stricter checks during these hours.

3. FLAG FOREIGN TRANSACTIONS ON NEW ACCOUNTS
   New accounts making foreign transactions
   is the strongest fraud signal in our data.

4. LIMIT TRANSACTIONS PER DAY
   Fraudsters make many transactions quickly.
   Flag accounts with more than 8 transactions
   in a single day for review.
""")
print("=" * 50)