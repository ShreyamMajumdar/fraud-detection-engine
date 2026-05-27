import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 1000

normal = pd.DataFrame({
    'amount' : np.random.randint(10, 500, n),
    'hour' : np.random.randint(8, 22, n),
    'is_foreign' : np.random.choice([0,1], n, p=[0.9, 0.1]),
    'is_new_device' : np.random.choice([0,1], n, p=[0.85, 0.15]),
    'transactions_today' : np.random.randint(1, 5, n),
    'account_age_days' : np.random.randint(30, 3000, n),
    'is_fraud' : 0
})

fraud = pd.DataFrame({
    'amount' : np.random.randint(500, 5000, 200),
    'hour' : np.random.randint(0, 6, 200),
    'is_foreign' : np.random.choice([0,1], 200, p=[0.2, 0.8]),
    'is_new_device' : np.random.choice([0,1], 200, p=[0.2, 0.8]),
    'transactions_today' : np.random.randint(8, 20, 200),
    'account_age_days' : np.random.randint(1, 30, 200),
    'is_fraud' : 1
})

df = pd.concat([normal, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, 'transaction_id', range(1, len(df)+1))

os.makedirs('data', exist_ok=True)
df.to_csv('data/transactions.csv', index=False)

print("Dataset created!")
print("Total transactions :", len(df))
print("Normal transactions:", len(df[df['is_fraud']==0]))
print("Fraud transactions :", len(df[df['is_fraud']==1]))
print("\nFirst 5 rows:")
print(df.head())