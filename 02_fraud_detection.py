import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

df = pd.read_csv('data/transactions.csv')
os.makedirs('outputs', exist_ok=True)

print("Data loaded!")
print("Total transactions:", len(df))
print("Fraud cases :", df['is_fraud'].sum())
print("Normal cases :", (df['is_fraud']==0).sum())

features = ['amount', 'hour', 'is_foreign', 'is_new_device', 'transactions_today', 'account_age_days']
X = df[features]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining fraud detection model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("\nModel Accuracy:", round(accuracy * 100, 1), "%")
print("\nDetailed Report:")
print(classification_report(y_test, preds, target_names=['Normal', 'Fraud']))

with open('data/fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to: data/fraud_model.pkl")

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6, 5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Fraud'],
            yticklabels=['Normal','Fraud'])
plt.title('Confusion Matrix\n(diagonal = correct predictions)', fontsize=13)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/chart1_confusion_matrix.png')
plt.show()
print("Chart 1 saved.")

importances = pd.Series(
    model.feature_importances_, index=features
).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Feature Importance -- What causes fraud?', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/chart2_feature_importance.png')
plt.show()
print("Chart 2 saved.")