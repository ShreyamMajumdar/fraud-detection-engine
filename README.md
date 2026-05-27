# 🔐 Financial Fraud Intelligence Engine
 
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Domain](https://img.shields.io/badge/Domain-FinTech-yellow)
 
## 📌 Overview
A financial fraud detection system that analyzes bank transactions,
assigns a risk score from 0 to 100 to every transaction and makes
real-time Allow, Review or Block decisions using Random Forest
classification and Isolation Forest anomaly detection.
 
## 🎯 Objective
- Detect fraudulent bank transactions using machine learning
- Assign a risk score (0-100) to every transaction
- Make real-time Allow / Review / Block decisions
- Detect unusual transactions using unsupervised anomaly detection
 
## 📂 Project Structure
```
fraud_project/
│
├── data/
│   ├── transactions.csv
│   ├── transactions_scored.csv
│   └── fraud_model.pkl
│
├── outputs/
│   └── (5 charts saved here)
│
├── 01_generate_data.py
├── 02_fraud_detection.py
├── 03_risk_scoring.py
└── 04_insights.py
```
 
## 📊 Dataset
- **Type:** Synthetic bank transaction data
- **Size:** 1200 rows (1000 normal + 200 fraudulent)
- **Features:** amount, hour, is_foreign, is_new_device,
  transactions_today, account_age_days
 
## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| pandas | Data handling |
| numpy | Data generation |
| matplotlib | Charts and visualizations |
| seaborn | Confusion matrix heatmap |
| scikit-learn | Random Forest, Isolation Forest, metrics |
| pickle | Save and load model |
 
## 🚨 Fraud Signals Used
| Feature | Normal | Fraudulent |
|---------|--------|------------|
| Amount | Rs. 10-500 | Rs. 500-5000 |
| Hour | 8am-10pm | 12am-6am |
| Foreign | 10% chance | 80% chance |
| New Device | 15% chance | 80% chance |
| Txns Today | 1-5 | 8-20 |
| Account Age | 30-3000 days | 1-30 days |
 
## 🎯 Risk Scoring System
| Risk Score | Label | Decision |
|------------|-------|----------|
| 0 to 39 | Low Risk | ALLOW |
| 40 to 69 | Medium Risk | REVIEW |
| 70 to 100 | High Risk | BLOCK |
 
## 🤖 Models Used
| Model | Type | Purpose |
|-------|------|---------|
| Random Forest Classifier | Supervised | Fraud detection with labels |
| Isolation Forest | Unsupervised | Anomaly detection without labels |
 
## 📈 Key Findings
- Model achieved very high accuracy in detecting fraud
- account_age_days was the most important feature
- Late night transactions (12am-6am) had 4x higher fraud rate
- Isolation Forest flagged suspicious patterns without using any labels
- Real-time screening evaluates a transaction in under 1 millisecond
 
## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
 
python 01_generate_data.py
python 02_fraud_detection.py
python 03_risk_scoring.py
python 04_insights.py
```
