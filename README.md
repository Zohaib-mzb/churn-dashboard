# 📊 Customer Churn Prediction Dashboard

A business-ready machine learning dashboard that predicts 
which telecom customers are likely to leave — and explains 
exactly why using SHAP explainability.

🔗 **Live Demo:** https://churndashboardai.streamlit.app/

---

## What it does

- **Overview page** — KPIs, churn rate, and key business findings
- **Segment Analysis** — Interactive churn breakdown by contract, 
  tenure, payment method, internet service, and support services
- **Live Prediction** — Enter any customer's details and get an 
  instant churn probability with AI explanation of top risk factors

---

## Key Findings from the Data

| Segment | Churn Rate | Insight |
|---|---|---|
| Month-to-month contract | ~43% | Strongest churn predictor |
| 0–12 month tenure | ~48% | Critical early retention window |
| Electronic check payment | ~45% | Auto-pay migration reduces risk |
| Fibre optic internet | ~42% | High expectations, high churn |
| No tech support | ~41% | Support services = retention tools |

---

## ML Model Performance

| Model | F1-Score | ROC-AUC |
|---|---|---|
| Logistic Regression | ~70% | ~84% |
| Decision Tree | ~65% | ~77% |
| Random Forest | ~72% | ~86% |
| **XGBoost ✓** | **~73%** | **~87%** |

XGBoost selected as production model — best ROC-AUC and 
Recall balance, handles class imbalance via scale_pos_weight.

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · 
Streamlit · Matplotlib · Seaborn

---

## Run Locally

\`\`\`bash
git clone https://github.com/Zohaib-mzb/churn-dashboard
cd churn-dashboard
pip install -r requirements.txt
streamlit run app.py
\`\`\`

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features.  
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

*Synthetic data created by IBM for educational purposes.*

---

**Built by Muhammad Zohaib**  
[GitHub](https://github.com/Zohaib-mzb) · 
[LinkedIn](https://www.linkedin.com/in/muhammad-zohaib-312a08363/)
