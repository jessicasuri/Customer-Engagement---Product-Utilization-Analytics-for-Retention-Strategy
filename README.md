# 🏦 Customer Engagement & Product Utilization Analytics for Retention Strategy
### European Central Bank | Behavioral Churn Analysis

---

## 📌 Project Overview

This project analyzes **customer behavioral patterns** in a European bank to identify the key drivers of churn. Rather than relying on demographics alone, the analysis focuses on **engagement levels** and **product utilization** to build actionable retention strategies.

**Mentor:** European Central Bank
**Dataset:** 10,000 bank customers across France, Spain, and Germany

---

## 🎯 Objectives

- Evaluate the relationship between customer engagement and churn
- Measure the retention impact of product count and product mix
- Identify disengaged yet high-value customers at risk of silent churn
- Predict churn using machine learning (Random Forest)
- Support engagement-driven retention strategies

---

## 📊 Dataset Description

| Column | Description |
|--------|-------------|
| CustomerId | Unique customer identifier |
| Geography | France, Spain, Germany |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance (€) |
| NumOfProducts | Number of bank products held |
| HasCrCard | Credit card ownership (0/1) |
| IsActiveMember | Activity indicator (0/1) |
| EstimatedSalary | Estimated annual salary (€) |
| Exited | Churn indicator — Target variable (0/1) |

---

## 🔑 Key Performance Indicators (KPIs)

| KPI | Formula | Value |
|-----|---------|-------|
| **Engagement Retention Ratio (ERR)** | Inactive Churn % ÷ Active Churn % | 1.88x |
| **Product Depth Index (PDI)** | 1 − Churn Rate (2-product customers) | 92.4% |
| **High-Balance Disengagement Rate (HBDR)** | Churn % \| Inactive & Balance > Median | 32.3% |
| **Credit Card Stickiness Score (CSS)** | Churn%(no card) − Churn%(has card) | 0.63pp |
| **Avg Relationship Strength Index (RSI)** | IsActive×30 + Products×30 + Card×10 + Tenure×20 + Balance×10 | /100 |

---

## 🤖 Machine Learning

A **Random Forest Classifier** (100 trees) was trained on all 10,000 customers to:
- Predict individual customer churn probability
- Identify the most important features driving churn
- Score each engagement segment with ML-predicted churn probability

The dashboard includes an **Individual Customer Churn Predictor** where you can input any customer's details and get an instant churn probability score.

---

## 📂 Project Structure

```
customer-engagement-retention-analytics/
├── app/
│   └── app.py                 ← Streamlit Dashboard (5 tabs + ML)
├── REPORTS/
│   ├── report.md              ← Full EDA Research Report
│   └── executive_summary.docx ← Executive Summary for Stakeholders
├── VISUALS/                   ← Charts generated from notebook
├── European_Bank.csv          ← Dataset
├── EDA_Analysis.ipynb         ← Jupyter Notebook — Full EDA & Analysis
├── README.md                  ← This file
└── requirements.txt           ← Python dependencies
```

---

## 🚀 How to Run

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-engagement-retention-analytics.git
cd customer-engagement-retention-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/app.py
```

### Deploy on Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app/app.py`
5. Click **Deploy** → Get your live link!

---

## 📈 Dashboard Modules

| Tab | Content |
|-----|---------|
| 📊 Engagement Overview | KPI cards with formulas, churn by activity, segment distribution, geography & gender |
| 📦 Product Utilization | Churn by product count, single vs multi-product, product mix by geography |
| ⚠️ At-Risk Customers | High-value disengaged detector, balance heatmap, top 20 table, CSV export |
| 🏆 Retention Scoring | RSI distribution, churn by RSI tier, segment scatter, KPI summary table |
| 🤖 ML Churn Prediction | Random Forest model, feature importance, segment churn probabilities, individual predictor |

### Sidebar Features
- Geography filter, Engagement status toggle, Product count slider
- Balance & Salary range sliders
- **⬇️ Download filtered data as CSV**

---

## 💡 Key Insights

1. **Engagement is #1** — Inactive members churn at 26.9% vs 14.3% for active (1.88x gap)
2. **2-product sweet spot** — 2-product customers churn at only 7.6% vs 27.7% single-product
3. **Silent churn risk** — 1,662 high-balance inactive customers still retained but at high risk (32.3%)
4. **Germany anomaly** — Germany churn (32.4%) is double France and Spain
5. **RSI predicts churn** — Retained customers average 5.66 RSI points higher than churned

---

## 🧑‍💼 Engagement Segments

| Segment | Criteria | Churn Rate | Strategy |
|---------|----------|------------|----------|
| 🟢 Loyal Core | Active + 2+ products | 9.66% | Reward & deepen |
| 🔵 Cross-sell Target | Active + 1 product | 18.92% | Push 2nd product bundle |
| 🟠 At-Risk Premium | Inactive + high balance | 32.33% | Urgent re-engagement |
| 🔴 Silent Churner | Inactive + low balance | 21.23% | Win-back campaigns |

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Visualizations
- **Scikit-learn** — Random Forest ML model
- **Streamlit** — Interactive dashboard
- **Jupyter Notebook** — EDA & analysis

---

## 📝 Reports

- **Research Report:** `REPORTS/report.md`
- **Executive Summary:** `REPORTS/executive_summary.docx`

---

## 👤 Author

**Jessica Suri**
