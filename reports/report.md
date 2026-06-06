# Customer Engagement & Product Utilization Analytics for Retention Strategy
### European Central Bank | Behavioral Churn Analysis

---

## 1. Introduction

Banks often focus on demographics like salary and credit score to predict churn, but this project takes a different approach. The idea is that **how a customer behaves** — whether they are active, how many products they use — is a stronger signal of whether they will leave.

This analysis uses a dataset of 10,000 European bank customers to evaluate engagement patterns, product utilization, and their combined impact on churn.

---

## 2. Dataset Overview

The dataset contains 10,000 records across 14 columns covering customers from France, Spain, and Germany.

| Column | Description |
|---|---|
| CustomerId | Unique identifier |
| Geography | France, Spain, Germany |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Credit card ownership (0/1) |
| IsActiveMember | Active status (0/1) |
| EstimatedSalary | Annual salary estimate |
| Exited | Churn — target variable (0/1) |

**Data Quality:** No null values, no duplicates. Binary columns validated. Dataset was ready for analysis without any cleaning required.

---

## 3. Methodology

The analysis follows a 4-stage pipeline:

**Stage 1 — Engagement Classification**
Customers were grouped into 4 segments based on activity status, product count, and balance level.

**Stage 2 — Product Utilization Analysis**
Churn rates were calculated across product counts to find the optimal product depth for retention.

**Stage 3 — Financial Commitment vs Engagement**
Cross-analysis of balance and activity to identify high-value customers at silent churn risk.

**Stage 4 — Retention Strength Scoring**
A Relationship Strength Index (RSI) was built as a composite score combining 5 behavioral signals.

---

## 4. Exploratory Data Analysis

### Overall Churn

The overall churn rate is **20.37%**, meaning about 1 in 5 customers left the bank.

### Churn by Geography

Germany shows significantly higher churn compared to France and Spain.

| Country | Churn Rate |
|---|---|
| Germany | 32.44% |
| Spain | 16.67% |
| France | 16.15% |

Germany's elevated churn could point to a service gap, product mismatch, or regional competitive pressure. It needs targeted investigation.

### Churn by Gender

Female customers churn at 25.07% compared to 16.46% for male customers — a notable gap that suggests retention strategies may need to be gender-aware.

---

## 5. Engagement Classification

Customers were classified into 4 segments:

| Segment | Criteria | Count | Churn Rate |
|---|---|---|---|
| Loyal Core | Active + 2+ products | 2,588 | 9.66% |
| Cross-sell Target | Active + 1 product | 2,563 | 18.92% |
| At-Risk Premium | Inactive + high balance | 2,456 | 32.33% |
| Silent Churner | Inactive + low balance | 2,393 | 21.23% |

The **At-Risk Premium** segment is the most concerning — these customers have high balances but are not engaged, making them the highest churn risk in the dataset.

---

## 6. KPI Analysis

### KPI 1 — Engagement Retention Ratio (ERR)

**ERR = Inactive Churn Rate ÷ Active Churn Rate = 1.88x**

- Active member churn: **14.27%**
- Inactive member churn: **26.85%**

Inactive members are 1.88x more likely to churn. This is the strongest single predictor in the dataset. Engagement is the #1 retention lever.

---

### KPI 2 — Product Depth Index (PDI)

**PDI = Retention rate of 2-product customers = 92.42%**

| Products | Churn Rate |
|---|---|
| 1 product | 27.71% |
| 2 products | 7.58% |
| 3 products | 82.71% |
| 4 products | 100.00% |

2 products is clearly the retention sweet spot. The sharp rise at 3+ products likely reflects forced cross-selling where customers were signed up for products they didn't need, leading to dissatisfaction and eventual exit.

---

### KPI 3 — High-Balance Disengagement Rate (HBDR)

**HBDR = 32.33%**

1,662 customers are currently inactive with above-median balances but have not yet churned. These are the most valuable at-risk customers — they hold significant assets but show no engagement signals. Without intervention, they are the most likely group to churn silently.

---

### KPI 4 — Credit Card Stickiness Score (CSS)

**CSS = 0.63 percentage points**

Credit card ownership has almost no independent effect on retention. The difference between card holders and non-holders is less than 1pp. This means credit cards alone are not a retention tool — engagement and product bundling matter far more.

---

### KPI 5 — Relationship Strength Index (RSI)

RSI is a composite score out of 100 built from:
- IsActiveMember → 30 pts
- NumOfProducts (max 3) → 30 pts
- HasCrCard → 10 pts
- Tenure (normalized) → 20 pts
- High Balance → 10 pts

| Group | Avg RSI |
|---|---|
| Retained customers | 53.93 |
| Churned customers | 48.27 |
| RSI Gap | 5.66 pts |

Retained customers consistently score higher on RSI. Customers with RSI below 30 show significantly elevated churn and can be flagged as early warning cases.

---

## 7. Bottleneck Identification

The analysis points to two main bottlenecks:

**Bottleneck 1 — Single Product Customers**
5,084 customers (50.8%) hold only 1 product. Their churn rate is 27.71% — nearly 4x higher than 2-product customers. Converting even a fraction of these to a 2-product bundle would have the highest retention impact.

**Bottleneck 2 — Inactive High-Balance Customers**
1,662 retained customers are inactive with above-median balances. These are premium customers who could leave at any time. They need personalized re-engagement before it's too late.

---

## 8. Recommendations

**For Cross-sell Target segment (Active + 1 product):**
Push targeted 2-product bundle offers. A move from 1 to 2 products reduces churn from 27.7% to 7.6% — this is the highest ROI retention action available.

**For At-Risk Premium segment (Inactive + high balance):**
Assign relationship managers for personal outreach. Exclusive offers, loyalty rewards, and check-in calls should be prioritized for this group.

**For Germany:**
Conduct region-specific customer feedback. The 32.44% churn rate is double that of France and needs dedicated investigation into local service quality, product fit, or competitor activity.

**For RSI < 30 customers:**
Use RSI as an early warning flag. Customers scoring below 30 should be automatically flagged in CRM systems for proactive retention campaigns.

**General:**
Avoid forcing 3+ products on customers. The data shows 100% churn for 4-product customers and 82.7% for 3-product customers, suggesting over-selling actively drives customers away.

---

## 9. Conclusion

The analysis confirms that behavioral engagement — not financial strength — is the primary driver of customer retention. A high balance does not prevent churn if the customer is inactive. Two products is the optimal relationship depth. And inactive high-balance customers represent the most urgent and addressable churn risk in the portfolio.

These findings support a shift from demographic-based to engagement-based retention strategy at the bank level.

---

*Dataset: European Bank Customer Churn | 10,000 records | ECB Mentorship Project*

---

## 10. Machine Learning — Churn Prediction

### Model: Random Forest Classifier

A Random Forest model with 100 decision trees was trained on all 10,000 customers to predict churn probability and validate the EDA findings through feature importance analysis.

**Setup:**
- Train/Test Split: 80% / 20% (random state = 42)
- Features: 11 variables including engineered RSI score
- Target: Exited (0 = Retained, 1 = Churned)
- Geography and Gender were label-encoded

### Model Performance

| Metric | Value |
|---|---|
| Accuracy | 86.75% |
| Precision (churn class) | 78.57% |
| Recall (churn class) | 44.78% |
| F1 Score | 57.05% |

The model performs well on the majority retained class. The lower recall on churned customers is expected given the class imbalance (20% churn rate) and can be improved with SMOTE or class weighting in future iterations.

### Feature Importance

The Random Forest feature importance ranking confirms the EDA findings:

- **Age** and **Balance** are the top predictors — older customers and those with high balances but no engagement are most at risk
- **IsActiveMember** ranks highly — directly validating the ERR KPI finding
- **RSI** (engineered feature) performs strongly — confirming it captures meaningful churn signal
- **NumOfProducts** is important — consistent with the PDI analysis
- **CreditScore** and **EstimatedSalary** rank lower — confirming demographics alone are weak predictors

### ML-Predicted Churn Probability by Segment

| Segment | ML-Predicted Churn Probability |
|---|---|
| At-Risk Premium | Highest |
| Silent Churner | High |
| Cross-sell Target | Moderate |
| Loyal Core | Lowest |

The ML predictions align closely with the EDA-derived churn rates — this cross-validation confirms that the behavioral segmentation framework is statistically sound.

### Why Random Forest?

Random Forest was chosen because it handles mixed data types without scaling, is robust to outliers, provides interpretable feature importance scores, and performs well on imbalanced datasets without extensive tuning.

---

## 11. Updated Conclusion

The machine learning analysis validates and extends the EDA findings:

1. Behavioral features (activity, product count, RSI) outperform demographic features (salary, credit score) in predicting churn
2. The engineered RSI score is one of the strongest predictors — confirming its value as an operational early-warning metric
3. The At-Risk Premium segment has the highest ML-predicted churn probability, reinforcing the urgency of re-engagement campaigns for inactive high-balance customers

The combination of EDA, KPI framework, and ML prediction provides a complete, actionable retention intelligence system for the bank.
