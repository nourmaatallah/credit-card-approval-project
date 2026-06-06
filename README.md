#Credit Card Approval Prediction — Machine Learning Pipeline

> A personal end-to-end machine learning project to predict credit card and loan approval decisions using real-world financial data.

---

## Project Overview

This project builds a complete ML pipeline that predicts whether a credit card or loan application should be approved or denied. Rather than relying on fixed scoring rules, the model analyzes a broader set of applicant signals — income, credit history, debts... — to generate more accurate and explainable decisions.

The project covers the full ML lifecycle: data collection, cleaning, exploratory analysis, feature engineering, model training, evaluation, and deployment via a Streamlit web app.

---

##Objectives

- Predict credit approval outcomes from real applicant financial data
- Compare multiple classification models and select the best performer
- Handle real-world challenges such as class imbalance and categorical encoding
- Deploy an interactive prediction interface accessible to non-technical users
- Ensure model transparency using SHAP explainability

---

## Project Structure

```
credit-approval-prediction/
│
├── models/
│   ├── credit_card_model.pkl             # Saved best model
│   └── model_columns.pkl     # Training column order for inference
│
├── app/
│   └── app.py                # Streamlit deployment app
│
├── requirements.txt
└── README.md
```

---

##  Project Phases

| Phase | Description |
|-------|-------------|
| **1. Data Collection** | Real dataset sourced from Kaggle / UCI Credit Card repository |
| **2. Data Cleaning** | Missing values, outliers |
| **3. EDA** | Correlation heatmaps, distribution plots, approval rate analysis |
| **4. Feature Engineering** | Encoding, scaling |
| **5. Modeling** | Logistic Regression → Random Forest with GridSearchCV tuning → XGBoost with RandomRandomizedSearchCV tuning  |
| **6. Evaluation** | ROC-AUC, accuracy, confusion matrix, classification report, SHAP feature importance |
| **7. Deployment** | Streamlit app for live predictions |

---

##  Models Used

Three models were trained and compared in progression:

- **Logistic Regression** — baseline model, simple and interpretable
- **Random Forest** — captures non-linear relationships via ensemble learning
- **XGBoost** — final model, gradient boosting

Final model selected: RandomForest based on ROC-AUC and ovrerall metrics on the test set.

---

##  Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.x |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML framework | Scikit-learn |
| Boosting | XGBoost |
| Explainability | SHAP |
| Imbalanced data | Imbalanced-learn |
| Deployment | Streamlit |
| Notebooks | GoogleColab |
| Version control | GitHub |

---


##  Results

| Model | ROC-AUC | F1-score |
|-------|---------|----------|
| Logistic Regression | 0.82 | 0.81 |
| Random Forest |0.87 | 0.86 |
| XGBoost | 0.85 | 0.84 |



---

##  Dataset

- **Source:** [Kaggle Credit Card Approval Dataset](https://www.kaggle.com/) 
- **Features:** Gender, Age, Debt, Married, Industtry, PriorDefault, Employed and more.         
- **Target:** Binary — Approved (1) / Denied (0)

---

##  Explainability

SHAP (SHapley Additive exPlanations) is used to interpret the final model's predictions. This ensures that approval and denial decisions are transparent and justifiable — a key requirement in real-world financial applications.

---

##  Author

**NourMaatallah** — Personal Project  
Built as part of a self-directed learning path in machine learning and AI applied to FinTech.

---

##  License

This project is for educational and portfolio purposes only.
