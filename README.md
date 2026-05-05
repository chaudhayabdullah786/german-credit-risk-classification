# 📊 German Credit Scoring Model

> **Predicting Creditworthiness Using Machine Learning**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset](https://img.shields.io/badge/UCI-German_Credit_Dataset-20BEFF.svg)
![SMOTE](https://img.shields.io/badge/Imbalance-SMOTE_Balanced-purple.svg)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Information](#-dataset-information)
3. [Results Summary](#-results-summary)
4. [Key Findings — Feature Importance](#-key-findings)
5. [Cost Analysis](#-cost-analysis)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Technical Details](#-technical-details)
9. [Repository Structure](#-repository-structure)
10. [Future Improvements](#-future-improvements)
11. [Author](#-author)
12. [License](#-license)

---

## 🎯 Project Overview

This project builds a credit scoring model to predict whether a customer has **good** or **bad** credit risk using the [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) from the UCI Machine Learning Repository. The model helps financial institutions make informed lending decisions by identifying high-risk applicants.

**Key Objectives:**

- ✅ Predict creditworthiness with high accuracy
- ✅ Handle imbalanced data (70% Good / 30% Bad credit)
- ✅ Minimize costly false negatives (missing bad customers)
- ✅ Provide interpretable feature importance

---

## 📊 Dataset Information

| Aspect | Details |
|--------|---------|
| **Source** | UCI Machine Learning Repository — Statlog German Credit Data |
| **Samples** | 1,000 credit applicants |
| **Features** | 20 total (7 numerical, 13 categorical) |
| **Target** | Good Credit (70%) / Bad Credit (30%) |
| **Missing Values** | None |

### Target Distribution

![Target Distribution](https://raw.githubusercontent.com/chaudhayabdulah786/german-credit-risk-classification/main/images/credit_risk_class.png)

| Class | Count | Percentage |
|-------|-------|------------|
| ✅ Good Credit (0) | 700 | 70.0% |
| ❌ Bad Credit (1) | 300 | 30.0% |

> ⚠️ **Note:** The dataset is imbalanced (30% default rate). SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set to balance class distribution and prevent model bias.

---

## 📈 Results Summary

### Model Performance Comparison

![Model Performance](https://raw.githubusercontent.com/chaudhayabdulah786/german-credit-risk-classification/main/images/model_performance.png)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.735 | 0.544 | **0.717** | 0.619 | **0.792** |
| Decision Tree | 0.655 | 0.437 | 0.517 | 0.473 | 0.597 |
| Random Forest | **0.765** | **0.623** | 0.550 | 0.584 | 0.787 |

### ROC Curve

![ROC Curve](https://raw.githubusercontent.com/chaudhayabdulah786/german-credit-risk-classification/main/images/roc_curve.png)

### 🏆 Best Model: Logistic Regression

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | **0.792** | Primary evaluation metric |
| Recall | **0.717** | Catches 71.7% of bad credit applicants |
| F1-Score | **0.619** | Best balance between precision and recall |

---

## 🔍 Key Findings

### Top 10 Most Important Features

![Feature Importance](https://raw.githubusercontent.com/chaudhayabdulah786/german-credit-risk-classification/main/images/feature.png)

| Rank | Feature | Coefficient | Impact |
|------|---------|-------------|--------|
| 1 | Attribute 8 | +0.355 | ⬆️ Increases Risk |
| 2 | Attribute 12 | +0.350 | ⬆️ Increases Risk |
| 3 | Attribute 2 | +0.333 | ⬆️ Increases Risk |
| 4 | Attribute 5 | +0.217 | ⬆️ Increases Risk |
| 5 | Attribute 16 | +0.073 | ⬆️ Increases Risk |
| 6 | Attribute 18 | +0.067 | ⬆️ Increases Risk |
| 7 | Attribute 17 | +0.019 | ⬆️ Increases Risk |
| 8 | Attribute 13 | -0.003 | ⬇️ Decreases Risk |
| 9 | Attribute 10 | -0.054 | ⬇️ Decreases Risk |
| 10 | Attribute 4 | -0.058 | ⬇️ Decreases Risk |

**Interpretation:**
- 🔴 **Positive coefficients** → Higher values increase predicted credit risk
- 🟢 **Negative coefficients** → Higher values decrease credit risk (protective factors)

---

## 💰 Cost Analysis

### Business Cost Matrix

|  | Predicted: Good | Predicted: Bad |
|--|----------------|----------------|
| **Actual: Good** | Cost: 0 ✅ | Cost: 1 (False Positive) |
| **Actual: Bad** | Cost: 5 💸 **(False Negative)** | Cost: 0 ✅ |

> 💡 **False Negatives are 5× more costly than False Positives** in real-world lending. The model is explicitly optimized to minimize this asymmetric cost.

### Threshold Optimization

| Metric | Value |
|--------|-------|
| Initial Model Cost | 121 units |
| Optimal Decision Threshold | **0.30** (default 0.50 is suboptimal) |
| Minimum Achievable Cost | **107 units** |
| Cost Reduction | **11.6% improvement** |

### Confusion Matrix at Optimal Threshold (= 0.30)

|  | Predicted Good | Predicted Bad |
|--|----------------|---------------|
| **Actual Good** | 73 ✅ True Negative | 67 False Positive |
| **Actual Bad** | 8 ❌ False Negative | 52 ✅ True Positive |

- ✅ **73** good customers correctly approved
- ✅ **52** bad customers correctly identified and declined
- ❌ **8** bad customers missed — reduced from 17 at default threshold
- ⚠️ **67** good customers incorrectly declined

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step-by-Step

**1. Clone the repository**
```bash
git clone https://github.com/chaudhayabdulah786/german-credit-risk-classification.git
cd german-credit-risk-classification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook german_credit_scoring.ipynb
```

### `requirements.txt`
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
ucimlrepo==0.0.3
```

---

## 🚀 Usage

### Load and Use the Trained Model

```python
import joblib
import pandas as pd

# Load trained model and preprocessors
model    = joblib.load('models/german_credit_model.pkl')
scaler   = joblib.load('models/feature_scaler.pkl')
features = joblib.load('models/feature_names.pkl')

# Prepare a new customer record (all 20 features required)
new_customer = pd.DataFrame([{
    'Attribute1': 1, 'Attribute2': 30, 'Attribute3': 2,
    'Attribute4': 3, 'Attribute5': 5000, 'Attribute6': 2,
    # ... include all 20 features
}])

# Scale and predict
customer_scaled = scaler.transform(new_customer[features])
probability = model.predict_proba(customer_scaled)[0, 1]

# Apply optimal threshold
if probability < 0.30:
    print(f"✅ APPROVED — Default probability: {probability:.1%}")
else:
    print(f"❌ DECLINED — Default probability: {probability:.1%}")
```

---

## 🔬 Technical Details

### Data Preprocessing

| Step | Method |
|------|--------|
| Categorical encoding | Label Encoding (13 features) |
| Feature scaling | StandardScaler (mean=0, std=1) |
| Train/test split | 80/20 with stratification |
| Class imbalance | SMOTE (560:240 → 560:560) |

### Algorithms Evaluated

| Algorithm | Strength | Best Metric |
|-----------|---------|-------------|
| **Logistic Regression** | Best recall & ROC-AUC | ROC-AUC: 0.792 |
| Random Forest | Highest accuracy | Accuracy: 76.5% |
| Decision Tree | Fast training | — |

### Evaluation Strategy

- **Primary metric:** ROC-AUC (threshold-independent discriminative ability)
- **Secondary metrics:** Recall (minimizing missed defaults), F1-Score, business cost matrix
- **Threshold tuned** to 0.30 to minimize total asymmetric lending cost

---

## 📁 Repository Structure

```
german-credit-risk-classification/
│
├── README.md                            # Project documentation
├── german_credit_scoring.ipynb          # Main notebook — full pipeline
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License
│
├── images/
│   ├── credit_risk_class.png            # Target distribution chart
│   ├── model_performance.png            # Model comparison chart
│   ├── roc_curve.png                    # ROC curve visualization
│   └── feature.png                      # Feature importance chart
│
├── models/
│   ├── german_credit_model.pkl          # Trained Logistic Regression model
│   ├── feature_scaler.pkl               # Fitted StandardScaler
│   └── feature_names.pkl                # Ordered feature columns
│
└── notebooks/
    └── exploratory_analysis.ipynb       # EDA notebook
```

---

## 💡 Future Improvements

- [ ] **Hyperparameter Tuning** — GridSearchCV for optimal parameters
- [ ] **XGBoost / Gradient Boosting** — benchmark against ensemble methods
- [ ] **SHAP Analysis** — enhanced interpretability for regulatory compliance
- [ ] **API Deployment** — Flask / FastAPI REST endpoint for production scoring
- [ ] **Model Monitoring** — drift detection to track performance over time
- [ ] **Feature Engineering** — interaction terms and domain-specific features
- [ ] **Streamlit Dashboard** — interactive UI for non-technical stakeholders

---

## 📚 References

- [UCI German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*
- [Scikit-learn Documentation](https://scikit-learn.org/)
- Thomas, Edelman & Crook — *Credit Scoring and Its Applications* (SIAM, 2002)

---

## 👨‍💻 Author

**Muhammad Abdullah** — Data Analyst & Machine Learning Engineer

[![GitHub](https://img.shields.io/badge/GitHub-chaudhayabdulah786-181717?logo=github)](https://github.com/chaudhayabdulah786)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad_Abdullah-0A66C2?logo=linkedin)](https://www.linkedin.com/in/muhammad-abdullah-65608a248)
[![Portfolio](https://img.shields.io/badge/Portfolio-pythonanywhere.com-orange)](https://mabdullahdataanalyst786.pythonanywhere.com/)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Copyright (c) 2024 Muhammad Abdullah
```

---

⭐ **If you found this project useful, please give it a star on GitHub!**
