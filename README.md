# 🩺 Diabetes Prediction Using Clinical & Lifestyle Data

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Deployed-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-22C55E?style=for-the-badge)

**An end-to-end machine learning pipeline for predicting diabetes risk using patient clinical and lifestyle data.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Hypothesis Testing](#-hypothesis-testing)
- [Machine Learning Models](#-machine-learning-models)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Key Insights](#-key-insights)
- [Limitations](#-limitations)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🔍 Overview

Diabetes is a chronic metabolic condition that affects millions globally and can lead to severe complications — cardiovascular disease, kidney failure, and nerve damage — if not detected early. This project builds a **machine learning-powered prediction system** that identifies patients at risk of diabetes using structured medical and lifestyle data.

> **Goal:** Predict whether a patient is diabetic or non-diabetic using clinical and lifestyle indicators, and deploy the model through an interactive Gradio web interface.

---

## 💼 Business Problem

Healthcare providers need to proactively identify at-risk patients across large populations. Traditional diagnostic methods:

- ⏳ Are time-consuming and rely heavily on clinical testing
- ❌ Do not always capture early risk patterns
- 📉 Miss the combined effect of multiple contributing factors

This project proposes a **data-driven early screening tool** that gives healthcare providers quick, automated risk assessments — enabling timely intervention, better resource allocation, and improved patient outcomes.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Total Records** | ~96,146 patient entries |
| **After Deduplication** | Cleaned and verified |
| **Target Variable** | `diabetes` (1 = Diabetic, 0 = Non-Diabetic) |
| **Task Type** | Binary Classification |
| **Class Distribution** | ~87,664 Non-Diabetic · ~8,482 Diabetic |

### Features Used

| Category | Features |
|---|---|
| **Demographics** | Age, Gender |
| **Clinical Indicators** | HbA1c Level, Blood Glucose Level, BMI |
| **Medical Conditions** | Hypertension, Heart Disease |
| **Lifestyle Factors** | Smoking History |

---

## 🔄 Project Workflow

```
Raw Data (CSV)
     │
     ▼
Data Preprocessing
 ├── Check for missing & duplicate values
 ├── Handle "No Info" in smoking history
 ├── Label encode categorical variables
 └── 80/20 train-test split
     │
     ▼
Exploratory Data Analysis
 ├── Diabetes case distribution
 ├── Age & BMI distributions
 ├── HbA1c & Glucose vs Diabetes (boxplots)
 └── Correlation heatmap
     │
     ▼
Hypothesis Testing
 ├── t-tests (Blood Glucose, HbA1c vs Diabetes)
 ├── Chi-square tests (Hypertension, Heart Disease, Smoking)
 └── Pearson Correlation (HbA1c vs Blood Glucose)
     │
     ▼
Model Training
 ├── Logistic Regression
 ├── Decision Tree
 ├── Random Forest
 └── XGBoost
     │
     ▼
Model Evaluation
 ├── Accuracy, Precision, Recall, F1 Score
 └── Confusion Matrix & Classification Report
     │
     ▼
Deployment (Gradio Interface)
```

---

## 📈 Exploratory Data Analysis

Key insights discovered during EDA:

- 📊 **HbA1c level** and **blood glucose level** showed the strongest association with diabetic outcomes — clear separation between diabetic and non-diabetic groups in boxplots
- ⚖️ **BMI** values were right-skewed, with higher BMI more common among diabetic patients
- 👴 Diabetes risk was concentrated in the **40–70 age range**, with older individuals showing higher likelihood
- 🔗 **Correlation heatmap** confirmed HbA1c and glucose as the strongest positive predictors; most other variables had weak to moderate correlations
- 🚬 Categorical variables (smoking history, hypertension, heart disease) showed weaker visual patterns but still contributed meaningfully in combination

---

## 🧪 Hypothesis Testing

All tests evaluated at **α = 0.05**:

| # | Variable Comparison | Test Type | p-value | Decision |
|---|---|---|---|---|
| 1 | Blood Glucose vs Diabetes | t-test | < 0.0001 | ✅ Reject H₀ |
| 2 | HbA1c Level vs Diabetes | t-test | < 0.0001 | ✅ Reject H₀ |
| 3 | Hypertension vs Diabetes | Chi-square | < 0.0001 | ✅ Reject H₀ |
| 4 | Heart Disease vs Diabetes | Chi-square | < 0.0001 | ✅ Reject H₀ |
| 5 | Smoking History vs Diabetes | Chi-square | < 0.0001 | ✅ Reject H₀ |
| 6 | HbA1c vs Blood Glucose | Pearson Correlation | < 0.0001 | ✅ Reject H₀ |

> All variables were statistically confirmed as significant predictors of diabetes at the 95% confidence level.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression
A linear baseline model using the **sigmoid function** to output diabetes probability. Useful for understanding which features linearly influence outcomes.

- ✅ Highly interpretable
- ✅ Efficient on structured tabular data
- ⚠️ Limited when feature relationships are non-linear

### 2. Decision Tree
A rule-based model that recursively splits data on features like HbA1c, glucose, and BMI using **Gini impurity** / **information gain**.

- ✅ Fully interpretable (visualizable tree)
- ✅ Captures non-linear patterns
- ⚠️ Prone to overfitting on deeper trees

### 3. Random Forest
An ensemble of multiple decision trees, each trained on random data/feature subsets. Final prediction via **majority voting**.

- ✅ Reduces overfitting vs. single Decision Tree
- ✅ Captures complex feature interactions
- ✅ More stable and generalizable

### 4. XGBoost ⭐ *(Best Model)*
A gradient boosting model that builds trees **sequentially**, each correcting errors from the previous one. Optimized for speed and performance.

- ✅ Highest accuracy and recall on this dataset
- ✅ Handles class imbalance better than other models
- ✅ Best for minimizing false negatives (missed diabetic cases)

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.96 | 0.85 | 0.78 | 0.81 |
| Decision Tree | 0.97 | 0.90 | 0.74 | 0.81 |
| Random Forest | 0.98 | 0.93 | 0.86 | 0.89 |
| 🥇 **XGBoost** | **0.99** | **0.95** | **0.89** | **0.92** |

```
XGBoost          ████████████████████████████████████████  99%
Random Forest    ███████████████████████████████████████░  98%
Decision Tree    ██████████████████████████████████████░░  97%
Log. Regression  █████████████████████████████████████░░░  96%
```

> **XGBoost** achieved the best overall performance. Since the dataset is class-imbalanced, **Recall** and **F1 Score** were prioritized — missing a diabetic case has serious healthcare consequences. XGBoost's 89% recall on diabetic cases makes it the most clinically suitable model.

---

## 🚀 Deployment

The best-performing XGBoost model was saved as a `.pkl` file and deployed using **Gradio** — an open-source Python library for building interactive ML demos.

### Features of the App

| Feature | Description |
|---|---|
| 🖥️ **Interactive Web UI** | No coding required — runs in the browser |
| 🧾 **Patient Input Form** | Enter gender, age, hypertension, heart disease, smoking history, BMI, HbA1c, and blood glucose |
| ⚡ **Real-time Prediction** | Instantly classifies as **Diabetic** or **Not Diabetic** |
| 📊 **Probability Score** | Displays model confidence as a percentage |
| 🚦 **Risk Level** | Categorises result as 🟢 Low Risk · 🟡 Moderate Risk · 🔴 High Risk |
| 📋 **Prediction History** | Stores all past predictions in a table for review |
| 🧪 **Sample Examples** | Pre-filled rows to quickly test the app |

### Sample Prediction Outputs

| Age | Hypertension | Heart Disease | Smoking | BMI | HbA1c | Glucose | Prediction | Risk |
|---|---|---|---|---|---|---|---|---|
| 30 | 0 | 0 | 0 | 25 | 5.5 | 100 | Not Diabetic | 🟢 Low Risk |
| 80 | 1 | 1 | 0 | 38 | 6.6 | 140 | Not Diabetic | 🟡 Moderate Risk |
| 40 | 1 | 1 | 3 | 28 | 6.8 | 120 | Not Diabetic | 🟢 Low Risk |
| 70 | 1 | 1 | 4 | 32 | 7.8 | 148 | **Diabetic** | 🔴 High Risk |

### Run the App
Simply run the **Gradio cell** at the bottom of the Jupyter Notebook — it launches the interface instantly in your browser (or generates a shareable public link). No separate `app.py` needed.

---

## 📁 Project Structure

```
diabetes-prediction/
│
├── 📓 HCA.ipynb                          # Main notebook (EDA + modeling + Gradio deployment)
├── 📄 Group14_HCA_Final_Report.pdf       # Full project report
├── 📊 diabetes_prediction_dataset.csv    # Dataset (add to repo or link source)
├── 🤖 diabetes_xgboost_model.pkl         # Saved trained XGBoost model
└── 📝 README.md                          # You are here
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost gradio jupyter
```

### 3. Launch the Notebook & Run the Gradio App
```bash
jupyter notebook HCA.ipynb
```

Once the notebook is open, run all cells. The **Gradio interface** will launch automatically at the bottom — opening in your browser or generating a public shareable link.

---

## 📈 Key Insights

- 🔬 **HbA1c level** and **blood glucose level** are the most powerful predictors — confirmed across EDA, hypothesis testing, and all ML models
- 🌳 **Tree-based models** significantly outperform Logistic Regression, confirming non-linear relationships in the data
- ⚠️ **Class imbalance** (91% non-diabetic) makes Recall and F1 Score more important than accuracy alone
- 🏆 **XGBoost** achieved 89% recall on diabetic cases — the highest among all models tested

---

## ⚠️ Limitations

- The `"No Info"` category in `smoking_history` was treated as a valid category rather than being imputed — this may introduce bias
- Models trained on a single dataset may not generalise well across different populations or healthcare systems
- Class imbalance was not explicitly handled with techniques like **SMOTE** — a potential area for improvement in future work

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Pandas / NumPy** | Data manipulation & preprocessing |
| **Matplotlib / Seaborn** | Data visualization & EDA |
| **Scikit-Learn** | ML model training & evaluation |
| **XGBoost** | Best-performing gradient boosting model |
| **Gradio** | Model deployment & interactive UI |
| **Jupyter Notebook** | Development environment |

---

## 📚 References

- World Health Organization. (2023). [Diabetes Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes)
- Kaggle. [Diabetes Prediction Dataset](https://www.kaggle.com/datasets)
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
- Chen & Guestrin. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.
- American Diabetes Association. (2022). Classification and diagnosis of diabetes. *Diabetes Care, 45*(S1), S17–S38.

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

*Made with ❤️ for healthcare innovation through data science*

</div>
