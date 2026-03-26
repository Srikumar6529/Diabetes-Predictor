# 🏥 Healthcare Risk Prediction System

An end-to-end machine learning system designed to predict diabetes risk using clinical data, with a focus on **high recall and decision support for healthcare applications**.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/healthcare-ml.git
cd healthcare-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
cd src
python train.py
```

This will generate:

* `model.pkl`
* `scaler.pkl`
* `threshold.pkl`

---

### 4. Run the Application

```bash
cd ../app
streamlit run app.py
```

---

## 📊 Exploratory Data Analysis (EDA)

The EDA is provided in `eda.ipynb` and focuses on:

* Understanding feature distributions
* Identifying data quality issues (invalid zero values)
* Analyzing correlations between features and target

### Key Findings:

* Glucose is the strongest predictor of diabetes
* BMI and Age also contribute significantly
* Several features contained invalid zero values representing missing data

### Preprocessing Decisions:

* Median imputation for missing values (robust to outliers)
* Standardization using `StandardScaler`
* Retained outliers to preserve real-world medical variability

---

## 🤖 Model Training & Selection

Multiple models were trained and evaluated:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)

### Evaluation Metrics:

* Accuracy
* Precision
* Recall
* ROC-AUC

---

## 🎯 Key Optimization Strategy

In healthcare applications, **recall is prioritized** to minimize false negatives (i.e., missing high-risk patients).

### Improvements Applied:

* Class imbalance handling (`class_weight="balanced"`)
* Hyperparameter tuning using GridSearchCV
* Threshold tuning for decision optimization

### Final Model:

* **Random Forest (Tuned)**
* ROC-AUC: ~0.83
* Recall: ~0.85 (optimized)

---

## ⚖️ Threshold Tuning

Instead of using the default threshold (0.5), we selected:

👉 **Threshold = 0.4**

This improves recall significantly while maintaining reasonable precision.

---

## 🖥️ Application Features

The Streamlit app provides:

* Interactive patient input
* Risk probability prediction
* Risk categorization (Low / Moderate / High)
* Risk factor analysis
* Basic recommendations

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.
It should not be used as a substitute for professional medical advice.

---

## 📌 Future Work

* Integration with real-world healthcare datasets
* Advanced models (e.g., deep learning)
* Deployment as a web service
* Enhanced explainability (e.g., SHAP values)

---

## 👨‍💻 Author

Built as part of a machine learning project focused on healthcare applications and research-oriented system design.
