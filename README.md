# Credit Card Fraud Detection System


## 1. Problem Statement
Credit card fraud is rare but costly. This project builds a machine
learning system to detect fraudulent transactions from anonymized,
real-world transaction data — the key challenge being **severe class
imbalance**: only 0.17% of transactions are fraudulent.

## 2. Dataset
- **Name:** Credit Card Fraud Detection Dataset (Kaggle, ULB Machine Learning Group)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.173%)
- **Features:** Time, Amount, and V1–V28 (PCA-anonymized for privacy)
- **Target:** `Class` (0 = Legit, 1 = Fraud)

## 3. Choosing this topic 
- Real datasets are almost never perfectly balanced — handling this
  correctly is a core applied ML skill, not just calling `.fit()`
- Accuracy is a misleading metric here (a model predicting "always legit"
  would score 99.8% accuracy while catching zero fraud) — so this project
  uses **Precision, Recall, F1, and ROC-AUC** instead
- Demonstrates **SMOTE** (Synthetic Minority Oversampling) applied
  correctly — only on the training set, never on test data, to avoid
  data leakage

## 4. Methodology / Pipeline
1. **EDA** – examine class distribution, confirm 0.173% fraud rate
2. **Feature Scaling** – StandardScaler on `Time` and `Amount`
   (V1–V28 are already PCA-scaled)
3. **Train/Test Split** – 80/20, stratified by class, **before** any
   resampling (critical — prevents synthetic fraud samples from leaking
   into the test set)
4. **SMOTE Oversampling** – applied only to training data, balancing
   227,451 legit vs 227,451 (synthetic) fraud samples
5. **Model Training** – compared 3 classifiers:
   - Logistic Regression
   - Decision Tree
   - Random Forest
6. **Evaluation** – Precision, Recall, F1, ROC-AUC, Confusion Matrix,
   ROC Curve comparison
7. **Deployment** – Streamlit app supporting sample transactions and
   CSV upload

## 5. Results

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 97.43%   | 5.81%     | 91.84% | 10.94% | 0.9698  |
| Decision Tree        | 98.36%   | 7.92%     | 80.61% | 14.43% | 0.8950  |
| **Random Forest**   | 99.95%   | **86.02%**| 81.63% | **83.77%** | **0.9791** |

**Best Model: Random Forest** (selected by F1-score, not accuracy —
see explanation below)

See `model_comparison.png`, `roc_curve.png`, `confusion_matrix.png`,
and `feature_importance.png` for visuals.

## 6. How to Run

```bash
# 1. Install dependencies
pip install pandas scikit-learn imbalanced-learn streamlit matplotlib seaborn

# 2. (Already done) Train the model — regenerates the .pkl files
#    NOTE: Random Forest training takes ~5-6 minutes due to dataset size
python train_model.py

# 3. Launch the demo app
streamlit run app.py
```

The app opens in your browser. It needs `creditcard.csv` in the same
folder for the "sample transaction" feature (CSV upload works without it).

## 7. Project Files
- `train_model.py` — full training pipeline
- `app.py` — Streamlit demo app
- `best_model.pkl` — saved trained model (Random Forest)
- `scaler.pkl` — saved StandardScaler
- `model_meta.pkl` — stores which model won, its F1, and feature columns
- `model_comparison.png` / `.csv` — model comparison chart/table
- `confusion_matrix.png` — confusion matrix for best model
- `roc_curve.png` — ROC curves for all 3 models
- `feature_importance.png` — top 10 most important features
- `creditcard.csv` — dataset (not included in output due to size — see note)

