import pandas as pd         #type:ignore
import numpy as np           #type:ignore
import pickle 
import time 

from sklearn.model_selection import train_test_split  #type:ignore
from sklearn.preprocessing import StandardScaler      #type:ignore
from sklearn.linear_model import LogisticRegression    #type:ignore
from sklearn.ensemble import RandomForestClassifier     #type:ignore
from sklearn.tree import DecisionTreeClassifier        #type:ignore
from sklearn.metrics import (  #type:ignore
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE  #type:ignore

import matplotlib #type:ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  #type:ignore
import seaborn as sns            #type:ignore

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("creditcard.csv")
print(f"Total transactions: {len(df)}")
print(f"Fraud cases: {df['Class'].sum()}  ({100*df['Class'].mean():.3f}%)")
print(f"Legit cases: {(df['Class']==0).sum()}")

# -----------------------------
# 2. FEATURE SCALING
# -----------------------------
# V1-V28 are already PCA-transformed (scaled). Only 'Time' and 'Amount' need scaling.
print("\nScaling Time and Amount...")
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
df = df.drop(["Time", "Amount"], axis=1)

# Reorder so scaled columns are up front (cosmetic, not required)
cols = ["scaled_amount", "scaled_time"] + [c for c in df.columns if c not in ["scaled_amount", "scaled_time", "Class"]] + ["Class"]
df = df[cols]

# -----------------------------
# 3. TRAIN / TEST SPLIT (BEFORE SMOTE - critical to avoid data leakage)
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  (fraud: {y_train.sum()})")
print(f"Test size:  {len(X_test)}  (fraud: {y_test.sum()})")

# -----------------------------
# 4. HANDLE IMBALANCE WITH SMOTE (on training data only!)
# -----------------------------
print("\nApplying SMOTE to balance training data...")
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")

# -----------------------------
# 5. TRAIN MULTIPLE MODELS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
best_model = None
best_model_name = None
best_f1 = 0  # Using F1, not accuracy, because accuracy is misleading on imbalanced data

for name, model in models.items():
    print(f"\nTraining {name}...")
    t0 = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - t0 

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "train_time": train_time, "y_pred": y_pred, "y_proba": y_proba
    }

    print(f"  Accuracy:  {acc:.4f}  (NOTE: misleading alone on imbalanced data)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Train time: {train_time:.2f}s")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\n{'='*55}")
print(f"BEST MODEL (by F1-score): {best_model_name}  (F1: {best_f1:.4f})")
print(f"{'='*55}")

# -----------------------------
# 6. DETAILED REPORT FOR BEST MODEL
# -----------------------------
y_pred_best = results[best_model_name]["y_pred"]
y_proba_best = results[best_model_name]["y_proba"]

print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=["Legit", "Fraud"]))

# -----------------------------
# 7. SAVE CONFUSION MATRIX PLOT
# -----------------------------
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

# -----------------------------
# 8. SAVE ROC CURVE PLOT
# -----------------------------
plt.figure(figsize=(7, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
print("Saved roc_curve.png")

# -----------------------------
# 9. SAVE MODEL COMPARISON PLOT
# -----------------------------
comparison_df = pd.DataFrame({
    name: {"Accuracy": r["accuracy"], "Precision": r["precision"],
           "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["auc"]}
    for name, r in results.items()
}).T

comparison_df.plot(kind="bar", figsize=(10, 5))
plt.title("Model Comparison (Fraud Detection)")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("Saved model_comparison.png")

comparison_df.to_csv("model_comparison.csv")
print("Saved model_comparison.csv")

# -----------------------------
# 10. FEATURE IMPORTANCE (if Random Forest or Decision Tree won)
# -----------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    top_features.sort_values().plot(kind="barh", color="darkred")
    plt.title(f"Top 10 Important Features - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved feature_importance.png")

# -----------------------------
# 11. SAVE MODEL + SCALER
# -----------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("model_meta.pkl", "wb") as f:
    pickle.dump({
        "best_model_name": best_model_name,
        "f1": best_f1,
        "feature_columns": list(X.columns),
    }, f)

print("\nSaved best_model.pkl, scaler.pkl, model_meta.pkl")
print("\nDONE. Training complete.")
