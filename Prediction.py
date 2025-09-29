# ====================================================
# 📌 Home Credit Default Prediction - Clean Notebook
# ====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1. Load Data
# -----------------------------------
df = pd.read_csv("/kaggle/input/rakamin-homecredit/application_train.csv")

target = "TARGET"
y = df[target]
X = df.drop(columns=[target])

# ====== EDA SEBELUM PREPROCESSING ======
print("Distribusi target:", y.value_counts(normalize=True).round(4))
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="Set2")
plt.title("Distribusi Target (0/1) - Sebelum Preprocessing")
plt.show()

print("\n📊 Info Data:")
print(df.info())

print("\n🔢 Missing Values per Kolom (Top 20):")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Contoh distribusi fitur numerik
num_sample = df.select_dtypes(exclude="object").iloc[:, :5]
num_sample.hist(bins=30, figsize=(15,6), layout=(2,3))
plt.suptitle("Distribusi Fitur Numerik (Contoh 5) - Sebelum Preprocessing")
plt.show()

# Korelasi numerik dg Target
corr = df[num_sample.columns.tolist() + [target]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelasi Fitur Numerik dg Target - Sebelum Preprocessing")
plt.show()

# Contoh fitur kategorikal
cat_sample = "NAME_CONTRACT_TYPE"
if cat_sample in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=cat_sample, data=df, palette="Set3", hue=y)
    plt.title(f"Distribusi {cat_sample} berdasarkan Target - Sebelum Preprocessing")
    plt.xticks(rotation=30)
    plt.show()

# -----------------------------------
# 2. Split Data
# -----------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# -----------------------------------
# 3. Preprocessing
# -----------------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)

# ====== EDA SESUDAH PREPROCESSING ======
X_train_all = preprocessor.fit_transform(X_train)
X_val_all   = preprocessor.transform(X_val)
X_test_all  = preprocessor.transform(X_test)

print("\n📊 Shape Sesudah Preprocessing:")
print("Train:", X_train_all.shape, "Val:", X_val_all.shape, "Test:", X_test_all.shape)

# Konversi ke DataFrame
feature_names = preprocessor.get_feature_names_out()
X_train_all_df = pd.DataFrame(X_train_all, columns=feature_names, index=X_train.index)

# Distribusi beberapa fitur numerik
X_train_all_df.iloc[:, :5].hist(bins=30, figsize=(15,6), layout=(2,3))
plt.suptitle("Distribusi Fitur Numerik (Contoh 5) - Sesudah Preprocessing")
plt.show()

print("\n📊 Statistik Ringkas Sesudah Preprocessing (5 fitur pertama):")
print(X_train_all_df.iloc[:, :5].describe())

# Korelasi antar fitur (contoh 10)
corr_post = X_train_all_df.iloc[:, :10].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_post, annot=False, cmap="viridis")
plt.title("Korelasi Antar Fitur (Contoh 10) - Sesudah Preprocessing")
plt.show()

# -----------------------------------
# 4. Models
# -----------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
}

roc_results = {}
pr_results = {}

# -----------------------------------
# 5. Train + Evaluasi per Model
# -----------------------------------
for name, model in models.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    # Probabilitas prediksi
    y_val_proba = pipe.predict_proba(X_val)[:, 1]
    y_val_pred = pipe.predict(X_val)

    # ROC
    auc = roc_auc_score(y_val, y_val_proba)
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)
    roc_results[name] = (fpr, tpr, auc)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
    ap = average_precision_score(y_val, y_val_proba)
    pr_results[name] = (precision, recall, ap)

    # Report
    print("="*50)
    print(f"{name}")
    print(classification_report(y_val, y_val_pred, digits=4))
    print(f"AUC: {auc:.4f} | Average Precision: {ap:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred, labels=[0,1])
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=3, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_mean, "o-", label="Train AUC")
    plt.plot(train_sizes, val_mean, "o-", label="Validation AUC")
    plt.title(f"Learning Curve - {name}")
    plt.xlabel("Training samples")
    plt.ylabel("AUC")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# -----------------------------------
# 6. ROC Curve Comparison
# -----------------------------------
plt.figure(figsize=(7,7))
for name, (fpr, tpr, auc) in roc_results.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
plt.plot([0,1],[0,1],'--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.show()

# -----------------------------------
# 7. Precision-Recall Curve Comparison
# -----------------------------------
plt.figure(figsize=(7,7))
for name, (precision, recall, ap) in pr_results.items():
    plt.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend(loc="lower left")
plt.show()

# -----------------------------------
# 8. Feature Importance (Random Forest)
# -----------------------------------
rf_pipe = Pipeline(steps=[("pre", preprocessor),
                          ("model", RandomForestClassifier(
                              n_estimators=200, random_state=42, 
                              class_weight="balanced", n_jobs=-1))])
rf_pipe.fit(X_train, y_train)

feature_names = rf_pipe.named_steps["pre"].get_feature_names_out()
importances = rf_pipe.named_steps["model"].feature_importances_

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(20)

print("\nTop 20 Feature Importances (Random Forest):")
print(feat_imp)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="viridis")
plt.title("Top 20 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# -----------------------------------
# 9. Submission ke Kaggle
# -----------------------------------
test_df = pd.read_csv("/kaggle/input/rakamin-homecredit/application_test.csv")
submission = pd.read_csv("/kaggle/input/rakamin-homecredit/sample_submission.csv")

best_model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
pipe_best = Pipeline(steps=[("pre", preprocessor), ("model", best_model)])
pipe_best.fit(X_train, y_train)

test_pred_proba = pipe_best.predict_proba(test_df)[:, 1]
submission["TARGET"] = test_pred_proba
submission.to_csv("submission.csv", index=False)

print("✅ File submission.csv berhasil dibuat!")
print(submission.head())
