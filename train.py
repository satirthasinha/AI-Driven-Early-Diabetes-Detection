import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# =================== Ensure Directories Exist ===================
os.makedirs("backend/plots", exist_ok=True)
os.makedirs("backend/models", exist_ok=True)

# =================== Load Dataset ===================
df = pd.read_csv("data/diabetes.csv")

# =================== Feature Engineering ===================
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = X.columns.tolist()

# =================== Train-Test Split ===================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =================== Feature Scaling ===================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =================== XGBoost with Grid Search (No Early Stopping) ===================

# StratifiedKFold for balanced splits
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Custom wrapper to use in GridSearchCV without early stopping
class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, reg_alpha=0, reg_lambda=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model = None

    def fit(self, X, y):
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        # For simplicity, fit on all training data without early stopping (or do CV split here)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

grid = GridSearchCV(
    XGBWrapper(),
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_.model
y_pred = best_model.predict(X_test_scaled)

# =================== Accuracy ===================
train_acc = accuracy_score(y_train, best_model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# =================== Confusion Matrix ===================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('backend/plots/confusion_matrix.png')
plt.close()

# =================== ROC Curve ===================
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('backend/plots/roc_curve.png')
plt.close()

# =================== Train vs Test Accuracy Plot ===================
plt.figure()
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_acc, test_acc], color=['green', 'blue'])
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.ylim(0, 1)
plt.savefig("backend/plots/train_vs_test_accuracy.png", dpi=300)
plt.close()

# =================== Feature Distributions ===================
for column in X.columns:
    plt.figure()
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.savefig(f'backend/plots/distribution_{column}.png')
    plt.close()

# =================== Correlation Heatmap ===================
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('backend/plots/correlation_heatmap.png')
plt.close()

# =================== GridSearch Heatmap (learning_rate = 0.01) ===================
# Reshape CV scores: max_depth (3), n_estimators (3), learning_rate (3)
results_df = pd.DataFrame(grid.cv_results_)

filtered_results = results_df[
    (results_df['param_learning_rate'] == 0.01) &
    (results_df['param_reg_alpha'] == 0) &
    (results_df['param_reg_lambda'] == 1)
]

heatmap_data = filtered_results.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".4f")
plt.title('Grid Search Accuracy\n(learning_rate=0.01, reg_alpha=0, reg_lambda=1)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.savefig('backend/plots/gridsearch_heatmap.png')
plt.close()

# =================== SHAP Explainability ===================
explainer = shap.Explainer(best_model, feature_names=feature_names)
shap_values = explainer(X_train_scaled)

# SHAP summary plot
shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot")
plt.savefig('backend/plots/shap_summary.png', bbox_inches='tight')
plt.close()

# SHAP bar plot
shap.plots.bar(
    shap.Explanation(
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=X_train_scaled,
        feature_names=feature_names
    ),
    max_display=10, show=False
)
plt.title("Top SHAP Features")
plt.savefig('backend/plots/shap_bar.png', bbox_inches='tight')
plt.close()

# SHAP dependence plot for 'Glucose'
shap.dependence_plot(
    'Glucose',
    shap_values.values,
    X_train,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Dependence Plot - Glucose")
plt.savefig('backend/plots/shap_dependence_glucose.png', bbox_inches='tight', dpi=300)
plt.close()

# =================== Save Model, Scaler, and Explainer ===================
joblib.dump(best_model, 'backend/models/xgb_model.pkl')
joblib.dump(scaler, 'backend/models/scaler.pkl')
joblib.dump(explainer, 'backend/models/shap_explainer.pkl')

print("✅ Training complete. Model, scaler, SHAP explainer and all plots saved in 'backend/'")
