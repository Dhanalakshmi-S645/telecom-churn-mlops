import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import get_data_splits


MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1
    ),
}


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }


def train_all():
    X_train, X_test, y_train, y_test = get_data_splits()

    mlflow.set_experiment("telecom-churn-prediction")

    results = []
    best_auc = 0
    best_model = None
    best_model_name = ""

    print("\n🚀 Training 6 models with MLflow tracking...\n")
    print(f"{'Model':<22} {'AUC-ROC':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Accuracy'}")
    print("-" * 75)

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            results.append({"model": name, **metrics})
            print(f"{name:<22} {metrics['auc_roc']:<10} {metrics['f1_score']:<10} "
                  f"{metrics['precision']:<12} {metrics['recall']:<10} {metrics['accuracy']}")

            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_model = model
                best_model_name = name

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(best_model_name, "models/best_model_name.pkl")

    print(f"\n🏆 Best model: {best_model_name} (AUC-ROC: {best_auc})")
    print("✅ Best model saved to models/best_model.pkl")
    print("✅ MLflow experiments saved — run: mlflow ui")

    # Save results table
    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("✅ Model comparison saved to reports/model_comparison.csv")

    return results_df


if __name__ == "__main__":
    train_all()