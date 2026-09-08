import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def train_and_evaluate(
    train_path: str = "dataset/processed/train.csv",
    test_path: str = "dataset/processed/test.csv",
    models_dir: str = "ML_Model/models",
    metrics_path: str = "ML_Model/evaluation/metrics.json"
):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed datasets missing. Run preprocess.py first.")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    print("[+] Loading processed datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "Outcome"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    models = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 8],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "estimator": XGBClassifier(random_state=42, eval_metric="logloss"),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1]
            }
        }
    }

    results = {}
    best_overall_model = None
    best_overall_score = -1.0
    best_model_name = ""

    print("\n[+] Initiating Grid Search Optimization...")
    for name, config in models.items():
        print(f"\n--- Tuning {name} ---")
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "best_params": grid.best_params_,
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(roc), 4),
            "f1_score": round(float(f1), 4)
        }

        print(f"Results for {name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {roc:.4f}")
        print(f"  F1-Score : {f1:.4f}")

        if roc > best_overall_score:
            best_overall_score = roc
            best_overall_model = best_estimator
            best_model_name = name

    print(f"\n[+] Champion Model: {best_model_name} (ROC-AUC: {best_overall_score:.4f})")

    # Save artifacts
    champion_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_overall_model, champion_path)
    print(f"[+] Saved champion model to: {champion_path}")

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Saved evaluation metrics to: {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate()