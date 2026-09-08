import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def run_preprocessing(
    raw_path: str = "dataset/raw/diabetes.csv",
    processed_dir: str = "dataset/processed",
    artifacts_dir: str = "ML_Model/models"
):
    """
    Cleans raw Pima diabetes data, imputes missing biological values,
    scales numeric features, and exports train/test splits + scaler artifact.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"[+] Loading dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    # 1. Biological zero validation: replace zeros with NaN
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    print("[+] Missing values identified per column (after zero-replacement):")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    target_col = 'Outcome'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Median Imputation (fit on train only to prevent data leakage)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )

    # 4. Feature Scaling (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=X.columns,
        index=X_test.index
    )

    # 5. Combine and save processed CSVs
    train_df = pd.concat([X_train_scaled, y_train], axis=1)
    test_df = pd.concat([X_test_scaled, y_test], axis=1)

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[+] Saved train split to {train_path} ({train_df.shape})")
    print(f"[+] Saved test split to {test_path} ({test_df.shape})")

    # 6. Save scaler & imputer artifacts for backend inference
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    imputer_path = os.path.join(artifacts_dir, "imputer.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    print(f"[+] Saved preprocessing artifacts to {artifacts_dir}/")

if __name__ == "__main__":
    run_preprocessing()