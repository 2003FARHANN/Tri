import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def generate_shap_explanations(
    test_path: str = "dataset/processed/test.csv",
    model_path: str = "ML_Model/models/best_model.pkl",
    output_dir: str = "ML_Model/explainable_ai/plots"
):
    """
    Generates SHAP TreeExplainer values for global and local interpretability,
    saving summary plots to visualize diabetes risk drivers.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    os.makedirs(output_dir, exist_ok=True)

    print("[+] Loading champion model and test dataset...")
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["Outcome"])

    print("[+] Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # 1. Global summary plot (Bar plot for feature ranking)
    print("[+] Generating global feature importance plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    bar_plot_path = os.path.join(output_dir, "shap_feature_importance.png")
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved feature importance plot to: {bar_plot_path}")

    # 2. Global beeswarm plot (Directional impact)
    print("[+] Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"[+] Saved summary beeswarm plot to: {beeswarm_path}")

    # 3. Export numeric values for backend API integration
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False)

    importance_json_path = os.path.join(output_dir, "feature_importance.json")
    importance_df.to_json(importance_json_path, orient="records", indent=4)
    print(f"[+] Saved feature importance JSON to: {importance_json_path}")


if __name__ == "__main__":
    generate_shap_explanations()
