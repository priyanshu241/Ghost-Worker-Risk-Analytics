"""
explain.py

SHAP-based feature attribution for the WorkerRiskGCN model.

Because GNNs are not natively supported by SHAP's TreeExplainer or
LinearExplainer, we use KernelExplainer which treats the model as a black box.
This is slower but model-agnostic and correct.

Usage:
    from src.models.explain import explain_worker_risks
    shap_df = explain_worker_risks()
    print(shap_df)
"""

import torch
import numpy as np
import pandas as pd
import shap
from src.models.gnn import WorkerRiskGCN
from src.graph.builder import load_data
from src.graph.features import build_worker_features
from src.models.train import build_pyg_data
from src.utils.config import MODELS_DIR, GNN_HIDDEN_CHANNELS, GNN_DROPOUT

FEATURE_NAMES = [
    "income_volatility",
    "platform_dependency",
    "avg_monthly_income",
    "log_count",
]


def load_trained_model(in_channels):
    """Loads the saved GCN weights from disk."""
    model = WorkerRiskGCN(
        in_channels=in_channels,
        hidden_channels=GNN_HIDDEN_CHANNELS,
        dropout=GNN_DROPOUT
    )
    state = torch.load(MODELS_DIR / "worker_risk_gcn.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def make_predict_fn(model, edge_index):
    """
    Returns a function f(X) -> risk_scores that SHAP can call.
    X is a 2D numpy array where each row is a node's feature vector.
    """
    def predict(X):
        x_tensor = torch.tensor(X, dtype=torch.float)
        with torch.no_grad():
            scores = model(x_tensor, edge_index).numpy().flatten()
        return scores
    return predict


def explain_worker_risks(use_synthetic=True, num_background=5):
    """
    Runs SHAP KernelExplainer on all worker nodes and returns a DataFrame
    with one SHAP value per feature per worker.

    Args:
        use_synthetic    use synthetic data (True) or real processed data (False)
        num_background   number of background samples for KernelExplainer
                         (keep small; synthetic dataset has only 8 workers)

    Returns:
        shap_df  pd.DataFrame with columns = FEATURE_NAMES + ['worker_id', 'risk_score']
    """
    workers, _, _, _, edges = load_data(use_synthetic=use_synthetic)
    data, worker_ids = build_pyg_data(workers, edges)

    model = load_trained_model(in_channels=data.x.shape[1])

    predict_fn = make_predict_fn(model, data.edge_index)

    X = data.x.numpy()
    background = X[:min(num_background, len(X))]

    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X, nsamples=50)

    shap_df = pd.DataFrame(shap_values, columns=FEATURE_NAMES)
    shap_df["worker_id"]  = worker_ids

    with torch.no_grad():
        shap_df["risk_score"] = model(data.x, data.edge_index).numpy().flatten()

    shap_df = shap_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return shap_df


def print_attribution_table(shap_df):
    """Prints a readable attribution table to stdout."""
    print(f"\n{'Worker':<30} {'Risk':>6}  " + "  ".join(f"{n:>20}" for n in FEATURE_NAMES))
    print("-" * 110)
    for _, row in shap_df.iterrows():
        vals = "  ".join(f"{row[n]:>+20.4f}" for n in FEATURE_NAMES)
        print(f"{row['worker_id']:<30} {row['risk_score']:>6.2f}  {vals}")


if __name__ == "__main__":
    df = explain_worker_risks()
    print_attribution_table(df)
