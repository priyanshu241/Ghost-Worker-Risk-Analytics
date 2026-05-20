import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score
from src.models.gnn import WorkerRiskGCN
from src.graph.builder import load_data
from src.graph.features import build_worker_features
from src.utils.config import (
    MODELS_DIR, GNN_HIDDEN_CHANNELS, GNN_DROPOUT,
    GNN_LEARNING_RATE, GNN_EPOCHS, GNN_PATIENCE
)


def build_pyg_data(workers_df, edges_df):
    """
    Converts the worker nodes and their edges into a PyTorch Geometric Data object.

    Only worker-to-worker edges (co-located or similar-skill) are used for
    message passing. Platform, skill, and district connections are encoded
    as node features instead.

    Returns:
        data        torch_geometric.data.Data
        worker_ids  list of worker ids in the same order as data.x rows
    """
    features, worker_ids = build_worker_features(workers_df, edges_df)
    id_to_idx = {wid: i for i, wid in enumerate(worker_ids)}

    # Build edge index only from edges between worker nodes
    src_list, dst_list = [], []
    for _, row in edges_df.iterrows():
        if row["source_id"] in id_to_idx and row["target_id"] in id_to_idx:
            src_list.append(id_to_idx[row["source_id"]])
            dst_list.append(id_to_idx[row["target_id"]])
            # Add reverse edge for undirected message passing
            src_list.append(id_to_idx[row["target_id"]])
            dst_list.append(id_to_idx[row["source_id"]])

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(workers_df["risk_label"].values, dtype=torch.float).unsqueeze(1)

    if len(src_list) == 0:
        # No worker-to-worker edges in the synthetic data;
        # create self-loops so GCN can still run
        n = len(worker_ids)
        edge_index = torch.stack([torch.arange(n), torch.arange(n)], dim=0)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Simple 60/20/20 split by index (small dataset)
    n = len(worker_ids)
    idx = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:max(1, int(0.6 * n))]] = True
    val_mask[idx[int(0.6 * n):int(0.8 * n)]] = True
    test_mask[idx[int(0.8 * n):]] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data, worker_ids


def train_one_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.binary_cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        preds  = model(data.x, data.edge_index)[mask].numpy().flatten()
        labels = data.y[mask].numpy().flatten()
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    auc = roc_auc_score(labels, preds)
    f1  = f1_score(labels, (preds >= 0.5).astype(int), zero_division=0)
    return auc, f1


def train(use_synthetic=True):
    workers, platforms, skills, districts, edges = load_data(use_synthetic=use_synthetic)
    data, worker_ids = build_pyg_data(workers, edges)

    model     = WorkerRiskGCN(
        in_channels=data.x.shape[1],
        hidden_channels=GNN_HIDDEN_CHANNELS,
        dropout=GNN_DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)

    best_val_auc  = -1
    patience_ctr  = 0
    best_state    = None

    for epoch in range(1, GNN_EPOCHS + 1):
        loss = train_one_epoch(model, data, optimizer)
        val_auc, val_f1 = evaluate(model, data, data.val_mask)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  loss={loss:.4f}  val_auc={val_auc:.3f}  val_f1={val_f1:.3f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= GNN_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_auc, test_f1 = evaluate(model, data, data.test_mask)
    print(f"\nTest  AUC={test_auc:.3f}  F1={test_f1:.3f}")

    # Save model and risk scores
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "worker_risk_gcn.pt")
    print(f"Model saved to {MODELS_DIR / 'worker_risk_gcn.pt'}")

    model.eval()
    with torch.no_grad():
        all_scores = model(data.x, data.edge_index).numpy().flatten()

    results = pd.DataFrame({"worker_id": worker_ids, "risk_score": all_scores})
    results = results.sort_values("risk_score", ascending=False).reset_index(drop=True)
    results.to_csv(MODELS_DIR / "risk_scores.csv", index=False)
    print(f"Risk scores saved to {MODELS_DIR / 'risk_scores.csv'}")

    return model, results


if __name__ == "__main__":
    train()
