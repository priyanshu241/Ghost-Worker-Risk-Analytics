import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_income_volatility(income_series):
    """
    Coefficient of variation (std / mean) for a sequence of monthly incomes.
    Higher value means more volatile, which is a proxy for higher displacement risk.
    """
    income_series = np.array(income_series, dtype=float)
    if len(income_series) == 0 or income_series.mean() == 0:
        return 0.0
    return float(income_series.std() / income_series.mean())


def compute_platform_dependency(worker_id, edges_df):
    """
    Herfindahl index on a worker cluster's platform connections.
    Returns 1.0 if the cluster depends on a single platform (maximum dependency),
    lower values if spread across multiple platforms.
    """
    worker_edges = edges_df[
        (edges_df["source_id"] == worker_id) &
        (edges_df["edge_type"] == "works_on")
    ]
    if len(worker_edges) == 0:
        return 1.0  # no platform at all means full dependency on informal channels

    n = len(worker_edges)
    shares = np.array([1 / n] * n)
    return float(np.sum(shares ** 2))


def build_worker_features(workers_df, edges_df):
    """
    Builds the node feature matrix for worker clusters.

    Features per worker:
        - income_volatility      (from CSV or computed from raw income series)
        - platform_dependency    (Herfindahl index)
        - avg_monthly_income     (normalised)
        - log_count              (log of estimated cluster size, normalised)

    Returns:
        features    np.ndarray of shape (num_workers, 4)
        worker_ids  list of worker ids in the same row order
    """
    records = []

    for _, row in workers_df.iterrows():
        volatility = float(row.get("income_volatility", 0.5))
        dependency = compute_platform_dependency(row["worker_id"], edges_df)
        income     = float(row.get("avg_monthly_income", 5000))
        count      = float(row.get("estimated_count", 10000))

        records.append({
            "worker_id":           row["worker_id"],
            "income_volatility":   volatility,
            "platform_dependency": dependency,
            "avg_monthly_income":  income,
            "log_count":           np.log1p(count),
        })

    df = pd.DataFrame(records)
    worker_ids = df["worker_id"].tolist()

    feature_cols = ["income_volatility", "platform_dependency",
                    "avg_monthly_income", "log_count"]

    scaler   = MinMaxScaler()
    features = scaler.fit_transform(df[feature_cols].values)

    return features, worker_ids


def build_platform_features(platforms_df):
    """
    Feature vectors for platform nodes:
        - exit_probability
        - digital_integration_score
        - log of worker_count
    """
    df = platforms_df.copy()
    df["log_worker_count"] = np.log1p(df["worker_count"].astype(float))
    feature_cols = ["exit_probability", "digital_integration_score", "log_worker_count"]
    platform_ids = df["platform_id"].tolist()

    scaler   = MinMaxScaler()
    features = scaler.fit_transform(df[feature_cols].values.astype(float))

    return features, platform_ids


def build_skill_features(skills_df):
    """
    Feature vectors for skill nodes:
        - automation_risk
        - transferability_score
    """
    feature_cols = ["automation_risk", "transferability_score"]
    skill_ids    = skills_df["skill_id"].tolist()

    scaler   = MinMaxScaler()
    features = scaler.fit_transform(skills_df[feature_cols].values.astype(float))

    return features, skill_ids


def build_district_features(districts_df):
    """
    Feature vectors for district nodes:
        - urbanisation_rate
        - gig_platform_penetration
    """
    feature_cols = ["urbanisation_rate", "gig_platform_penetration"]
    district_ids = districts_df["district_id"].tolist()

    scaler   = MinMaxScaler()
    features = scaler.fit_transform(districts_df[feature_cols].values.astype(float))

    return features, district_ids
