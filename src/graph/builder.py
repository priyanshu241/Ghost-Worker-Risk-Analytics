import pandas as pd
import networkx as nx
from src.utils.config import DATA_PROCESSED, DATA_SYNTHETIC


def load_data(use_synthetic=True):
    """
    Loads all node and edge dataframes from CSV files.
    Set use_synthetic=False once you have real CMIE CPHS data in data/processed/.
    """
    data_dir = DATA_SYNTHETIC if use_synthetic else DATA_PROCESSED

    workers   = pd.read_csv(data_dir / "workers.csv")
    platforms = pd.read_csv(data_dir / "platforms.csv")
    skills    = pd.read_csv(data_dir / "skills.csv")
    districts = pd.read_csv(data_dir / "districts.csv")
    edges     = pd.read_csv(data_dir / "edges.csv")

    return workers, platforms, skills, districts, edges


def build_networkx_graph(workers, platforms, skills, districts, edges):
    """
    Builds a directed graph from the loaded dataframes.
    Node attributes come directly from CSV columns.
    """
    G = nx.DiGraph()

    for _, row in workers.iterrows():
        G.add_node(row["worker_id"], node_type="worker", **row.to_dict())

    for _, row in platforms.iterrows():
        G.add_node(row["platform_id"], node_type="platform", **row.to_dict())

    for _, row in skills.iterrows():
        G.add_node(row["skill_id"], node_type="skill", **row.to_dict())

    for _, row in districts.iterrows():
        G.add_node(row["district_id"], node_type="district", **row.to_dict())

    for _, row in edges.iterrows():
        G.add_edge(
            row["source_id"],
            row["target_id"],
            edge_type=row["edge_type"],
            weight=float(row.get("weight", 1.0))
        )

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def push_to_neo4j(workers, platforms, skills, districts, edges):
    """
    Pushes the full graph to a running Neo4j instance.
    Update credentials in src/utils/config.py before running.
    """
    from src.utils.db import GraphDB
    db = GraphDB()

    print("Clearing existing graph...")
    db.clear_graph()

    print("Writing worker nodes...")
    for _, row in workers.iterrows():
        db.create_worker_node(
            worker_id=row["worker_id"],
            label=row["label"],
            count=int(row["estimated_count"]),
            risk=float(row["risk_label"]),
            district_id=row["district_id"]
        )

    print("Writing platform nodes...")
    for _, row in platforms.iterrows():
        db.create_platform_node(
            platform_id=row["platform_id"],
            name=row["name"],
            exit_risk=float(row["exit_probability"])
        )

    print("Writing skill nodes...")
    for _, row in skills.iterrows():
        db.create_skill_node(
            skill_id=row["skill_id"],
            name=row["name"],
            automation_risk=float(row["automation_risk"])
        )

    print("Writing district nodes...")
    for _, row in districts.iterrows():
        db.create_district_node(
            district_id=row["district_id"],
            name=row["name"],
            state=row["state"]
        )

    print("Writing edges...")
    for _, row in edges.iterrows():
        db.create_edge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=row["edge_type"].upper(),
            weight=float(row.get("weight", 1.0))
        )

    db.close()
    print("Graph pushed to Neo4j successfully.")


if __name__ == "__main__":
    workers, platforms, skills, districts, edges = load_data(use_synthetic=True)
    G = build_networkx_graph(workers, platforms, skills, districts, edges)
