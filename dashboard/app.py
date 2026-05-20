"""
app.py

Streamlit dashboard for the Ghost Worker Graph project.

Run with:
    streamlit run dashboard/app.py

Make sure you are in the project root (ghost-worker-graph/) when running.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from src.graph.builder import load_data, build_networkx_graph
from src.nlp.headwind import compute_headwind_index
from src.utils.config import MODELS_DIR, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM, GRAPH_STABILISATION_THRESHOLD

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ghost Worker Graph",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_all():
    workers, platforms, skills, districts, edges = load_data(use_synthetic=True)
    G = build_networkx_graph(workers, platforms, skills, districts, edges)
    headwind = compute_headwind_index(use_synthetic=True)

    risk_path = MODELS_DIR / "risk_scores.csv"
    if risk_path.exists():
        risk_scores = pd.read_csv(risk_path)
        workers = workers.merge(risk_scores, on="worker_id", how="left")
        workers["risk_score"] = workers["risk_score"].fillna(workers["risk_label"].astype(float))
    else:
        # Fall back to synthetic scores derived from income_volatility
        workers["risk_score"] = workers["income_volatility"].values

    return workers, platforms, skills, districts, edges, G, headwind

workers, platforms, skills, districts, edges, G, headwind = load_all()

# ── Risk label helper ────────────────────────────────────────────────────────

def risk_label(score):
    if score >= RISK_THRESHOLD_HIGH:
        return "High"
    elif score >= RISK_THRESHOLD_MEDIUM:
        return "Medium"
    return "Low"

def risk_color(score):
    if score >= RISK_THRESHOLD_HIGH:
        return "#e74c3c"
    elif score >= RISK_THRESHOLD_MEDIUM:
        return "#f39c12"
    return "#27ae60"

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("## Ghost Worker Graph")
st.markdown(
    "Knowledge graph and GNN displacement risk intelligence for India's informal workforce. "
    "Designed as a deployable advisory tool for state labour ministries."
)

# ── KPI row ──────────────────────────────────────────────────────────────────

high_risk = workers[workers["risk_score"] >= RISK_THRESHOLD_HIGH]
total_workers_est = workers["estimated_count"].sum()
latest_headwind = headwind["headwind_index"].iloc[-1]
prev_headwind   = headwind["headwind_index"].iloc[-2]
delta_headwind  = latest_headwind - prev_headwind

col1, col2, col3, col4 = st.columns(4)
col1.metric("Worker clusters mapped", len(workers), f"~{total_workers_est:,} estimated workers")
col2.metric("High-risk clusters", len(high_risk), f"GNN risk score >= {RISK_THRESHOLD_HIGH}")
col3.metric("Platform nodes", len(platforms), f"{len(platforms[platforms['exit_probability'] >= 0.6])} flagged high exit-risk")
col4.metric("Policy headwind (latest)", f"{latest_headwind:.2f}", f"{delta_headwind:+.2f} vs prior month")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Knowledge graph", "Risk scores", "Intervention simulator", "Policy signals"])

# ── Tab 1: Knowledge graph ───────────────────────────────────────────────────

with tab1:
    st.markdown("**Force-directed knowledge graph** — workers (circles), platforms (squares), skills (diamonds), districts (triangles). Node size encodes population or worker count. Color encodes risk level for worker nodes.")

    pos = nx.spring_layout(G, seed=42, k=2.5)

    node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = data.get("node_type", "unknown")

        if ntype == "worker":
            risk = float(data.get("risk_score", data.get("risk_label", 0.5)))
            count = int(data.get("estimated_count", 10000))
            label_str = data.get("label", node)
            node_text.append(f"{label_str}<br>Risk: {risk:.0%}<br>Pop: {count:,}")
            node_color.append(risk_color(risk))
            node_size.append(max(12, min(40, count / 4000)))
            node_symbol.append("circle")

        elif ntype == "platform":
            exit_p = float(data.get("exit_probability", 0.3))
            node_text.append(f"{data.get('name', node)}<br>Exit risk: {exit_p:.0%}")
            node_color.append("#e74c3c" if exit_p >= 0.6 else "#3498db")
            node_size.append(18)
            node_symbol.append("square")

        elif ntype == "skill":
            auto = float(data.get("automation_risk", 0.5))
            node_text.append(f"{data.get('name', node)}<br>Automation risk: {auto:.0%}")
            node_color.append("#9b59b6")
            node_size.append(14)
            node_symbol.append("diamond")

        else:  # district
            node_text.append(f"{data.get('name', node)}, {data.get('state', '')}")
            node_color.append("#1abc9c")
            node_size.append(14)
            node_symbol.append("triangle-up")

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig_graph = go.Figure()
    fig_graph.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.7, color="#cccccc"),
        hoverinfo="none", showlegend=False
    ))
    fig_graph.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color, symbol=node_symbol,
                    line=dict(width=1, color="#ffffff")),
        text=[t.split("<br>")[0] for t in node_text],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False
    ))
    fig_graph.update_layout(
        height=520, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_graph, use_container_width=True)

    st.caption(
        "Worker circles: red = high risk, orange = medium, green = low. "
        "Blue squares = platforms (red if exit probability >= 60%). "
        "Purple diamonds = skill clusters. Teal triangles = districts."
    )

# ── Tab 2: Risk scores ───────────────────────────────────────────────────────

with tab2:
    st.markdown("**GNN displacement vulnerability scores** — trained on income volatility, platform dependency, and cluster size. "
                "Node size encodes population; scores from 0 to 1.")

    sorted_workers = workers.sort_values("risk_score", ascending=False).reset_index(drop=True)
    sorted_workers["risk_level"] = sorted_workers["risk_score"].apply(risk_label)
    sorted_workers["color"]      = sorted_workers["risk_score"].apply(risk_color)

    fig_bar = go.Figure(go.Bar(
        x=sorted_workers["risk_score"],
        y=sorted_workers["label"],
        orientation="h",
        marker_color=sorted_workers["color"],
        text=sorted_workers["risk_score"].map(lambda s: f"{s:.0%}"),
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=380,
        xaxis=dict(title="Risk score", tickformat=".0%", range=[0, 1.1]),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=0, r=60, t=20, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        sorted_workers[["label", "risk_score", "risk_level", "estimated_count",
                         "avg_monthly_income", "income_volatility"]].rename(columns={
            "label": "Worker cluster",
            "risk_score": "GNN risk score",
            "risk_level": "Risk level",
            "estimated_count": "Estimated workers",
            "avg_monthly_income": "Avg monthly income (Rs)",
            "income_volatility": "Income volatility",
        }),
        use_container_width=True, hide_index=True
    )

# ── Tab 3: Intervention simulator ────────────────────────────────────────────

with tab3:
    st.markdown(
        "**What-if simulator** — select a platform exit scenario and see which worker "
        "clusters are affected, how many workers are at risk, and what minimum reskilling "
        "threshold stabilises the graph."
    )

    platform_options = platforms["name"].tolist()
    selected_platform = st.selectbox("Platform exit scenario", platform_options)
    prow = platforms[platforms["name"] == selected_platform].iloc[0]

    platform_id = prow["platform_id"]
    affected_edges = edges[(edges["source_id"].isin(workers["worker_id"])) &
                           (edges["target_id"] == platform_id)]
    affected_worker_ids = affected_edges["source_id"].tolist()
    affected = workers[workers["worker_id"].isin(affected_worker_ids)].copy()

    st.markdown(f"#### If {selected_platform} exits")
    c1, c2, c3 = st.columns(3)
    c1.metric("Affected clusters", len(affected))
    c2.metric("Workers at risk", f"{int(affected['estimated_count'].sum()):,}")
    c3.metric("Platform exit probability", f"{prow['exit_probability']:.0%}")

    if len(affected) == 0:
        st.info("No worker clusters directly connected to this platform in the current graph.")
    else:
        affected["risk_level"] = affected["risk_score"].apply(risk_label)
        affected["reskilling_needed"] = (
            (affected["risk_score"] - GRAPH_STABILISATION_THRESHOLD)
            .clip(lower=0)
            .map(lambda x: f"{x:.0%} of cluster workforce")
        )

        st.dataframe(
            affected[["label", "estimated_count", "risk_score", "risk_level", "reskilling_needed"]].rename(columns={
                "label": "Worker cluster",
                "estimated_count": "Workers",
                "risk_score": "Current risk score",
                "risk_level": "Risk level",
                "reskilling_needed": "Reskilling intervention needed",
            }),
            use_container_width=True, hide_index=True
        )

        st.markdown(
            f"**Advisory note:** The stabilisation threshold is set at {GRAPH_STABILISATION_THRESHOLD:.0%}. "
            "Clusters with a risk score above this threshold require active reskilling to remain stable after "
            f"the exit of {selected_platform}. "
            "Connect to NITI Aayog's National Skill Development Corporation (NSDC) programmes for targeted intervention."
        )

# ── Tab 4: Policy signals ─────────────────────────────────────────────────────

with tab4:
    st.markdown(
        "**Policy headwind index** — computed monthly from NER-extracted signals in "
        "NITI Aayog reports, Ministry of Labour press releases, and e-Shram portal "
        "announcements. Higher values indicate more adverse regulatory conditions for "
        "informal workers."
    )

    fig_hw = px.line(
        headwind, x="month", y="headwind_index",
        labels={"month": "Month", "headwind_index": "Policy headwind index"},
        markers=True,
    )
    fig_hw.add_hline(y=0.6, line_dash="dot", line_color="#e74c3c",
                     annotation_text="High headwind threshold (0.60)")
    fig_hw.update_traces(line_color="#2c3e50", marker_color="#e74c3c")
    fig_hw.update_layout(
        height=380,
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_hw, use_container_width=True)

    st.caption(
        "Index is synthetic in this PoC, generated from a model calibrated against CMIE CPHS "
        "income volatility patterns. Replace with real NER pipeline output once policy PDFs are downloaded."
    )

    # Show recent signal table
    recent = headwind.tail(6).copy()
    recent["month"] = recent["month"].dt.strftime("%b %Y")
    recent["headwind_index"] = recent["headwind_index"].map("{:.2f}".format)
    st.dataframe(recent.rename(columns={"month": "Month", "headwind_index": "Headwind index"}),
                 use_container_width=True, hide_index=True)
