# Ghost Worker Graph: Predicting Displacement Risk Among India's Informal Workforce Using Graph Neural Networks

Draft notes for potential submission to AAAI Social Impact Track or ACM COMPASS.

---

## Abstract (draft)

India's informal workforce of approximately 450 million workers is effectively invisible to conventional labour data systems. These workers leave no payroll records, appear in no social registries, and are underrepresented in household surveys. Yet they leave digital traces — in EPFO contribution gaps, gig platform transaction logs, and government scheme enrollment records.

This paper introduces the Ghost Worker Graph, a heterogeneous knowledge graph that connects anonymised worker clusters, digital platforms, skill types, and geographic districts sourced from public datasets including CMIE CPHS, the Periodic Labour Force Survey, and e-Shram portal enrollment records. We train a Graph Convolutional Network on the resulting graph to predict which worker clusters face the highest displacement risk from platform exits or automation shocks, and we demonstrate a simulation interface through which a state labour ministry can run counterfactual policy scenarios.

---

## 1. Introduction

The formalisation of gig work has created a new category of visible-yet-unprotected labour. Platforms like Zomato, Ola, and Urban Company employ tens of millions of workers who are classified as independent contractors, excluded from standard social protections, and subject to rapid displacement when platform economics shift.

Prior work on labour market vulnerability (citations) has primarily relied on macro-level statistics that lag real displacement events by 12-18 months. Network approaches to labour markets have been explored in formal sector contexts (citations) but rarely applied to India's informal economy.

We make three contributions:

1. A methodology for constructing a knowledge graph of informal worker relationships from heterogeneous public data sources.
2. A GCN-based vulnerability classifier that propagates risk signals through the graph structure.
3. An interactive intervention simulator evaluated with a state labour ministry stakeholder.

---

## 2. Data Sources

- CMIE CPHS (Centre for Monitoring Indian Economy Consumer Pyramids Household Survey) — monthly income records for ~170,000 households, public access subset.
- Periodic Labour Force Survey (PLFS) — annual occupation and income data.
- e-Shram portal — 300 million+ registered unorganised workers (aggregated district-level data, public).
- iMaS (Inclusive Markets and Supply Chains) — gig economy worker profiles.
- NITI Aayog reports — policy signals extracted via NLP pipeline.

---

## 3. Graph Construction

Nodes: worker clusters (anonymised by occupation and district), platforms, skill types, districts.
Edges: works_on, has_skill, located_in, displaced_from (when data available), co_located_with.

Node features:
- Worker: income volatility (coefficient of variation), platform dependency (Herfindahl index), average income, log cluster size.
- Platform: exit probability (modelled from funding events and market reports), digital integration score.
- Skill: automation risk (sourced from Oxford FHI dataset), transferability score.
- District: urbanisation rate, gig platform penetration.

---

## 4. Model

Three-layer GCN with batch normalisation and dropout. Binary classification: risk label = 1 if income drop > 40% observed in any 6-month window. Evaluated on held-out worker clusters using ROC-AUC and macro F1.

SHAP KernelExplainer used for post-hoc attribution.

---

## 5. Intervention Simulator

The simulator allows a "client" (state ministry official) to select a platform exit scenario and observe: which clusters are affected, estimated workers at risk, and minimum reskilling intervention to bring risk scores below the stabilisation threshold.

---

## 6. Limitations and Future Work

- Synthetic risk labels in the current version; real labels require longitudinal panel data.
- Graph is static; temporal GNNs (e.g. TGN) could model dynamic displacement events.
- Cluster-level analysis loses individual worker heterogeneity.
- Scope limited to five platforms and four districts in the PoC.

---

## Target Venues

- AAAI 2026/2027 — AI for Social Impact Track
- ACM COMPASS — Computing and Sustainable Societies
- CODS-COMAD — India-focused data science conference
