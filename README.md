# Ghost Worker Graph

A knowledge graph and Graph Neural Network system for predicting displacement risk among India's informal workforce. Designed as a deployable advisory tool for state labour ministries and a portfolio project for data analytics and AI roles.

---

## What this project does

India has approximately 450 million informal workers who appear in no payroll records, no clean dataset, no social registry. They leave traces — in CMIE household surveys, gig platform transaction logs, PLFS occupation records, and government scheme enrollment data.

This project connects those traces into a knowledge graph, trains a Graph Convolutional Network on top of it, and builds an intervention simulator that tells a government client: if this platform exits, here are exactly which worker clusters collapse, and what reskilling threshold stabilises them.

---

## Project structure

```
ghost-worker-graph/
│
├── README.md
│
├── requirements.txt
│
├── data/
│   ├── raw/                        real CMIE CPHS data goes here
│   ├── processed/                  cleaned node and edge CSVs
│   └── synthetic/                  generated data for PoC (used by default)
│       ├── workers.csv
│       ├── platforms.csv
│       ├── skills.csv
│       ├── districts.csv
│       └── edges.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                exploratory data analysis
│   ├── 02_graph_construction.ipynb build knowledge graph, push to Neo4j
│   ├── 03_gnn_training.ipynb       train GCN, visualise embeddings
│   ├── 04_nlp_pipeline.ipynb       NER on policy docs, headwind index
│   └── 05_time_series.ipynb        income volatility, Prophet forecasting
│
├── src/
│   ├── graph/
│   │   ├── builder.py              load CSVs, build NetworkX graph, push to Neo4j
│   │   └── features.py             node feature engineering
│   ├── models/
│   │   ├── gnn.py                  WorkerRiskGCN (PyTorch Geometric)
│   │   ├── train.py                training loop, early stopping, evaluation
│   │   └── explain.py              SHAP feature attribution
│   ├── nlp/
│   │   ├── scraper.py              download policy PDFs from government sites
│   │   ├── ner.py                  spaCy NER pipeline
│   │   └── headwind.py             monthly policy headwind index
│   └── utils/
│       ├── config.py               paths, hyperparameters, constants
│       └── db.py                   Neo4j connection wrapper
│
├── dashboard/
│   └── app.py                      Streamlit application (the PoC deliverable)
│
├── models_saved/                   trained GCN weights and risk score CSVs
│
├── logs/                           saved plots from notebooks
│
└── paper/
    └── draft.md                    notes toward a conference submission
```

---

## Getting started

### 1. Install dependencies

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For PyTorch Geometric, follow the installation guide at https://pytorch-geometric.readthedocs.io. The correct command depends on your OS and CUDA version.

### 2. Run the dashboard (no training required)

The dashboard uses synthetic data by default and works immediately after installation.

```
cd ghost-worker-graph
streamlit run dashboard/app.py
```

Open the URL that Streamlit prints (usually http://localhost:8501).

### 3. Run the notebooks in order

```
cd notebooks
jupyter notebook
```

Run them in order: 01 through 05. Each notebook is self-contained and imports from `src/`.

### 4. Train the GNN

```
python -m src.models.train
```

This saves model weights to `models_saved/worker_risk_gcn.pt` and risk scores to `models_saved/risk_scores.csv`. The dashboard automatically uses real scores if that file exists.

### 5. Connect real data

Replace `data/synthetic/` CSVs with real CMIE CPHS data:

- Register for free access at https://cmie.com
- Export household income records
- Clean and reshape to match the column schema in the synthetic CSVs
- Set `use_synthetic=False` in any call to `load_data()`

---

## Neo4j setup (optional)

Neo4j is used for graph querying beyond what NetworkX supports. It is not required to run the dashboard or notebooks.

1. Download Neo4j Community Edition from https://neo4j.com/download/
2. Start the server and set a password
3. Update `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in `src/utils/config.py`
4. Run `push_to_neo4j()` from `src/graph/builder.py` to populate the graph

---

## Resume bullet

Ghost Worker Graph — Knowledge graph and GNN for predicting displacement risk among India's 450M informal workers. Built graph construction pipeline (NetworkX, Neo4j), trained GCN vulnerability classifier (PyTorch Geometric, SHAP explainability), NLP policy signal extractor (spaCy, BERTopic), and interactive intervention simulator (Streamlit). Designed as a deployable advisory tool for state labour ministries.

---

## Data sources

| Source | What it provides | Access |
|---|---|---|
| CMIE CPHS | Monthly household income records | Free registration at cmie.com |
| PLFS (MoSPI) | Annual occupation and earnings data | Public download |
| e-Shram portal | 300M+ registered unorganised workers (district aggregates) | Public |
| iMaS | Gig economy worker profiles | Application required |
| NITI Aayog reports | Policy signals | Free PDF download |

---

## Publication path

The project has a plausible path to a conference paper at:
- AAAI Social Impact Track
- ACM COMPASS (Computing and Sustainable Societies)
- CODS-COMAD (India)

See `paper/draft.md` for a working abstract and section outline.
