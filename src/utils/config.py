from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
MODELS_DIR     = ROOT / "models_saved"
LOGS_DIR       = ROOT / "logs"

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password_here"

GNN_HIDDEN_CHANNELS = 64
GNN_NUM_LAYERS      = 3
GNN_DROPOUT         = 0.3
GNN_LEARNING_RATE   = 0.01
GNN_EPOCHS          = 200
GNN_PATIENCE        = 20

RISK_THRESHOLD_HIGH   = 0.70
RISK_THRESHOLD_MEDIUM = 0.45

GRAPH_STABILISATION_THRESHOLD = 0.35

SPACY_MODEL     = "en_core_web_sm"
POLICY_ENTITIES = ["ORG", "GPE", "NORP", "LAW", "MONEY", "PERCENT"]
