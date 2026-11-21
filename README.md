# KG Fact-Checker (`kg-fact-checker`)

Reproducible Knowledge Graph Fact-Checking pipeline for **FactBench** evaluated on **GERBIL** (`http://gerbil-kbc.aksw.org/gerbil/config`), targeting **ROC AUC > 0.60** before **January 15, 2027**.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt