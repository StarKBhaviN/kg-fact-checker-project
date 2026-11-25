# KG Fact-Checker (`kg-fact-checker`)

A reproducible Knowledge Graph Fact-Checking pipeline for **FactBench** (`FoKG26`), evaluated on **GERBIL** (`http://gerbil-kbc.aksw.org/gerbil/config`), targeting **ROC AUC > 0.60**.

---

## Results & Progress

Through iterative improvements in DBpedia SPARQL querying, cache reliability, feature engineering, and model tuning, the 5-fold Stratified Cross-Validation **ROC AUC** progressed as follows:

| Iteration | ROC AUC (5-Fold CV) | Key Milestone |
| :--- | :---: | :--- |
| **Baseline** | **~0.50** | Initial pipeline with basic graph features and offline/timeout-affected signals |
| **Iteration 2** | **~0.70** | Improved entity/type signals and initial SPARQL cache integration |
| **Iteration 3** | **~0.87** | Added multi-threaded querying, `dbo:`/`dbp:` matching, and tuned gradient boosting |
| **Final Model** | **0.9527** | Fixed timeout cache poisoning, added conflict detection (`has_other_dbo`/`dbp`), fuzzy string similarity, and Wikipedia link signals |

### Final Evaluation Metrics (5-Fold CV on `data/FoKG26-train.nt`, 765 statements)
- **ROC AUC**: **0.9527** (target: > 0.60)
- **Accuracy**: **87.58%** (670 / 765 statements correctly classified at threshold 0.5)
- **Log Loss**: **0.2608**

---

## Setup

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Execution Guide

### Step 1: Train the Model
Train the model with 5-fold stratified cross-validation on `data/FoKG26-train.nt` and save the trained artifact to `models/fact_checker.joblib`:

```bash
python -m src.train \
  --input data/FoKG26-train.nt \
  --model-out models/fact_checker.joblib \
  --cache-dir cache \
  --cv-folds 5 \
  --workers 8
```
- **Expected output**: Prints 5-fold OOF `ROC AUC`, `Acc`, and `LogLoss`, and saves `models/fact_checker.joblib`.
- **Note**: When `cache/sparql_cache.sqlite` is populated, this step completes in a few seconds. Add `--offline` to skip live DBpedia queries completely.

### Step 2: Generate Test Predictions (for GERBIL `FoKG 2026 Test`)
Run inference on `data/FoKG26-test.nt` and output GERBIL-compatible Turtle triples:

```bash
python -m src.predict \
  --input data/FoKG26-test.nt \
  --model-in models/fact_checker.joblib \
  --output outputs/result.ttl \
  --cache-dir cache \
  --workers 8
```
- **Output file**: `outputs/result.ttl` (327 validated triples)
- **GERBIL dataset**: Select **FoKG 2026 Test** and upload `outputs/result.ttl`.

### Step 3: (Optional) Generate Train Predictions (for GERBIL `FoKG 2026 Train`)
```bash
python -m src.predict \
  --input data/FoKG26-train.nt \
  --model-in models/fact_checker.joblib \
  --output outputs/result_train.ttl \
  --cache-dir cache \
  --workers 8
```
- **Output file**: `outputs/result_train.ttl` (765 validated triples)

### Step 4: Verify the Generated Output
Check the first 10 lines of `outputs/result.ttl`:

```bash
head -n 10 outputs/result.ttl
```
*(On Windows PowerShell: `Get-Content outputs/result.ttl -TotalCount 10`)*

You should see valid Turtle triples with `xsd:double` truth values in `[0.0, 1.0]`:
```turtle
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://swc2017.aksw.org/task2/dataset/stmt_60> <http://swc2017.aksw.org/hasTruthValue> "0.9521"^^<http://www.w3.org/2001/XMLSchema#double> .
<http://swc2017.aksw.org/task2/dataset/stmt_204> <http://swc2017.aksw.org/hasTruthValue> "0.0435"^^<http://www.w3.org/2001/XMLSchema#double> .
```

### Step 5: Run Unit Tests
Verify RDF parsing, Turtle validation, SQLite caching, and feature matrix construction:

```bash
python -m pytest
```

---

## How the Project Works (Pipeline Overview)

1. **RDF Input & Output (`src/io_rdf.py`)**:
   - Parses reified `rdf:Statement` triples (`subject`, `predicate`, `object`, and optional `swc:hasTruthValue`) from `.nt` files.
   - Writes predictions in valid Turtle (`.ttl`) format and validates URI coverage and `[0.0, 1.0]` score bounds.
2. **Cached DBpedia SPARQL Client (`src/kg_client.py`)**:
   - Queries the public DBpedia SPARQL endpoint (`https://dbpedia.org/sparql`) with retry/backoff logic and stores successful responses in a thread-safe SQLite cache (`cache/sparql_cache.sqlite`).
   - Failed or timed-out queries are never cached as permanent negatives.
3. **Leak-Free Feature Extraction (`src/features.py`)**:
   - Extracts structural and semantic signals for each triple: direct ontology/property matches (`dbo:`, `dbp:`), conflict indicators (whether DBpedia already has a different known object for the subject-predicate pair), string/substring similarity against known objects, Wikipedia link co-occurrence, entity degrees, and RDF type compatibility.
   - Computes fold-local smoothed target statistics inside each CV fold to prevent data leakage.
4. **Model Training & Inference (`src/train.py`, `src/predict.py`)**:
   - Evaluates a `HistGradientBoostingClassifier` pipeline using 5-fold Stratified Cross-Validation, then fits on the full training set and serializes the model + encoder via `joblib`.
