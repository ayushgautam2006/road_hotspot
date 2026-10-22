# 🚧 road_hotspots – Road Accident Hotspot Analysis

A Python toolkit for identifying, scoring, and visualising road accident hotspots using spatial clustering, severity modelling, and interactive maps.

---

## 📂 Project Structure

```
road_hotspots/
├── data/
│   ├── raw/                  # Original accident datasets (CSV / GeoJSON)
│   └── processed/            # Cleaned & feature-enriched data
├── outputs/
│   ├── maps/                 # Exported Folium HTML maps
│   ├── figures/              # Matplotlib charts and SHAP plots
│   └── models/               # Serialised trained models
├── src/
│   ├── __init__.py
│   ├── config.py             # All paths, constants & tunable parameters
│   ├── data_loader.py        # CSV / GeoJSON ingestion & validation
│   ├── cleaning.py           # Missing-value, outlier & dedup pipeline
│   ├── features.py           # Temporal, categorical & severity features
│   ├── hotspots.py           # HDBSCAN spatial clustering
│   ├── severity_model.py     # LightGBM classifier + SHAP explanations
│   ├── priority.py           # Composite priority scoring for clusters
│   └── map_builder.py        # Folium map generation & export
├── tests/
│   ├── __init__.py
│   └── test_config.py        # Starter smoke tests
├── notebooks/                # Exploratory Jupyter notebooks
├── dashboard.py              # Streamlit interactive dashboard
├── requirements.txt          # Pinned dependencies
└── README.md                 # This file
```

---

## ⚙️ Quick Start

### 1. Prerequisites

- **Python 3.9+**
- **pip** (or **conda**)

### 2. Create a virtual environment

```bash
# Create venv
python -m venv .venv

# Activate – Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate – Windows (cmd)
.\.venv\Scripts\activate.bat

# Activate – macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

---

## 🔧 Configuration

All tunable parameters live in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `RANDOM_SEED` | `42` | Global reproducibility seed |
| `CLUSTER_RADIUS_KM` | `0.5` | HDBSCAN neighbourhood radius (km) |
| `MIN_CLUSTER_SIZE` | `5` | Minimum accidents to form a cluster |
| `SEVERITY_WEIGHTS` | `{fatal: 5, serious: 3, slight: 1}` | Numeric weights per severity class |
| `PRIORITY_WEIGHT_*` | `0.4 / 0.4 / 0.2` | Frequency / severity / trend weights |
| `LIGHTGBM_PARAMS` | see config | LightGBM hyper-parameters |

---

## 📑 Module Overview

| Module | Responsibility |
|---|---|
| `data_loader` | Ingest CSV / GeoJSON, validate schema |
| `cleaning` | Handle nulls, duplicates, type coercion, bounds filtering |
| `features` | Derive temporal bins, encode categoricals, map severity weights |
| `hotspots` | HDBSCAN clustering on haversine coordinates |
| `severity_model` | Train LightGBM, evaluate, SHAP explanations, save/load |
| `priority` | Composite scoring (frequency + severity + trend) |
| `map_builder` | Folium heatmaps, cluster markers, HTML export |
| `dashboard` | Streamlit app tying everything together |

---

## 📄 License

MIT – see `LICENSE` for details.
