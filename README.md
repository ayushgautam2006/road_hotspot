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

## 📊 Feature Dictionary

All features produced by [`src/features.py`](src/features.py) via `build_features()`.
Severity and num_casualties are **excluded** from *X* to prevent target leakage.

### Temporal Features

| Feature | Type | Description | Why it matters |
|---|---|---|---|
| `hour` | int 0-23 | Hour of day extracted from datetime | Accident frequency and severity vary by time of day |
| `weekday` | int 0-6 | Day of week (0 = Monday) | Weekend vs weekday traffic patterns differ |
| `month` | int 1-12 | Calendar month | Seasonal weather and daylight effects |
| `is_weekend` | binary | 1 if Saturday or Sunday | Weekend driving behaviour (alcohol, leisure trips) |
| `is_rush_hour` | binary | 1 if hour ∈ {7,8,9,16,17,18} | Congestion raises minor-accident frequency |
| `is_night` | binary | 1 if hour ∈ {22,23,0,1,2,3,4,5} | Reduced visibility, fatigue, impaired driving |

### Interaction Features

| Feature | Type | Description | Why it matters |
|---|---|---|---|
| `night_and_unlit` | binary | Night-time AND dark unlit road | Extreme low-visibility compound risk |
| `wet_and_high_speed` | binary | Wet/icy/flooded surface AND speed ≥ 50 | Loss-of-control compound risk |
| `junction_and_urban` | binary | At a junction AND urban area | Complex turning manoeuvres in dense traffic |

### Spatial Features

| Feature | Type | Description | Why it matters |
|---|---|---|---|
| `cell_accident_count` | int ≥ 0 | Historical accident count in the same ~250 m grid cell (**leave-one-out**: excludes the row itself) | Proxy for location-level danger independent of clustering |

### Road / Environment (one-hot encoded)

| Feature group | Source column | Values | Why it matters |
|---|---|---|---|
| `road_type_*` | road_type | single_carriageway, dual_carriageway, roundabout, one_way, slip_road, unknown | Road geometry affects collision dynamics |
| `light_conditions_*` | light_conditions | daylight, dark_lit, dark_unlit, dawn_dusk | Visibility directly influences severity |
| `weather_*` | weather | fine, rain, fog, snow, high_wind, other, unknown | Adverse weather reduces grip and visibility |
| `road_surface_*` | road_surface | dry, wet, ice, snow, flood, unknown | Surface condition affects braking distance |
| `junction_type_*` | junction_type | not_junction, crossroads, t_junction, roundabout, slip_road, other, unknown | Junction geometry creates conflict points |
| `urban_rural_*` | urban_rural | urban, rural | Rural roads have higher speed, different hazards |

### Numeric Features (kept as-is)

| Feature | Type | Description | Why it matters |
|---|---|---|---|
| `speed_limit` | int | Posted speed limit (mph) – ordinal numeric | Higher speed → longer stopping distance, more severe impact |
| `num_vehicles` | int | Number of vehicles involved | Multi-vehicle collisions tend to be more severe |

### Target Variables (returned separately, NOT in X)

| Target | Type | Description |
|---|---|---|
| `y_binary` (`fatal_or_serious`) | binary | 1 if severity ∈ {1, 2}, else 0 |
| `y_multiclass` (`severity`) | int 1/2/3 | 1 = fatal, 2 = serious, 3 = slight |

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
