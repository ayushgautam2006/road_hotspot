"""Streamlit dashboard for the road_hotspots project.

Launch with:  ``streamlit run dashboard.py``

Provides interactive controls for exploring accident hotspots,
severity predictions, and priority rankings on an embedded map.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for the Streamlit dashboard.

    Sets up the page layout, sidebar filters, and renders:
    - An interactive Folium map of hotspots
    - Cluster summary table with priority scores
    - Severity-model metrics and SHAP feature-importance chart
    - Temporal and categorical distribution charts
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
