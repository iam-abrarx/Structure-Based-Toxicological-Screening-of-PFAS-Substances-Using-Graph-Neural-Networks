# PFAS Risk Intelligence Platform

## Overview

The **PFAS Risk Intelligence Platform** is a comprehensive computational data pipeline and interactive dashboard designed to assess, predict, and visualize the risks associated with Per- and Polyfluoroalkyl Substances (PFAS). This project integrates diverse datasets—from high-throughput screening (Tox21) to environmental monitoring (drinking water/groundwater)—to provide a holistic view of PFAS toxicity and exposure.

## Key Features

- **Global Risk Landscape**: Interactive 2D/3D visualizations of chemical space and risk metrics.
- **Toxicity Prediction**: Machine learning models (Random Forest, Gradient Boosting) and Graph Neural Networks (GNNs) to predict toxicity endpoints like LD50 and mutagenicity.
- **Data Integration**: Harmonizes structural data, experimental toxicity results, and real-world environmental occurrence data.
- **Interactive Dashboard**: A Streamlit-based user interface for policymakers and researchers to explore data, query specific molecules (SMILES), and view structural clusters.

## Technology Stack

This project leverages a modern Python-based stack for cheminformatics, machine learning, and web development.

| Component | Technologies |
| :--- | :--- |
| **Core Runtime** | Python 3.10+ |
| **Frontend/Dashboard** | [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/), [Stmol](https://github.com/napoles-uach/stmol) |
| **Cheminformatics** | [RDKit](https://www.rdkit.org/), Py3Dmol |
| **Deep Learning** | [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (GNNs) |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/), Joblib |
| **Data Processing** | Pandas, NumPy |

## Dataset Inventory

The platform utilizes several key data sources:

- **Structural Data**: EPA PFAS Master List V2.
- **Toxicology Data**: Tox21 high-throughput screening assays (approx. 69 endpoints e.g., PPAR-gamma, p53).
- **Environmental Data**: Drinking water and groundwater contamination records.
- **Derived Data**: Computed risk matrices, GNN embeddings, and structural clusters.

## Installation & Setup

### Prerequisites

- Python 3.10+
- Recommended to use a virtual environment (Conda or venv).

### Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd pfas_datasets
    ```

2. Install dependencies:

    ```bash
    pip install streamlit plotly rdkit stmol py3dmol ipython_genutils torch torch-geometric scikit-learn pandas numpy
    # Note: RDKit is often easier to install via Conda:
    # conda install -c conda-forge rdkit
    ```

## Usage

To launch the interactive dashboard:

```bash
streamlit run platform_dashboard/app.py
```

The application will open in your browser (usually at `http://localhost:8501`).

### Dashboard Structure

- **Global Policy Dashboard**: View "Risk vs. Uncertainty" matrix.
- **Molecule Screener**: Input SMILES strings to visualize 3D structures and predict risk.
- **Cluster Analysis**: Explore chemical structural classes identified by the AI models.

## Project Structure

- `platform_dashboard/`: Streamlit application code.
- `approach1_gnn/`: Graph Neural Network implementation.
- `approach2_tox_pred/`: Traditional ML toxicity prediction models.
- `approach5_tox21/`: Tox21 data scraping and processing.
- `data/`: Raw and processed datasets.
- `run_full_pipeline.py`: Orchestration script for the data pipeline.
