# PFAS Policy Platform: Startup Guide

## Prerequisites

Ensure you have the following installed (the setup script likely handled this):

- Python 3.10+
- Dependencies: `streamlit`, `plotly`, `rdkit` (via conda/pip), `stmol`, `py3dmol`, `ipython_genutils`.

## Quick Start

1. Open a terminal in the project root: `C:\Users\Abrar\Downloads\pfas_datasets`.
2. Run the following command:

    ```bash
    streamlit run platform_dashboard/app.py
    ```

3. The dashboard will open automatically in your default browser at `http://localhost:8501` (or 8502 if 8501 is busy).

## Troubleshooting

- **ModuleNotFoundError: 'stmol'**: Run `pip install stmol`.
- **ModuleNotFoundError: 'ipython_genutils'**: Run `pip install ipython_genutils`.
- **3D Molecules not showing**: Ensure your browser supports WebGL. 2D images will always work as fallback.

## Dashboard Features

- **Tab 1: Global Policy Dashboard:** View the "Risk vs. Uncertainty" matrix. Hover over points to identify specific chemicals.
- **Tab 2: Molecule Screener:** Enter a SMILES code to visualize structure and predicted risk.
- **Tab 3: Cluster Analysis:** deeply explore the 7 structural classes found by the AI.
