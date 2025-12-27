# Project Technology Stack

This document outlines the software technologies, libraries, and frameworks used across the PFAS Risk Intelligence Platform.

## 1. Core Runtime

* **Language:** Python 3.10+
* **Environment Management:** Conda / Pip

## 2. Web Application & Dashboard

The interactive user interface is built with **Streamlit**, enabling rapid data visualization and model interaction.

* **[Streamlit](https://streamlit.io/):** Main framework for the web app (`app.py`).
* **[Plotly Express & Graph Objects](https://plotly.com/python/):** Interactive charting (Scatter plots, Risk Matrices).
* **[Stmol](https://github.com/napoles-uach/stmol):** Embedding 3D molecule viewers in Streamlit.
* **[Py3Dmol](https://pypi.org/project/py3Dmol/):** 3D molecular visualization (WebGL).

## 3. Cheminformatics & Data Processing

Handling chemical structures and molecular descriptors.

* **[RDKit](https://www.rdkit.org/):** The core cheminformatics library used for:
  * SMILES parsing and validation.
  * 2D structure generation.
  * 3D conformer generation (finding coordinates).
  * Calculating molecular properties (MW, LogP, TPSA, etc.).
* **[Pandas](https://pandas.pydata.org/):** Extensive use for data handling (`DataFrame` manipulation, combining CSVs).
* **[NumPy](https://numpy.org/):** High-performance numerical operations and array handling.

## 4. Deep Learning & Graph Neural Networks (Approach 1)

Used for learning molecular representations directly from chemical graphs.

* **[PyTorch](https://pytorch.org/):** The primary deep learning framework.
* **[PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/):** Extension library for Graph Neural Networks.
  * **GINConv:** Graph Isomorphism Network layers for powerful graph embeddings.
  * **Global Pooling:** Aggregating node features into graph-level vectors.
* **Contrastive Learning:** Implementation of SimCLR-style self-supervised learning with NT-Xent loss.

## 5. Machine Learning & Predictive Modeling (Approach 2)

Used for toxicity prediction (LD50, Mutagenicity) based on features from RDKit and GNNs.

* **[Scikit-Learn](https://scikit-learn.org/):**
  * **RandomForestRegressor / Classifier:** Robust ensemble models for predictions.
  * **GradientBoosting:** Advanced boosting algorithms.
  * **Model Selection:** `train_test_split`, `KFold`, `StratifiedKFold`, `cross_val_score`.
  * **Metrics:** RMSE, R2, ROC-AUC, Precision-Recall.
* **[Joblib](https://joblib.readthedocs.io/):** Saving and loading trained models (`.pkl` files).

## 6. Visualization & Reporting

Static plots and analysis generation.

* **[Matplotlib](https://matplotlib.org/):** Base plotting library for loss curves and custom visualizations.
* **[Seaborn](https://seaborn.pydata.org/):** Statistical data visualization (Regression plots, Feature Importance bars).

## 7. Workflow Orchestration

* **Custom Python Scripts:** The pipeline is orchestrated via `run_full_pipeline.py`, which uses the `subprocess` module to manage dependencies and execution order across different "Approach" directories.

## Summary Table

| Category | Key Technologies |
| :--- | :--- |
| **Frontend** | Streamlit, Plotly, Stmol, Py3Dmol |
| **Chemistry** | RDKit |
| **Deep Learning** | PyTorch, PyTorch Geometric |
| **Traditional ML** | Scikit-Learn, Joblib |
| **Data** | Pandas, NumPy |
| **Viz** | Matplotlib, Seaborn |
