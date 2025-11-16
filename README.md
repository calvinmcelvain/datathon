# Datathon 2025

## Team Information

**Team Name:** Rocket Team

**Team Members:**

- Mahesh Kanuri
- Ariadna Orbe Vivero
- Bowen Su
- Calvin McElvain

## Project Structure

```text
datathon/
├── _data/
│   ├── test.csv
│   └── train.csv
├── docs/
│   └── datathon_2025.pdf
├── code/
│   └── .
├── README.md
└── requirements.txt
```

## Check List

### Feature Selection

- [X] Apply `VarClusHi` for hierarchical variable clustering.
  - [X] Determine optimal number of clusters.
  - [X] Select representative variables from each cluster.

### Models

- [X] XGBoost
- [X] Random Forest
- [X] TabPFN
- [ ] LightGBM

For each model:

- [X] Build baseline frequency model (predict `clm` or `numclaims`) and baseline severity model (predict `claimcst0` for claims only).
- [X] Handle exposure offset/weights.
- [X] Apply interactive hyperparameter tuning.
- [X] Document hyperparameters tested and final selections.
- [X] Evaluate on validation set (`sample=='2|val'`).

### Model Selection

- [X] Model Explainability: Apply `SHAP` analysis to best performing model(s).
- [X] Compare all models on validation set.
  - [X] Calculate performance metrics (RMSE, MAE, $R^2$, etc.)
  - [X] Create comparison visualizations.

### Inference

- [X] Generate predictions on `test.csv` dataset.
- [X] Identify underpriced/overpriced segments.
- [X] Develop rate adjustment recommendations.
- [X] Create targeted marketing campaign strategies.

### Submissions

- [X] Code
- [X] One page report
- [X] Presentation
