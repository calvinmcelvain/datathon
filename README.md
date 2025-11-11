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

- [ ] Apply `VarClusHi` for hierarchical variable clustering.
  - [ ] Determine optimal number of clusters.
  - [ ] Select representative variables from each cluster.

### Models

- [ ] XGBoost
- [ ] Random Forest
- [ ] TabPFN
- [ ] LightGBM

For each model:

- [ ] Build baseline frequency model (predict `clm` or `numclaims`) and baseline severity model (predict `claimcst0` for claims only).
- [ ] Handle exposure offset/weights.
- [ ] Apply interactive hyperparameter tuning.
- [ ] Document hyperparameters tested and final selections.
- [ ] Evaluate on validation set (`sample=='2|val'`).

### Model Selection

- [ ] Model Explainability: Apply `SHAP` analysis to best performing model(s).
- [ ] Compare all models on validation set.
  - [ ] Calculate performance metrics (RMSE, MAE, $R^2$, etc.)
  - [ ] Create comparison visualizations.

### Inference

- [ ] Generate predictions on `test.csv` dataset.
- [ ] Identify underpriced/overpriced segments.
- [ ] Develop rate adjustment recommendations.
- [ ] Create targeted marketing campaign strategies.

### Submissions

- [ ] Code
- [ ] One page report
- [ ] Presentation
