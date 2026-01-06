# Fetal Health Classification from Cardiotocography (CTG)

Biostat 203A final project: predicting fetal health status (**Normal**, **Suspect**, **Pathological**) from CTG-derived numeric features using multinomial and tree-based machine learning models.

## Project goal
CTG interpretation can be subjective; this project builds models to classify fetal health from summary CTG features, with extra attention to performance on the minority (higher-risk) classes **Suspect** and **Pathological**.

## Data
- Source: Kaggle “Fetal Health Classification” dataset (`fetal_health.csv`)
- Size: 2,126 CTG exams, 21 numeric predictors + 1 multiclass outcome
- Class distribution (imbalanced): Normal ~77.8%, Suspect ~13.9%, Pathological ~8.3%
- Missingness: none (all predictors/outcome complete)

## Methods (R + tidymodels)
- Stratified split: 80% train / 20% test
- 5-fold cross-validation on the training set (stratified by class)
- Metrics: accuracy + balanced accuracy (macro-average recall)
- Preprocessing:
  - GLM models: zero-variance removal + standardization (mean 0, SD 1)
  - Tree-based models: zero-variance removal only
- Models fit:
  1. Multinomial logistic regression (`nnet`)
  2. Penalized multinomial logistic regression (elastic-net, `glmnet`) — tuned
  3. Random forest (`ranger`, 1000 trees)
  4. Gradient boosted trees (`xgboost`) — tuned via Latin-hypercube grid search

## Key results (held-out test set)
Overall accuracy:
- **XGBoost:** 0.962
- **Random Forest:** 0.958
- **Penalized Multinomial (glmnet):** 0.911
- **Multinomial (nnet):** 0.908

Minority-class sensitivity (recall):
- **XGBoost:** Suspect 0.906, Pathological 1.000
- **Random Forest:** Suspect 0.920, Pathological 1.000
- **glmnet:** Suspect 0.717, Pathological 0.806
- **nnet:** Suspect 0.704, Pathological 0.763

## Repository contents
- `final-project-203a.Rmd` — full analysis (EDA, modeling, tuning, evaluation)
- `final-project-203a.pdf` — rendered report
- `fetal_health.csv` — dataset (or instructions to download from Kaggle)
