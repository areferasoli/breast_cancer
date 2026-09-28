# Breast Cancer Diagnosis: Logistic Regression vs Random Forest

Binary classification (malignant vs benign) on the UCI **Breast Cancer Wisconsin (Diagnostic)** dataset, with an emphasis on rigorous evaluation: leakage-free pipelines, **nested cross-validation**, and a decision threshold tuned on training data only.

> Educational project. Not a clinical tool.

## Dataset
- 569 samples, 30 numeric features (mean / standard error / worst of 10 cell-nucleus measurements)
- 212 malignant (37.3%), 357 benign (62.7%), no missing values
- Loaded via `sklearn.datasets.load_breast_cancer`, so no manual download is needed
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Method
1. EDA: class balance, distributions, correlations (many features are highly collinear, 21 pairs with |r| > 0.9)
2. Stratified 80/20 train/test split; the test set is used only for final checks
3. Models in `Pipeline`s (scaling inside the pipeline, so no leakage): Logistic Regression and Random Forest
4. Nested CV: 3-fold inner loop for hyperparameter tuning, 5-fold outer loop for evaluation
5. Final model selected **programmatically** by nested-CV ROC-AUC
6. Decision threshold chosen on **out-of-fold training predictions** (maximising F2 to favour recall), then checked once on the test set
7. Bootstrap confidence intervals to show how uncertain a 114-sample test set is

## Results

Nested cross-validation (mean over 5 outer folds, training data only):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.974 | 0.989 | 0.941 | 0.964 | **0.995** |
| Random Forest | 0.965 | 0.964 | 0.941 | 0.952 | 0.989 |

Held-out test set (114 samples, 42 malignant), final model = Logistic Regression:

| Threshold | Accuracy | Precision | Recall | False negatives | False positives |
|---|---|---|---|---|---|
| 0.50 (default) | 0.965 | 0.975 | 0.929 | 3 | 1 |
| 0.30 (tuned on training data) | 0.982 | 0.976 | 0.976 | 1 | 1 |

Bootstrap 95% CI for recall at threshold 0.30: about 0.92 to 1.00 (wide, because the test set is small).

![Confusion matrix](images/confusion_matrix.png)
![ROC and PR curves](images/roc_pr_curves.png)

## Key takeaways
- A regularised linear model was enough: it matched or beat a tuned Random Forest and is simpler.
- In cancer screening a missed malignant case costs more than a false alarm, so lowering the threshold to favour recall is reasonable. It cut false negatives from 3 to 1 on the test set.
- Feature importance is only a rough guide because of strong multicollinearity.

## Limitations
- Small dataset, single hold-out split, no external validation
- Probabilities not calibrated; no clinical cost model
- Not validated for any clinical use

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Breast_Cancer_ML.ipynb
```

## Repository structure
```
.
├── Breast_Cancer_ML.ipynb   # full analysis with outputs
├── requirements.txt
├── README.me
```
