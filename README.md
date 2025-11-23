# Consumer Complaint Dispute Prediction

This repository contains my work for a supervised machine-learning project that predicts whether a consumer will dispute a complaint resolution (“Consumer disputed” column) using the Consumer Complaints dataset.

### Files
- `Q1_LogisticRegression.ipynb` — Baseline preprocessing, TF-IDF text features, categorical/time features, weighted logistic regression, and evaluation.
- `Q1_stratifiedLogisticRegression.ipynb` — Variant using stratified train/test splitting to test stability and class-balance handling.
- `Q1_XGBoost.ipynb` — Gradient-boosting attempt; included for comparison even though it did not outperform the logistic model.
- `submission1.csv` — Final submission file containing predictions for the test set.


The project involved cleaning and preparing both structured and unstructured complaint data, converting narrative text to TF-IDF features, engineering time-based variables, and removing leakage-prone identifiers. Multiple models were evaluated, and the weighted logistic-regression setup achieved the required AUC benchmark on the held-out evaluation.

### Author
Parmeet Singh Majethiya
