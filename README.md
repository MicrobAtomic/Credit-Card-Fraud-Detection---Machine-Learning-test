# Credit Card Fraud Detection

> Version française : [README.fr.md](README.fr.md)

Machine learning experiments on the Kaggle dataset
`mlg-ulb/creditcardfraud`.

The dataset is highly imbalanced:

- 284,807 transactions
- 492 fraud cases
- about 0.172% fraud

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python data-import.py
python models/random-forest.py
```

`data-import.py` downloads `creditcard.csv` locally next to the project files.
The dataset file is ignored by git.

## Tested Models

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. LightGBM
6. CatBoost
7. Isolation Forest
8. KNN
9. SVM

## Project Files

- `data-import.py`: dataset download
- `utils/model_utils.py`: data loading, train/test split, shared result display
- `utils/feature_utils.py`: experimental feature creation
- `models/`: standalone model scripts
- `experiments/isolation-feature-test.py`: Isolation Forest score used as a feature
- `experiments/blending-test.py`: weighted blending between strong models
- `experiments/blending-knn-test.py`: weighted blending with a small KNN weight
- `experiments/false-negative-overlap.py`: analysis of frauds missed by each model

## Evaluation

Accuracy alone is not enough because fraud is very rare. The main metrics are:

| Metric | Meaning |
| --- | --- |
| Precision | How many predicted frauds are real frauds. |
| Recall | How many real frauds are detected. |
| F1-score | Balance between precision and recall. |
| ROC-AUC | Global ranking quality of fraud scores. |
| PR-AUC | Precision/recall quality on the rare fraud class. |
| FP | False positives: normal transactions predicted as fraud. |
| FN | False negatives: frauds predicted as normal. |

## Results

The table is ordered by the practical FN/FP trade-off. A small FN gain is not
useful if it creates too many false positives.

| Model | Precision | Recall | F1-score | Accuracy | ROC-AUC | PR-AUC | FP | FN | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9247 | 0.8776 | 0.9005 | 0.9997 | 0.9581 | 0.8678 | 7 | 12 | Best current trade-off |
| CatBoost | 0.8947 | 0.8673 | 0.8808 | 0.9996 | 0.9781 | 0.8780 | 10 | 13 | Very close to Random Forest |
| XGBoost | 0.9222 | 0.8469 | 0.8830 | 0.9996 | 0.9728 | 0.8874 | 7 | 15 | Excellent PR-AUC, more FN |
| LightGBM | 0.9111 | 0.8367 | 0.8723 | 0.9996 | 0.9717 | 0.8584 | 8 | 16 | Good tree-based baseline |
| SVM | 0.8367 | 0.8367 | 0.8367 | 0.9994 | 0.9742 | 0.7516 | 16 | 16 | Correct, but less competitive |
| KNN | 0.5959 | 0.8878 | 0.7131 | 0.9988 | 0.9484 | 0.8308 | 59 | 11 | Lowest FN, but too many FP |
| Logistic Regression | 0.7155 | 0.8469 | 0.7757 | 0.9992 | 0.9706 | 0.7449 | 33 | 15 | Good interpretable baseline |
| Gradient Boosting | 0.8889 | 0.8163 | 0.8511 | 0.9995 | 0.9792 | 0.8540 | 10 | 18 | Low FP, too many FN |
| Isolation Forest | 0.3509 | 0.4082 | 0.3774 | 0.9977 | 0.9612 | 0.3292 | 74 | 58 | Useful idea, weak standalone model |

Current best model: `Random Forest` with FP 7 and FN 12.

## Experiments

### Isolation Forest As Feature

Adding the Isolation Forest anomaly score as a new feature did not improve the
best supervised models.

| Model | Result with `isolation_score` | Comparison |
| --- | --- | --- |
| Random Forest | FP 14, FN 12, F1 0.8687 | Same FN, more FP |
| CatBoost | FP 10, FN 16, F1 0.8632 | More FN |
| XGBoost | FP 13, FN 15, F1 0.8557 | Same FN, more FP |
| LightGBM | FP 16, FN 18, F1 0.8247 | More FP and more FN |

### Simple Blending

Weighted blending between Random Forest, CatBoost and XGBoost was tested.

| Result | FP | FN | F1-score | Weights | Comparison |
| --- | --- | --- | --- | --- | --- |
| Best F1 | 4 | 14 | 0.9032 | RF 0.90, CatBoost 0.05, XGBoost 0.05, threshold 0.43 | Slightly better F1, but more FN |
| Best low-FN with FP <= 50 | 8 | 12 | 0.8958 | RF 0.80, CatBoost 0.15, XGBoost 0.05, threshold 0.27 | Same FN as Random Forest, more FP |

Grid tested: 171 weight combinations, 46 thresholds, 7,866 total combinations.

### Blending With KNN

Adding a small KNN weight did not improve the main trade-off.

| Result | FP | FN | F1-score | Weights | Comparison |
| --- | --- | --- | --- | --- | --- |
| Best F1 | 5 | 14 | 0.8984 | RF 0.80, CatBoost 0.05, XGBoost 0.10, KNN 0.05, threshold 0.45 | Good F1, more FN than Random Forest |
| Best low-FN with FP <= 50 | 37 | 11 | 0.7838 | RF 0.20, CatBoost 0.70, XGBoost 0.05, KNN 0.05, threshold 0.10 | Lower FN, but too many FP |

Grid tested: 605 weight combinations, 56 thresholds, 33,880 total combinations.

### False Negative Overlap

The most important finding is that the strongest models miss the same hard
fraud cases.

| Analysis | Result |
| --- | --- |
| Fraud cases in the test set | 98 |
| Frauds missed by at least one model | 22 |
| Frauds missed by all tested models | 12 |

Fraud indices missed by all tested models:
`623`, `10497`, `50537`, `68067`, `72757`, `96341`, `119714`, `157585`,
`157918`, `182992`, `219025`, `245347`.

This explains why blending does not really improve the best model: the models
are making mistakes on the same difficult cases.

## Next Steps

The next work should focus on the 12 frauds missed by all tested models.

1. Analyze the 12 missed frauds  
   Compare their `Time`, `Amount` and `V1` to `V28` values with normal
   transactions and detected frauds.

2. Compare missed frauds vs detected frauds  
   Look for columns where difficult frauds differ from easier frauds.

3. Inspect model scores on the 12 cases  
   Check if models are strongly confident they are normal, or if the scores are
   just under the decision threshold.

4. Test targeted features  
   Add only features motivated by the analysis, for example around `Amount`,
   `Time`, or distance to known frauds.

5. Validate with several train/test splits  
   Check whether the conclusion is stable or only caused by the current 80/20
   split.
