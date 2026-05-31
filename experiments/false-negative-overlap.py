# Check if models miss the same fraud transactions
#
# This script answers one key question:
# "Are models missing different frauds, or are they all missing the same frauds?"
#
# Why this matters:
# - If models miss different frauds, blending/voting/stacking can help.
# - If models miss the same frauds, combining them will probably not fix the
#   main problem.
#
# The output lists:
# - false negatives for each model
# - how many frauds are missed by all models
# - pairwise overlap between missed frauds
# - whether another model catches the frauds missed by Random Forest

from pathlib import Path
import sys

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# This file is in experiments/.
# The project root is one folder above, and it contains utils/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# get_metrics is reused to calculate FP, FN, precision, recall and F1.
from utils.experiment_utils import get_metrics
from utils.model_utils import RANDOM_STATE, load_train_test_data


# Load the same train/test split used everywhere else.
# This is important because we want to compare models on the exact same test rows.
X_train, X_test, y_train, y_test = load_train_test_data()


# Each tuple contains:
# - model name
# - model object
# - threshold used to convert score into class 0/1
# - score method
#
# Most models use predict_proba, which returns a fraud probability.
# LinearSVC does not have predict_proba by default, so it uses decision_function.
models = [
    (
        "Random Forest",
        make_pipeline(
            # Kept for consistency with other scripts.
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=90,
                class_weight={0: 1, 1: 15},
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        0.30,
        "predict_proba",
    ),
    (
        "CatBoost",
        # CatBoost works directly with numeric features.
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.09,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1, 40],
            random_seed=RANDOM_STATE,
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
        ),
        0.40,
        "predict_proba",
    ),
    (
        "XGBoost",
        make_pipeline(
            # Kept to match the main XGBoost script.
            StandardScaler(),
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.2,
                max_depth=5,
                scale_pos_weight=25,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        0.60,
        "predict_proba",
    ),
    (
        "LightGBM",
        # Tree boosting model, close family to XGBoost/CatBoost.
        LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=100,
            scale_pos_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        0.30,
        "predict_proba",
    ),
    (
        "Gradient Boosting",
        make_pipeline(
            # HistGradientBoostingClassifier is sklearn's faster gradient boosting.
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                class_weight={0: 1, 1: 5},
                random_state=RANDOM_STATE,
            ),
        ),
        0.60,
        "predict_proba",
    ),
    (
        "Logistic Regression",
        make_pipeline(
            # Logistic Regression needs scaling because coefficients are affected
            # by feature scale.
            StandardScaler(),
            LogisticRegression(
                C=0.01,
                class_weight={0: 1, 1: 10},
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
        0.60,
        "predict_proba",
    ),
    (
        "SVM",
        make_pipeline(
            # SVM needs scaling because it uses margins/distances.
            StandardScaler(),
            LinearSVC(
                C=0.001,
                class_weight={0: 1, 1: 40},
                dual=False,
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
        0.60,
        "decision_function",
    ),
]


def get_score(model, score_method):
    # Return one score per row in X_test.
    #
    # For probabilistic models:
    # predict_proba(X_test) returns two columns:
    # - column 0: probability of normal
    # - column 1: probability of fraud
    #
    # For LinearSVC:
    # decision_function returns a margin score.
    # A higher score means "more likely to be fraud".
    if score_method == "predict_proba":
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def get_false_negative_indices(y_score, threshold):
    # Convert scores into final predictions.
    # If score >= threshold => predict fraud (1)
    # Else => predict normal (0)
    y_pred = (y_score >= threshold).astype(int)

    # A false negative is:
    # - real class is fraud (y_test == 1)
    # - predicted class is normal (y_pred == 0)
    #
    # We return the original dataframe indices of these fraud rows.
    return set(y_test[(y_test == 1) & (y_pred == 0)].index)


def get_false_positive_indices(y_score, threshold):
    # Same prediction rule as above.
    y_pred = (y_score >= threshold).astype(int)

    # A false positive is:
    # - real class is normal (y_test == 0)
    # - predicted class is fraud (y_pred == 1)
    return set(y_test[(y_test == 0) & (y_pred == 1)].index)


print("\n=== False negative overlap experiment ===")
print("Goal: check if models miss the same fraud transactions.")

# These dictionaries store sets of row indices.
# Example:
# false_negatives["Random Forest"] = {623, 10497, ...}
#
# Sets are useful because we can calculate intersections and unions easily.
false_negatives = {}
false_positives = {}

for name, model, threshold, score_method in models:
    print(f"\nTraining {name}...")

    # Train the current model.
    model.fit(X_train, y_train)

    # Get the model score on the test set.
    y_score = get_score(model, score_method)

    # Calculate global metrics for this model and threshold.
    metrics = get_metrics(y_test, y_score, threshold)

    # Store the exact fraud indices missed by this model.
    false_negatives[name] = get_false_negative_indices(y_score, threshold)

    # Store false positives too.
    # It is not central for the overlap conclusion, but it can be useful later.
    false_positives[name] = get_false_positive_indices(y_score, threshold)

    # Print a compact metrics line for the model.
    print(
        f"{name:<20}"
        f" FP={metrics['fp']:<4}"
        f" FN={metrics['fn']:<4}"
        f" precision={metrics['precision']:.4f}"
        f" recall={metrics['recall']:.4f}"
        f" f1={metrics['f1']:.4f}"
    )

    # Print the exact missed fraud indices.
    # These indices can be used later to inspect the raw rows in the dataset.
    print("Missed fraud indices:", sorted(false_negatives[name]))


# Keep model names in the same order as the dictionary insertion order.
model_names = list(false_negatives)

# set.intersection(...) keeps only indices present in every model's FN set.
# These are frauds missed by all tested models.
all_missed = set.intersection(*(false_negatives[name] for name in model_names))

# set.union(...) keeps indices present in at least one model's FN set.
# These are frauds missed by at least one model.
missed_by_any = set.union(*(false_negatives[name] for name in model_names))

print("\n=== Summary ===")
print("Fraud cases in test set:", int((y_test == 1).sum()))
print("Frauds missed by at least one model:", len(missed_by_any))
print("Frauds missed by all models:", len(all_missed))
print("Fraud indices missed by all models:", sorted(all_missed))

print("\n=== Pair overlap matrix ===")
print("Each value = number of frauds missed by both models")

# Print a small table header.
# name[:10] keeps each column short enough for terminal output.
print("Model".ljust(22) + "".join(name[:10].ljust(12) for name in model_names))

for left_name in model_names:
    row = left_name.ljust(22)
    for right_name in model_names:
        # Intersection between two FN sets:
        # frauds missed by both models.
        overlap_count = len(false_negatives[left_name] & false_negatives[right_name])
        row += str(overlap_count).ljust(12)
    print(row)

print("\n=== Random Forest missed frauds ===")

# Random Forest is the current best model.
# This section checks if any other model catches the frauds Random Forest missed.
rf_missed = false_negatives["Random Forest"]
for fraud_index in sorted(rf_missed):
    # If fraud_index is NOT in false_negatives[name],
    # then that model caught this fraud.
    caught_by = [
        name
        for name in model_names
        if name != "Random Forest" and fraud_index not in false_negatives[name]
    ]

    # If caught_by is empty, nobody caught this fraud.
    print(f"{fraud_index}: caught by {caught_by if caught_by else 'nobody'}")
