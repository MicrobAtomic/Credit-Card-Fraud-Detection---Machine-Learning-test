# Test weighted blending with KNN as a light extra signal
#
# This experiment is close to blending-test.py, but adds KNN.
#
# Why KNN?
# KNN has a different logic from tree models:
# - tree models learn split rules
# - KNN compares distances between transactions
#
# The hope:
# KNN may catch some frauds that tree models miss.
#
# The risk:
# KNN can create many false positives and is slow on large datasets.
# Because of that, KNN only gets a small weight in the final blend.

from pathlib import Path
import sys

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# The script is inside experiments/.
# The project root is one parent folder above this file.
# Adding it to sys.path allows imports from utils/.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Shared helpers:
# - get_metrics: calculate precision, recall, F1, FP, FN, etc.
# - print_row: print one metrics row
# - best_by_f1: select the best global F1-score
# - best_by_low_fn: select the lowest FN under an FP limit
from utils.experiment_utils import get_metrics, print_row, best_by_f1, best_by_low_fn
from utils.model_utils import RANDOM_STATE, load_train_test_data


# Number of normal transactions kept for KNN training.
#
# We keep all frauds, but only a sample of normal transactions.
# This is a speed compromise:
# - using all normal rows makes KNN very slow
# - using too few normal rows can make KNN unstable
TRAIN_NORMAL_SAMPLE_SIZE = 40000


def weight_grid(step=0.05):
    # KNN is noisy alone, so it only gets a small weight here
    #
    # step=0.05 means:
    # - 1 unit = 0.05
    # - 20 units = 1.00
    #
    # All weights must add up to 1.00.
    units = int(1 / step)

    # KNN is limited to 0.05, 0.10, 0.15, 0.20 or 0.25.
    # It should be an extra signal, not the main model.
    knn_units_list = [1, 2, 3, 4, 5] # 0.05 to 0.25

    for knn_units in knn_units_list:
        # The remaining weight is shared between RF, CatBoost and XGBoost.
        remaining_units = units - knn_units

        for rf_units in range(1, remaining_units):
            for cat_units in range(1, remaining_units - rf_units):
                # XGBoost gets the remaining non-KNN weight.
                xgb_units = remaining_units - rf_units - cat_units
                if xgb_units <= 0:
                    continue

                # Convert integer units back to real weights.
                yield {
                    "Random Forest": round(rf_units * step, 2),
                    "CatBoost": round(cat_units * step, 2),
                    "XGBoost": round(xgb_units * step, 2),
                    "KNN": round(knn_units * step, 2),
                }


def make_knn_train_set(X_train, y_train):
    # KNN is slow on a big train set because it compares distances with many rows
    # Keep all frauds, then add a fixed sample of normal transactions
    #
    # This keeps the fraud signal complete while reducing the number of
    # normal rows used for distance comparisons.
    fraud_index = y_train[y_train == 1].index

    # sample(...) keeps the experiment reproducible because random_state is fixed.
    normal_index = y_train[y_train == 0].sample(
        n=TRAIN_NORMAL_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).index

    # union(...) combines fraud indices and sampled normal indices.
    sample_index = normal_index.union(fraud_index)

    # Return the smaller KNN-specific train set.
    return X_train.loc[sample_index], y_train.loc[sample_index]


# Load the shared 80/20 train/test split.
X_train, X_test, y_train, y_test = load_train_test_data()


# First define the three strong supervised models.
# KNN is added after creating its smaller train set.
models = [
    (
        "Random Forest",
        make_pipeline(
            # Kept for consistency with the other scripts.
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=90,
                class_weight={0: 1, 1: 15},
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        X_train,
        y_train,
    ),
    (
        "CatBoost",
        # CatBoost does not need StandardScaler.
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
        X_train,
        y_train,
    ),
    (
        "XGBoost",
        make_pipeline(
            # Kept for consistency with the XGBoost model script.
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
        X_train,
        y_train,
    ),
]


# Build the smaller train set used only by KNN.
X_train_knn, y_train_knn = make_knn_train_set(X_train, y_train)

# Add KNN to the list.
# The tuple is slightly different from blending-test.py:
# it also stores X_fit and y_fit because KNN uses a smaller train set.
models.append(
    (
        "KNN",
        make_pipeline(
            # Scaling is very important for KNN because KNN uses distances.
            # Without scaling, large-value columns can dominate the distance.
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=9,
                weights="distance",
                n_jobs=-1,
            ),
        ),
        X_train_knn,
        y_train_knn,
    )
)


print("\n=== Blending with light KNN experiment ===")
print("Models: Random Forest + CatBoost + XGBoost + small KNN weight")

# Store each model fraud score on X_test.
# The scores are arrays with one value per test row.
scores = {}

for name, model, X_fit, y_fit in models:
    print(f"\nTraining {name}...")

    # Train each model on its own train data.
    # RF, CatBoost and XGBoost use the full train set.
    # KNN uses the reduced train set.
    model.fit(X_fit, y_fit)

    # Keep only the fraud probability column.
    # Warning: KNN predict_proba can be slow because it computes distances
    # between test rows and many train rows.
    scores[name] = model.predict_proba(X_test)[:, 1]


# Try every allowed weight mix.
weights_list = list(weight_grid())

# Test thresholds from 0.10 to 0.65.
# The lower start is used because KNN/blended scores can need a lower threshold
# to reduce false negatives.
thresholds = [round(0.10 + 0.01 * index, 2) for index in range(56)]

# Store one result per weight + threshold combination.
rows = []

for weights in weights_list:
    # Weighted average between four models.
    # KNN has a small weight, so it can influence the result without dominating.
    blend_score = (
        weights["Random Forest"] * scores["Random Forest"]
        + weights["CatBoost"] * scores["CatBoost"]
        + weights["XGBoost"] * scores["XGBoost"]
        + weights["KNN"] * scores["KNN"]
    )

    for threshold in thresholds:
        # Evaluate this blend and this threshold.
        row = get_metrics(y_test, blend_score, threshold)

        # Keep weights to understand which blend created this result.
        row["weights"] = weights
        rows.append(row)


# Global best F1-score.
best_f1 = best_by_f1(rows)

# Lowest false negatives while keeping false positives under 50.
best_low_fn = best_by_low_fn(rows, max_fp=50)

print("\nGrid tested:")
print(
    len(weights_list),
    "weight combinations,",
    len(thresholds),
    "thresholds,",
    len(rows),
    "total combinations",
)

print("\nBest F1:")
print_row("Blend + KNN", best_f1)
print("weights:", best_f1["weights"])

if best_low_fn is not None:
    print("\nBest low-FN with FP <= 50:")
    print_row("Blend + KNN", best_low_fn)
    print("weights:", best_low_fn["weights"])
