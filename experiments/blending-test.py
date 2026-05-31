# Test simple weighted blending between strong models
#
# Blending means:
# - train several models separately
# - get one fraud score from each model
# - compute a weighted average of these scores
# - test many thresholds on this final average score
#
# Goal:
# Check if mixing Random Forest, CatBoost and XGBoost gives a better
# FP/FN trade-off than using Random Forest alone.

from pathlib import Path
import sys

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# This experiment is inside the experiments/ folder.
# The utils/ folder is one level above, so we add the project root to sys.path.
# This allows imports like "from utils.model_utils import ...".
sys.path.append(str(Path(__file__).resolve().parents[1]))

# get_metrics calculates precision, recall, F1, FP, FN, etc.
# print_row prints one result line in a readable format.
# best_by_f1 selects the row with the best F1-score.
# best_by_low_fn selects the row with the lowest FN while keeping FP under a limit.
from utils.experiment_utils import get_metrics, print_row, best_by_f1, best_by_low_fn
from utils.model_utils import RANDOM_STATE, load_train_test_data


def weight_grid(step=0.05):
    # Positive weights only, so each model really participates in the blend
    # step=0.05 means weights can be 0.05, 0.10, 0.15, ...
    # Because all weights must add up to 1.00, we work with integer units.
    #
    # Example with step=0.05:
    # 1 unit = 0.05
    # 20 units = 1.00
    #
    # If RF gets 10 units, CatBoost gets 6 units, XGBoost gets 4 units:
    # RF weight = 0.50
    # CatBoost weight = 0.30
    # XGBoost weight = 0.20
    units = int(1 / step)

    # Try every possible split of the total weight between the three models.
    for rf_units in range(1, units):
        for cat_units in range(1, units - rf_units):
            # XGBoost receives the remaining weight.
            # This guarantees that the three weights sum to exactly 1.00.
            xgb_units = units - rf_units - cat_units
            if xgb_units <= 0:
                continue

            # Convert integer units back into real weights.
            yield {
                "Random Forest": round(rf_units * step, 2),
                "CatBoost": round(cat_units * step, 2),
                "XGBoost": round(xgb_units * step, 2),
            }


# Load the same 80/20 train/test split used by the model scripts.
# X_train / y_train are used to train models.
# X_test / y_test are used only to evaluate the final predictions.
X_train, X_test, y_train, y_test = load_train_test_data()


# Each tuple contains:
# - the model name used in outputs and dictionaries
# - the model object itself
#
# The parameters are copied from the best current individual scripts.
# We do not run a new model grid here; this experiment only tests blending.
models = [
    (
        "Random Forest",
        make_pipeline(
            # The scaler is kept for consistency with other scripts.
            # Random Forest does not really need scaling, but this keeps the
            # pipeline shape similar across experiments.
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=90,
                class_weight={0: 1, 1: 15},
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ),
    (
        "CatBoost",
        # CatBoost already handles numeric columns directly.
        # No StandardScaler is needed here.
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
    ),
    (
        "XGBoost",
        make_pipeline(
            # The scaler is kept here to stay close to the other scripts.
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
    ),
]


print("\n=== Simple blending experiment ===")
print("Models: Random Forest + CatBoost + XGBoost")

# scores will store one array per model.
# Example:
# scores["Random Forest"] = [0.01, 0.92, 0.03, ...]
# Each value is the model fraud probability for one row in X_test.
scores = {}

for name, model in models:
    print(f"\nTraining {name}...")

    # Train the model only on the training set.
    model.fit(X_train, y_train)

    # predict_proba returns two columns:
    # column 0 = probability of class 0 (normal)
    # column 1 = probability of class 1 (fraud)
    # We only keep column 1 because we need the fraud score.
    scores[name] = model.predict_proba(X_test)[:, 1]


# Build all weight combinations before testing thresholds.
weights_list = list(weight_grid())

# Test thresholds from 0.15 to 0.60.
# A lower threshold usually finds more frauds but creates more false positives.
# A higher threshold usually creates fewer false positives but misses more frauds.
thresholds = [round(0.15 + 0.01 * index, 2) for index in range(46)]

# rows will contain one metrics dictionary for each weight + threshold pair.
rows = []

for weights in weights_list:
    # Weighted average of the three model scores.
    # If one model has a larger weight, it has more influence on the final score.
    blend_score = (
        weights["Random Forest"] * scores["Random Forest"]
        + weights["CatBoost"] * scores["CatBoost"]
        + weights["XGBoost"] * scores["XGBoost"]
    )

    for threshold in thresholds:
        # Convert the blended score into final predictions using this threshold,
        # then calculate metrics.
        row = get_metrics(y_test, blend_score, threshold)

        # Keep the weights inside the row so we know which blend produced it.
        row["weights"] = weights
        rows.append(row)


# Best F1 is the best global precision/recall compromise.
best_f1 = best_by_f1(rows)

# Best low-FN is useful because this project cares a lot about missed frauds.
# max_fp=50 avoids selecting a model that catches many frauds but creates too
# many false alerts.
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
print_row("Blend", best_f1)
print("weights:", best_f1["weights"])

if best_low_fn is not None:
    print("\nBest low-FN with FP <= 50:")
    print_row("Blend", best_low_fn)
    print("weights:", best_low_fn["weights"])
