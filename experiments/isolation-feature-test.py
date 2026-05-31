# Test Isolation Forest score as an extra feature
#
# Isolation Forest is weak as a final classifier in this project.
# But it may still give useful information:
# - it can score how "unusual" a transaction looks
# - this anomaly score can be added as a new input column
#
# Goal:
# Check if supervised models improve when they receive this extra
# isolation_score feature.

from pathlib import Path
import sys

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# This script lives in experiments/.
# The utils/ folder is one level above, so we add the project root to sys.path.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# add_isolation_score_feature creates one new column:
# - it trains Isolation Forest on X_train
# - it adds an anomaly score to X_train and X_test
#
# The experiment helpers calculate and print metrics.
from utils.feature_utils import add_isolation_score_feature
from utils.experiment_utils import get_metrics, print_row, best_by_f1, best_by_low_fn
from utils.model_utils import RANDOM_STATE, load_train_test_data


# Load the same train/test split as the normal model scripts.
X_train, X_test, y_train, y_test = load_train_test_data()

print("Adding Isolation Forest score as a new feature...")

# X_train_plus and X_test_plus contain all original columns plus isolation_score.
# Important:
# The Isolation Forest feature must be built from train data first.
# Then the same fitted Isolation Forest is used to score test data.
X_train_plus, X_test_plus = add_isolation_score_feature(X_train, X_test)


# Each tuple contains:
# - model name
# - model object
# - thresholds to test
#
# The model parameters are the same as the main scripts.
# Only the input data changes because it now has one extra feature.
models = [
    (
        "Random Forest",
        make_pipeline(
            # Scaling is not required for Random Forest, but is kept for
            # consistency with the other model pipelines.
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=90,
                class_weight={0: 1, 1: 15},
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        # Thresholds around the best Random Forest threshold.
        [0.2, 0.25, 0.3, 0.35, 0.4],
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
        # Thresholds around the best CatBoost threshold.
        [0.3, 0.4, 0.5, 0.6],
    ),
    (
        "XGBoost",
        make_pipeline(
            # Kept to stay close to the main XGBoost script.
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
        # Thresholds around the best XGBoost threshold.
        [0.4, 0.5, 0.6, 0.7],
    ),
    (
        "LightGBM",
        # LightGBM is another tree boosting model.
        # It can use the new isolation_score like any other numeric feature.
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
        # Thresholds around the best LightGBM threshold.
        [0.2, 0.3, 0.4, 0.5],
    ),
]


print("\n=== Isolation feature experiment ===")
print("Same model parameters as the main scripts, only one extra feature is added.")

for name, model, thresholds in models:
    print(f"\nTraining {name}...")

    # Train with the extra isolation_score column.
    model.fit(X_train_plus, y_train)

    # Get fraud probabilities on the test set with the same extra column.
    y_score = model.predict_proba(X_test_plus)[:, 1]

    # Test every threshold listed for this model.
    # Each threshold can change FP/FN a lot.
    rows = [get_metrics(y_test, y_score, threshold) for threshold in thresholds]

    # Best F1 = best overall precision/recall compromise.
    f1_row = best_by_f1(rows)

    # Best low-FN = lowest false negatives while keeping FP reasonable.
    low_fn_row = best_by_low_fn(rows)

    print("Best F1:")
    print_row(name, f1_row)

    if low_fn_row is not None:
        print("Best low-FN with FP <= 50:")
        print_row(name, low_fn_row)
