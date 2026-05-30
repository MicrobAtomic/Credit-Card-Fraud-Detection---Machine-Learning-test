# Test Isolation Forest score as an extra feature

from pathlib import Path
import sys

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.feature_utils import add_isolation_score_feature
from utils.experiment_utils import get_metrics, print_row, best_by_f1, best_by_low_fn
from utils.model_utils import RANDOM_STATE, load_train_test_data


X_train, X_test, y_train, y_test = load_train_test_data()

print("Adding Isolation Forest score as a new feature...")
X_train_plus, X_test_plus = add_isolation_score_feature(X_train, X_test)


models = [
    (
        "Random Forest",
        make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=90,
                class_weight={0: 1, 1: 15},
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        [0.2, 0.25, 0.3, 0.35, 0.4],
    ),
    (
        "CatBoost",
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
        [0.3, 0.4, 0.5, 0.6],
    ),
    (
        "XGBoost",
        make_pipeline(
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
        [0.4, 0.5, 0.6, 0.7],
    ),
    (
        "LightGBM",
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
        [0.2, 0.3, 0.4, 0.5],
    ),
]


print("\n=== Isolation feature experiment ===")
print("Same model parameters as the main scripts, only one extra feature is added.")

for name, model, thresholds in models:
    print(f"\nTraining {name}...")
    model.fit(X_train_plus, y_train)
    y_score = model.predict_proba(X_test_plus)[:, 1]

    rows = [get_metrics(y_test, y_score, threshold) for threshold in thresholds]
    f1_row = best_by_f1(rows)
    low_fn_row = best_by_low_fn(rows)

    print("Best F1:")
    print_row(name, f1_row)

    if low_fn_row is not None:
        print("Best low-FN with FP <= 50:")
        print_row(name, low_fn_row)
