# Test simple weighted blending between strong models

from pathlib import Path
import sys

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.experiment_utils import get_metrics, print_row, best_by_f1, best_by_low_fn
from utils.model_utils import RANDOM_STATE, load_train_test_data


def weight_grid(step=0.05):
    # Positive weights only, so each model really participates in the blend
    units = int(1 / step)
    for rf_units in range(1, units):
        for cat_units in range(1, units - rf_units):
            xgb_units = units - rf_units - cat_units
            if xgb_units <= 0:
                continue
            yield {
                "Random Forest": round(rf_units * step, 2),
                "CatBoost": round(cat_units * step, 2),
                "XGBoost": round(xgb_units * step, 2),
            }


X_train, X_test, y_train, y_test = load_train_test_data()


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
    ),
]


print("\n=== Simple blending experiment ===")
print("Models: Random Forest + CatBoost + XGBoost")

scores = {}

for name, model in models:
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    scores[name] = model.predict_proba(X_test)[:, 1]


weights_list = list(weight_grid())
thresholds = [round(0.15 + 0.01 * index, 2) for index in range(46)]
rows = []

for weights in weights_list:
    blend_score = (
        weights["Random Forest"] * scores["Random Forest"]
        + weights["CatBoost"] * scores["CatBoost"]
        + weights["XGBoost"] * scores["XGBoost"]
    )

    for threshold in thresholds:
        row = get_metrics(y_test, blend_score, threshold)
        row["weights"] = weights
        rows.append(row)


best_f1 = best_by_f1(rows)
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
