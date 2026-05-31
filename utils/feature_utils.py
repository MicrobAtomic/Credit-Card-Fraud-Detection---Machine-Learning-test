# Feature helpers for experiments
#
# This file contains helper functions that create new columns for experiments.
# The goal is to keep feature engineering outside the model scripts.

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.model_utils import RANDOM_STATE


def add_isolation_score_feature(X_train, X_test):
    # - Train Isolation Forest only on train data
    # The score is added as a new column for supervised models
    # A higher score means the transaction looks more anomalous / suspicious
    #
    # Important:
    # We fit Isolation Forest only on X_train.
    # Then we use the fitted model to score both X_train and X_test.
    #
    # This avoids data leakage:
    # the test set must not influence the feature creation step.
    isolation_model = make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=200,
            max_samples=4096,
            max_features=0.5,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )

    # Learn the anomaly structure from train data only.
    isolation_model.fit(X_train)

    # Make copies so the original X_train and X_test are not modified.
    X_train_plus = X_train.copy()
    X_test_plus = X_test.copy()

    # decision_function returns higher values for more normal rows.
    # We multiply by -1 so higher isolation_score means "more suspicious".
    X_train_plus["isolation_score"] = -isolation_model.decision_function(X_train)
    X_test_plus["isolation_score"] = -isolation_model.decision_function(X_test)

    # Return the enriched datasets with the new column.
    return X_train_plus, X_test_plus
