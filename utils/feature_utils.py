# Feature helpers for experiments

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.model_utils import RANDOM_STATE


def add_isolation_score_feature(X_train, X_test):
    # - Train Isolation Forest only on train data
    # The score is added as a new column for supervised models
    # A higher score means the transaction looks more anomalous / suspicious
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

    isolation_model.fit(X_train)

    X_train_plus = X_train.copy()
    X_test_plus = X_test.copy()

    X_train_plus["isolation_score"] = -isolation_model.decision_function(X_train)
    X_test_plus["isolation_score"] = -isolation_model.decision_function(X_test)

    return X_train_plus, X_test_plus
