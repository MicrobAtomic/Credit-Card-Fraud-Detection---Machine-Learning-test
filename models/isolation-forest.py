# Isolation Forest model for credit card fraud detection

from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.model_utils import RANDOM_STATE, load_train_test_data, show_results


X_train, X_test, y_train, y_test = load_train_test_data()


# - Creating the model
# StandardScaler is kept here to stay close to the other model files
# IsolationForest is an anomaly detection model
# It does not learn with y_train like supervised models
# Here we train it on all train transactions as an unsupervised anomaly detector
# n_estimators = number of trees in the forest
# max_samples = number of samples used to train each tree
# max_features = part of columns used by each tree
model = make_pipeline( # Making a batch of automatic step
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


# - Train the model
# The model learns without using y_train
model.fit(X_train)


# - Test the model
# IsolationForest has no fraud probability
# decision_function gives higher values for normal points
# We multiply by -1 so a higher score means more anomalous / more suspicious
y_score = -model.decision_function(X_test)

# Predict the top suspicious transactions as fraud
# TOP_ANOMALY_RATE = 0.002 means top 0.2% most anomalous transactions
TOP_ANOMALY_RATE = 0.002
threshold = np.quantile(y_score, 1 - TOP_ANOMALY_RATE)
y_pred = (y_score >= threshold).astype(int) # 'astype(int)' change true/false in 1/0

show_results(y_test, y_score, y_pred=y_pred)
