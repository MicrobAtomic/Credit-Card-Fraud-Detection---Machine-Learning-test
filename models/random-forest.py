# Random Forest model for credit card fraud detection

from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.model_utils import RANDOM_STATE, load_train_test_data, show_results


X_train, X_test, y_train, y_test = load_train_test_data()


# - Creating the model
# StandardScaler is kept here to stay close to logistic-regression.py
# RandomForestClassifier is the classification model
# class_weight helps because fraud cases are rare
# n_estimators = number of trees in the forest
# n_jobs=-1 uses all available CPU cores
model = make_pipeline( # Making a batch of automatic step
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=90,
        class_weight={0: 1, 1: 15},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
)


# - Train the model
# The model only learns from the train data => 80% of the data set
model.fit(X_train, y_train)


# - Test the model
# y_score contains the fraud probability
y_score = model.predict_proba(X_test)[:, 1] # All lines and only the second column => proba fraud [[0.99, 0.01], ...] => [0.01]
THRESHOLD = 0.30
show_results(y_test, y_score, threshold=THRESHOLD)
