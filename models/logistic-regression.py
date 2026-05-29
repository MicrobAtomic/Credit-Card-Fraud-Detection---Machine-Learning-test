# Logistic Regression model for credit card fraud detection

from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.model_utils import RANDOM_STATE, load_train_test_data, show_results


X_train, X_test, y_train, y_test = load_train_test_data()


# - Creating the model
# StandardScaler puts the columns on a similar scale
# For each value, it uses this formula: z = (x - mean) / standard_deviation
# After this, the column has mean close to 0 and standard deviation close to 1
# LogisticRegression is the classification model
# class_weight helps because fraud cases are rare
model = make_pipeline( # Making a batch of automatic step
    StandardScaler(),
    LogisticRegression(
        C=0.01, # Small C = model more simple
        class_weight={0: 1, 1: 10},
        max_iter=3000,
        random_state=RANDOM_STATE,
    ),
)


# - Train the model
# The model only learns from the train data => 80% of the data set
model.fit(X_train, y_train)


# - Test the model
# y_score contains the fraud probability
y_score = model.predict_proba(X_test)[:, 1] # All lines and only the second column => proba fraud [[0.99, 0.01], ...] => [0.01]
THRESHOLD = 0.6
show_results(y_test, y_score, threshold=THRESHOLD)
