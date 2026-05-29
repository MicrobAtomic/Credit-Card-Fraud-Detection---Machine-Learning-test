# CatBoost model for credit card fraud detection

from pathlib import Path
import sys

from catboost import CatBoostClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.model_utils import RANDOM_STATE, load_train_test_data, show_results


X_train, X_test, y_train, y_test = load_train_test_data() # True to show data info


# - Creating the model
# CatBoost trains many decision trees one after the other
# Each new tree tries to correct the errors made by the previous trees
# class_weights helps because fraud cases are rare
# iterations = number of boosting steps
# depth = max depth of each tree
model = CatBoostClassifier(
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
)


# - Train the model
# The model only learns from the train data => 80% of the data set
model.fit(X_train, y_train)


# - Test the model
# y_score contains the fraud probability
y_score = model.predict_proba(X_test)[:, 1] # All lines and only the second column => proba fraud [[0.99, 0.01], ...] => [0.01]
THRESHOLD = 0.4
show_results(y_test, y_score, threshold=THRESHOLD)
