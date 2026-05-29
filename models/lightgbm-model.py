# LightGBM model for credit card fraud detection

from pathlib import Path
import sys

from lightgbm import LGBMClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.model_utils import RANDOM_STATE, load_train_test_data, show_results


X_train, X_test, y_train, y_test = load_train_test_data()


# - Creating the model
# LGBMClassifier is the LightGBM classification model
# scale_pos_weight can help when fraud cases are rare
# n_estimators = number of boosting steps
# num_leaves = max number of leaves in one tree
model = LGBMClassifier(
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
)


# - Train the model
# The model only learns from the train data => 80% of the data set
model.fit(X_train, y_train)


# - Test the model
# y_score contains the fraud probability
y_score = model.predict_proba(X_test)[:, 1] # All lines and only the second column => proba fraud [[0.99, 0.01], ...] => [0.01]
THRESHOLD = 0.3
show_results(y_test, y_score, threshold=THRESHOLD)
