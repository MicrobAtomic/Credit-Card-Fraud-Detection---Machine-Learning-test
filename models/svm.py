# SVM model for credit card fraud detection

from pathlib import Path
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.model_utils import RANDOM_STATE, load_train_test_data, show_results


X_train, X_test, y_train, y_test = load_train_test_data()


# - Creating the model
# StandardScaler is important for SVM because SVM uses distances and margins
# LinearSVC is a linear SVM, faster than kernel SVM on a big dataset
# C controls how strict the model is with errors
# class_weight helps because fraud cases are rare
model = make_pipeline( # Making a batch of automatic step
    StandardScaler(),
    LinearSVC(
        C=0.001,
        class_weight={0: 1, 1: 40},
        dual=False,
        max_iter=5000,
        random_state=RANDOM_STATE,
    ),
)


# - Train the model
# The model only learns from the train data => 80% of the data set
model.fit(X_train, y_train)


# - Test the model
# y_score contains the SVM score, not a probability
# A bigger score means the model is more confident about fraud
y_score = model.decision_function(X_test)
THRESHOLD = 0.6
show_results(y_test, y_score, threshold=THRESHOLD)
