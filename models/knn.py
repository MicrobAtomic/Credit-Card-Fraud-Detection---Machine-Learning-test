# KNN model for credit card fraud detection

from pathlib import Path
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.model_utils import RANDOM_STATE, load_train_test_data, show_results


TRAIN_NORMAL_SAMPLE_SIZE = 40000


X_train, X_test, y_train, y_test = load_train_test_data()


# - Make a smaller train set for KNN
# KNN is slow on a big train set because it compares distances with many rows
# Keep all frauds, then add a fixed sample of normal transactions
fraud_index = y_train[y_train == 1].index
normal_index = y_train[y_train == 0].sample(
    n=TRAIN_NORMAL_SAMPLE_SIZE,
    random_state=RANDOM_STATE,
).index
sample_index = normal_index.union(fraud_index)

X_train_small = X_train.loc[sample_index]
y_train_small = y_train.loc[sample_index]


# - Creating the model
# StandardScaler is important for KNN because KNN uses distances
# KNeighborsClassifier compares a transaction with its closest neighbors
# n_neighbors = number of neighbors used to decide
# weights='distance' gives more importance to close neighbors
model = make_pipeline( # Making a batch of automatic step
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=9,
        weights="distance",
        n_jobs=-1,
    ),
)


# - Train the model
# KNN stores the train data and uses it later to compare distances
model.fit(X_train_small, y_train_small)


# - Test the model
# y_score contains the fraud probability
y_score = model.predict_proba(X_test)[:, 1] # All lines and only the second column => proba fraud [[0.99, 0.01], ...] => [0.01]
THRESHOLD = 0.2
show_results(y_test, y_score, threshold=THRESHOLD)
